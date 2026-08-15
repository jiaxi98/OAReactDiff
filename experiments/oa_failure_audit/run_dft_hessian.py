#!/usr/bin/env python3
"""Run resumable wB97X/6-31G(d) force and Hessian calculations on a DFT subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from oa_reactdiff.analyze.failure_audit import (
    ANGSTROM_M,
    BOHR_M,
    ELECTRON_VOLT_J,
    HARTREE_J,
    eigenvalues_to_wavenumbers,
    read_xyz,
    vibrational_eigenvalues,
)


IDENTITY_FIELDS = [
    "dft_selection_rank",
    "dft_stratum",
    "dataset_index",
    "source_reaction_index",
    "candidate_id",
    "candidate_path",
    "paper_style_rmsd",
    "permutation_kabsch_rmsd",
]
RESULT_FIELDS = IDENTITY_FIELDS + [
    "model_label",
    "backend",
    "pyscf_version",
    "gpu4pyscf_version",
    "xc",
    "basis",
    "molecular_charge",
    "spin",
    "scf_converged",
    "energy_hartree",
    "force_rms_ev_per_angstrom",
    "lowest_frequency_cm",
    "second_frequency_cm",
    "strict_imaginary_modes",
    "significant_imaginary_modes",
    "is_stationary",
    "is_index_one",
    "is_ts_like",
    "artifact_path",
    "wall_seconds",
    "error",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["pyscf", "gpu4pyscf"], default="pyscf")
    parser.add_argument("--xc", default="wb97x")
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=nonnegative_int, default=0, help="PySCF spin: number of unpaired electrons.")
    parser.add_argument("--conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--conv-tol-grad", type=float, default=1.0e-5)
    parser.add_argument("--conv-tol-cpscf", type=float, default=1.0e-6)
    parser.add_argument("--grid-level", type=nonnegative_int, default=3)
    parser.add_argument("--max-cycle", type=positive_int, default=200)
    parser.add_argument("--max-memory-mb", type=positive_int, default=12000)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--frequency-threshold-cm", type=float, default=50.0)
    parser.add_argument("--force-rms-threshold", type=float, default=0.05)
    parser.add_argument("--start-rank", type=positive_int, default=1)
    parser.add_argument("--max-candidates", type=positive_int)
    parser.add_argument("--ranks", nargs="+", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    possibilities = (
        [candidate]
        if candidate.is_absolute()
        else [manifest_path.parent / candidate, Path.cwd() / candidate]
    )
    for possibility in possibilities:
        if possibility.is_file():
            return possibility.resolve()
    raise FileNotFoundError(f"Cannot resolve candidate path {raw_path!r}")


def selection_rank(row: Mapping[str, str], ordinal: int) -> int:
    value = row.get("dft_selection_rank", "")
    return int(value) if value else ordinal


def select_rows(args: argparse.Namespace, rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    ranked = [(selection_rank(row, ordinal), row) for ordinal, row in enumerate(rows, start=1)]
    if len({rank for rank, _ in ranked}) != len(ranked):
        raise ValueError("DFT selection ranks must be unique")
    if args.ranks is not None:
        if args.start_rank != 1:
            raise ValueError("--ranks and a non-default --start-rank cannot be used together")
        if len(set(args.ranks)) != len(args.ranks):
            raise ValueError("--ranks values must be unique")
        requested = set(args.ranks)
        selected = [row for rank, row in ranked if rank in requested]
        missing = requested.difference(rank for rank, _ in ranked)
        if missing:
            raise IndexError(f"DFT selection ranks are absent: {sorted(missing)}")
    else:
        selected = [row for rank, row in ranked if rank >= args.start_rank]
    if args.max_candidates is not None:
        selected = selected[: args.max_candidates]
    if not selected:
        raise ValueError("No DFT candidates were selected")
    return selected


def candidate_key(row: Mapping[str, object]) -> Tuple[str, str]:
    return str(row.get("dataset_index", "demo")), str(row["candidate_id"])


def identity_values(row: Mapping[str, object]) -> Dict[str, object]:
    return {field: row.get(field, "") for field in IDENTITY_FIELDS}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def row_charge_and_spin(row: Mapping[str, str], args: argparse.Namespace) -> Tuple[int, int]:
    raw_charge = row.get("molecular_charge", "")
    raw_spin = row.get("spin", "")
    return (
        int(raw_charge) if raw_charge != "" else args.charge,
        int(raw_spin) if raw_spin != "" else args.spin,
    )


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def scalar_float(value) -> float:
    array = to_numpy(value)
    return float(array.reshape(-1)[0])


def write_npz_atomic(path: Path, **arrays) -> None:
    temporary_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(path)


def comparable_settings(settings: Mapping[str, object]) -> Dict[str, object]:
    ignored = {"hostname", "platform"}
    return {key: value for key, value in settings.items() if key not in ignored}


def prepare_output(output_dir: Path, settings: Mapping[str, object], resume: bool) -> List[Dict[str, str]]:
    settings_path = output_dir / "settings.json"
    results_path = output_dir / "dft_results.csv"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory is not empty; pass --resume: {output_dir}")
        if not settings_path.is_file() or not results_path.is_file():
            raise FileNotFoundError("Resume requires settings.json and dft_results.csv")
        previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if comparable_settings(previous_settings) != comparable_settings(settings):
            raise ValueError("Resume settings differ from the existing settings.json")
        return read_csv(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(settings_path, settings)
    write_csv_atomic(results_path, [])
    return []


def make_mean_field(symbols, coordinates, charge, spin, args, log_path, pyscf_modules):
    dft_module, gto_module = pyscf_modules
    molecule = gto_module.Mole()
    molecule.atom = [(symbol, tuple(coordinate)) for symbol, coordinate in zip(symbols, coordinates)]
    molecule.basis = args.basis
    molecule.charge = charge
    molecule.spin = spin
    molecule.unit = "Angstrom"
    molecule.max_memory = args.max_memory_mb
    molecule.output = str(log_path)
    molecule.verbose = 4
    molecule.build()

    mean_field = dft_module.RKS(molecule) if spin == 0 else dft_module.UKS(molecule)
    mean_field.xc = args.xc
    mean_field.conv_tol = args.conv_tol
    mean_field.conv_tol_grad = args.conv_tol_grad
    mean_field.max_cycle = args.max_cycle
    mean_field.max_memory = args.max_memory_mb
    mean_field.grids.level = args.grid_level
    if args.backend == "gpu4pyscf":
        mean_field = mean_field.to_gpu()
    return molecule, mean_field


def run_candidate(row, subset_path, output_dir, args, versions, pyscf_modules) -> Dict[str, object]:
    output = identity_values(row)
    metric_fields = set(IDENTITY_FIELDS)
    output.update({field: "" for field in RESULT_FIELDS if field not in metric_fields})
    output.update(
        {
            "model_label": "dft_reference",
            "backend": args.backend,
            "pyscf_version": versions["pyscf_version"],
            "gpu4pyscf_version": versions["gpu4pyscf_version"],
            "xc": args.xc,
            "basis": args.basis,
        }
    )
    start_time = time.perf_counter()
    molecule = None
    try:
        candidate_path = resolve_manifest_path(row["candidate_path"], subset_path)
        output["candidate_path"] = Path(
            os.path.relpath(candidate_path, output_dir.resolve())
        ).as_posix()
        symbols, coordinates = read_xyz(candidate_path)
        charge, spin = row_charge_and_spin(row, args)
        output["molecular_charge"] = charge
        output["spin"] = spin
        rank = row.get("dft_selection_rank", "") or "unranked"
        artifact_dir = output_dir / (
            f"rank_{safe_component(rank)}_reaction_{safe_component(row.get('dataset_index', 'demo'))}_"
            f"candidate_{safe_component(row['candidate_id'])}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        molecule, mean_field = make_mean_field(
            symbols,
            coordinates,
            charge,
            spin,
            args,
            artifact_dir / "pyscf.log",
            pyscf_modules,
        )
        energy = mean_field.kernel()
        converged = bool(mean_field.converged)
        output["scf_converged"] = bool_text(converged)
        output["energy_hartree"] = scalar_float(energy)
        if not converged:
            raise RuntimeError("SCF did not converge; Hessian was not attempted")

        gradient = to_numpy(mean_field.nuc_grad_method().kernel())
        hartree_per_bohr_to_ev_per_angstrom = HARTREE_J / ELECTRON_VOLT_J * ANGSTROM_M / BOHR_M
        forces = -gradient * hartree_per_bohr_to_ev_per_angstrom
        force_rms = float(np.sqrt(np.mean(forces**2)))

        hessian_object = mean_field.Hessian()
        hessian_object.conv_tol_cpscf = args.conv_tol_cpscf
        hessian = to_numpy(hessian_object.kernel())
        eigenvalues = vibrational_eigenvalues(hessian, symbols, coordinates)
        frequencies = eigenvalues_to_wavenumbers(eigenvalues)
        strict_modes = int(np.sum(frequencies < 0.0))
        significant_modes = int(np.sum(frequencies < -args.frequency_threshold_cm))
        stationary = force_rms <= args.force_rms_threshold
        index_one = strict_modes == 1
        artifact_path = artifact_dir / "dft_efh.npz"
        write_npz_atomic(
            artifact_path,
            coordinates_angstrom=coordinates,
            eigenvalues_hartree_per_bohr2_amu=eigenvalues,
            energy_hartree=np.asarray([scalar_float(energy)]),
            forces_ev_per_angstrom=forces,
            frequencies_cm=frequencies,
            hessian_hartree_per_bohr2=hessian,
            symbols=np.asarray(symbols),
        )
        output.update(
            {
                "force_rms_ev_per_angstrom": force_rms,
                "lowest_frequency_cm": float(frequencies[0]),
                "second_frequency_cm": float(frequencies[1]),
                "strict_imaginary_modes": strict_modes,
                "significant_imaginary_modes": significant_modes,
                "is_stationary": bool_text(stationary),
                "is_index_one": bool_text(index_one),
                "is_ts_like": bool_text(stationary and index_one),
                "artifact_path": artifact_path.relative_to(output_dir).as_posix(),
            }
        )
    except Exception as error:  # Preserve completed cases when a long DFT batch encounters one failure.
        output["error"] = f"{type(error).__name__}: {error}"
    finally:
        if molecule is not None and hasattr(molecule, "stdout"):
            molecule.stdout.flush()
            molecule.stdout.close()
    output["wall_seconds"] = time.perf_counter() - start_time
    return output


def main() -> None:
    args = parse_args()
    if not args.subset.is_file():
        raise FileNotFoundError(args.subset)
    if args.conv_tol <= 0 or args.conv_tol_grad <= 0 or args.conv_tol_cpscf <= 0:
        raise ValueError("SCF and CPSCF tolerances must be positive")
    if args.frequency_threshold_cm < 0 or args.force_rms_threshold < 0:
        raise ValueError("Frequency and force thresholds must be non-negative")
    rows = read_csv(args.subset)
    selected_rows = select_rows(args, rows)
    for row in selected_rows:
        resolve_manifest_path(row["candidate_path"], args.subset)
    base_settings: Dict[str, object] = {
        "backend": args.backend,
        "basis": args.basis,
        "charge_default": args.charge,
        "conv_tol": args.conv_tol,
        "conv_tol_cpscf": args.conv_tol_cpscf,
        "conv_tol_grad": args.conv_tol_grad,
        "force_rms_threshold_ev_per_angstrom": args.force_rms_threshold,
        "frequency_threshold_cm": args.frequency_threshold_cm,
        "grid_level": args.grid_level,
        "hessian_grid_response": False,
        "max_cycle": args.max_cycle,
        "max_memory_mb": args.max_memory_mb,
        "selected_candidate_keys": [list(candidate_key(row)) for row in selected_rows],
        "spin_default": args.spin,
        "subset": str(args.subset.resolve()),
        "subset_sha256": sha256_file(args.subset),
        "threads": args.threads,
        "xc": args.xc,
    }
    if args.dry_run:
        print(json.dumps({**base_settings, "candidate_count": len(selected_rows)}, indent=2, sort_keys=True))
        return

    import pyscf
    from pyscf import dft, gto, lib

    lib.num_threads(args.threads)
    gpu4pyscf_version: Optional[str] = None
    if args.backend == "gpu4pyscf":
        import gpu4pyscf

        gpu4pyscf_version = getattr(gpu4pyscf, "__version__", None)
        if gpu4pyscf_version is None:
            for distribution_name in ("gpu4pyscf", "gpu4pyscf-cuda11x", "gpu4pyscf-cuda12x"):
                try:
                    gpu4pyscf_version = importlib.metadata.version(distribution_name)
                    break
                except importlib.metadata.PackageNotFoundError:
                    continue
    versions = {
        "gpu4pyscf_version": gpu4pyscf_version,
        "pyscf_version": pyscf.__version__,
    }
    settings = {
        **base_settings,
        **versions,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    existing_rows = prepare_output(args.output_dir, settings, args.resume)
    result_by_key = {candidate_key(row): row for row in existing_rows}
    completed_keys = {candidate_key(row) for row in existing_rows if not row.get("error")}
    results_path = args.output_dir / "dft_results.csv"

    for ordinal, row in enumerate(selected_rows, start=1):
        key = candidate_key(row)
        if key in completed_keys:
            print(f"[{ordinal}/{len(selected_rows)}] {key}: already complete", flush=True)
            continue
        result = run_candidate(row, args.subset, args.output_dir, args, versions, (dft, gto))
        result_by_key[key] = result
        ordered = [
            result_by_key[candidate_key(source_row)]
            for source_row in selected_rows
            if candidate_key(source_row) in result_by_key
        ]
        write_csv_atomic(results_path, ordered)
        status = "ok" if not result["error"] else result["error"]
        print(f"[{ordinal}/{len(selected_rows)}] {key}: {status} ({float(result['wall_seconds']):.1f}s)", flush=True)

    final_rows = [result_by_key[candidate_key(row)] for row in selected_rows]
    write_csv_atomic(results_path, final_rows)
    successful = sum(not row["error"] for row in final_rows)
    print(f"Wrote {successful}/{len(final_rows)} successful DFT Hessians to {args.output_dir}")


if __name__ == "__main__":
    main()
