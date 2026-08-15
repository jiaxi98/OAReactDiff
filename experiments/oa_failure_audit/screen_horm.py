#!/usr/bin/env python3
"""Screen generated TS candidates with a verified HORM LEFTNet checkpoint.

Run this script once for ``left_orig.ckpt`` (E/F training) and once for
``left.ckpt`` (E/F/H training). It computes conservative forces and the two
lowest rigid-mode-projected, mass-weighted Hessian eigenpairs through exact
autograd Hessian-vector products; it never materializes a full Hessian.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from oa_reactdiff.analyze.failure_audit import (
    ev_angstrom_eigenvalues_to_wavenumbers,
    projected_mass_weighted_hvp,
    read_xyz,
    vibrational_basis,
)


PINNED_HORM_COMMIT = "b4c2a35a28985c72ca47261bad0a96b2bc2ba084"
KNOWN_CHECKPOINT_SHA256 = {
    "left.ckpt": "55b1f2d21897ad4f7870986397ed981185989fc947f08aff172c65cd41a1f2a0",
    "left_orig.ckpt": "1c286d36152781d1923cf6ab778d2f5227cf8bb626e07604dc8b86f15a6ac6fa",
}
ATOM_TO_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8}
ATOM_TO_ONE_HOT_INDEX = {"H": 0, "C": 1, "N": 2, "O": 3}
IDENTITY_FIELDS = [
    "dataset_index",
    "source_reaction_index",
    "candidate_id",
    "candidate_path",
    "paper_style_rmsd",
    "permutation_kabsch_rmsd",
]
RESULT_FIELDS = IDENTITY_FIELDS + [
    "model_label",
    "checkpoint_sha256",
    "energy_ev",
    "force_rms_ev_per_angstrom",
    "lowest_eigenvalue_ev_per_angstrom2_amu",
    "second_eigenvalue_ev_per_angstrom2_amu",
    "lowest_frequency_cm",
    "second_frequency_cm",
    "strict_negative_mode_class",
    "significant_negative_mode_class",
    "is_stationary",
    "has_one_significant_imaginary_mode",
    "is_ts_like",
    "eigensolver_residual_1",
    "eigensolver_residual_2",
    "hvp_calls",
    "wall_seconds",
    "error",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horm-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Stable output prefix, e.g. horm_left_efh.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-candidates", type=positive_int)
    parser.add_argument("--frequency-threshold-cm", type=float, default=50.0)
    parser.add_argument("--force-rms-threshold", type=float, default=0.05)
    parser.add_argument("--eigsh-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--eigsh-maxiter", type=positive_int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-horm-commit", default=PINNED_HORM_COMMIT)
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    parser.add_argument("--allow-unpinned-horm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest, repository revision, and checkpoint without importing PyTorch.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    raise FileNotFoundError(f"Cannot resolve {raw_path!r} relative to {manifest_path.parent} or the working directory")


def candidate_key(row: Mapping[str, object]) -> Tuple[str, str]:
    dataset_index = row.get("dataset_index", "demo")
    return str(dataset_index), str(row["candidate_id"])


def identity_values(row: Mapping[str, object]) -> Dict[str, object]:
    return {field: row.get(field, "") for field in IDENTITY_FIELDS}


def mode_class(frequencies: np.ndarray, threshold_cm: float) -> str:
    """Classify a negative-mode count from the two algebraically lowest modes."""
    negative_count = int(np.sum(frequencies < -threshold_cm))
    return ">=2" if negative_count >= 2 else str(negative_count)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def validate_inputs(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.horm_repo.is_dir():
        raise FileNotFoundError(args.horm_repo)
    if not (args.horm_repo / "training_module.py").is_file():
        raise FileNotFoundError(f"Not a HORM checkout: {args.horm_repo}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.frequency_threshold_cm < 0:
        raise ValueError("--frequency-threshold-cm must be non-negative")
    if args.force_rms_threshold < 0:
        raise ValueError("--force-rms-threshold must be non-negative")
    if args.eigsh_tolerance <= 0:
        raise ValueError("--eigsh-tolerance must be positive")

    rows = read_csv(args.manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {args.manifest}")
    required = {"candidate_id", "candidate_path"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if len({candidate_key(row) for row in rows}) != len(rows):
        raise ValueError("Manifest candidate keys are not unique")
    if args.max_candidates is not None:
        rows = rows[: args.max_candidates]
    for row in rows:
        resolve_manifest_path(row["candidate_path"], args.manifest)

    revision = git_revision(args.horm_repo)
    if revision != args.expected_horm_commit and not args.allow_unpinned_horm:
        raise ValueError(
            f"HORM revision is {revision}; expected {args.expected_horm_commit}. "
            "Use --allow-unpinned-horm only for an intentional comparison."
        )

    checkpoint_sha256 = sha256_file(args.checkpoint)
    expected_sha256 = args.expected_checkpoint_sha256 or KNOWN_CHECKPOINT_SHA256.get(args.checkpoint.name)
    if expected_sha256 is None and not args.allow_unverified_checkpoint:
        raise ValueError("No known checkpoint checksum; pass --expected-checkpoint-sha256")
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: found {checkpoint_sha256}, expected {expected_sha256}")

    settings: Dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "eigsh_maxiter": args.eigsh_maxiter,
        "eigsh_tolerance": args.eigsh_tolerance,
        "force_rms_threshold_ev_per_angstrom": args.force_rms_threshold,
        "frequency_threshold_cm": args.frequency_threshold_cm,
        "horm_commit": revision,
        "horm_repo": str(args.horm_repo.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "max_candidates": args.max_candidates,
        "model_label": args.label,
        "requested_device": args.device,
        "seed": args.seed,
    }
    return rows, settings


def settings_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".settings.json")


def comparable_settings(settings: Mapping[str, object]) -> Dict[str, object]:
    ignored = {"gpu_name", "resolved_device"}
    return {key: value for key, value in settings.items() if key not in ignored}


def prepare_output(output_path: Path, settings: Mapping[str, object], resume: bool) -> List[Dict[str, str]]:
    settings_path = settings_path_for(output_path)
    if output_path.exists() or settings_path.exists():
        if not resume:
            raise FileExistsError(f"Output already exists; pass --resume or choose another path: {output_path}")
        if not output_path.is_file() or not settings_path.is_file():
            raise FileNotFoundError("Resume requires both the output CSV and its settings JSON")
        previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if comparable_settings(previous_settings) != comparable_settings(settings):
            raise ValueError("Resume settings differ from the existing settings JSON")
        return read_csv(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(settings_path, settings)
    write_csv_atomic(output_path, [])
    return []


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch_module.device(requested)


def synchronize_cuda(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def make_horm_batch(symbols, coordinates, torch_module, data_class, device):
    unsupported = sorted(set(symbols).difference(ATOM_TO_NUMBER))
    if unsupported:
        raise ValueError(f"HORM LEFTNet supports CHON only; found {unsupported}")
    atomic_numbers = [ATOM_TO_NUMBER[symbol] for symbol in symbols]
    one_hot_indices = [ATOM_TO_ONE_HOT_INDEX[symbol] for symbol in symbols]
    one_hot = torch_module.zeros((len(symbols), 5), dtype=torch_module.int64, device=device)
    one_hot[torch_module.arange(len(symbols), device=device), one_hot_indices] = 1
    return data_class(
        ae=torch_module.zeros(1, dtype=torch_module.float32, device=device),
        batch=torch_module.zeros(len(symbols), dtype=torch_module.int64, device=device),
        charges=torch_module.tensor(atomic_numbers, dtype=torch_module.int64, device=device),
        natoms=torch_module.tensor([len(symbols)], dtype=torch_module.int64, device=device),
        one_hot=one_hot,
        pos=torch_module.tensor(coordinates, dtype=torch_module.float32, device=device),
    )


def two_lowest_eigenpairs(matvec, dimension: int, tolerance: float, maxiter: int, seed: int):
    from scipy.sparse.linalg import LinearOperator, eigsh

    if dimension < 2:
        raise ValueError(f"Need at least two vibrational degrees of freedom; found {dimension}")
    if dimension <= 3:
        dense = np.column_stack([matvec(np.eye(dimension)[:, index]) for index in range(dimension)])
        dense = 0.5 * (dense + dense.T)
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        return eigenvalues[:2], eigenvectors[:, :2]
    operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=np.float64)
    initial_vector = np.random.default_rng(seed).normal(size=dimension)
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=2,
        which="SA",
        v0=initial_vector,
        tol=tolerance,
        maxiter=maxiter,
    )
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def screen_candidate(
    row: Mapping[str, str],
    manifest_path: Path,
    potential,
    torch_module,
    data_class,
    device,
    checkpoint_sha256: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    metric_fields = set(IDENTITY_FIELDS + ["model_label", "checkpoint_sha256"])
    output = identity_values(row)
    output.update(
        {
            "model_label": args.label,
            "checkpoint_sha256": checkpoint_sha256,
            **{field: "" for field in RESULT_FIELDS if field not in metric_fields},
        }
    )
    start_time = time.perf_counter()
    hvp_calls = 0
    try:
        candidate_path = resolve_manifest_path(row["candidate_path"], manifest_path)
        output["candidate_path"] = Path(
            os.path.relpath(candidate_path, args.output.parent.resolve())
        ).as_posix()
        symbols, coordinates = read_xyz(candidate_path)
        internal_basis, sqrt_mass_vector = vibrational_basis(symbols, coordinates)
        batch = make_horm_batch(symbols, coordinates, torch_module, data_class, device)

        synchronize_cuda(torch_module, device)
        with torch_module.enable_grad():
            energy, forces = potential.forward_autograd(batch)

            def cartesian_hvp(vector: np.ndarray) -> np.ndarray:
                nonlocal hvp_calls
                hvp_calls += 1
                torch_vector = torch_module.as_tensor(
                    vector.reshape(len(symbols), 3),
                    dtype=batch.pos.dtype,
                    device=device,
                )
                result = -torch_module.autograd.grad(
                    forces,
                    batch.pos,
                    grad_outputs=torch_vector,
                    retain_graph=True,
                )[0]
                return result.detach().cpu().numpy().reshape(-1)

            def projected_matvec(vector: np.ndarray) -> np.ndarray:
                return projected_mass_weighted_hvp(
                    vector,
                    internal_basis,
                    sqrt_mass_vector,
                    cartesian_hvp,
                )

            eigenvalues, eigenvectors = two_lowest_eigenpairs(
                projected_matvec,
                internal_basis.shape[1],
                args.eigsh_tolerance,
                args.eigsh_maxiter,
                args.seed + int(row.get("candidate_id", 0)),
            )
            residuals = np.asarray(
                [
                    np.linalg.norm(
                        projected_matvec(eigenvectors[:, index])
                        - eigenvalues[index] * eigenvectors[:, index]
                    )
                    for index in range(2)
                ]
            )

        synchronize_cuda(torch_module, device)
        frequencies = ev_angstrom_eigenvalues_to_wavenumbers(eigenvalues)
        force_rms = float(torch_module.sqrt(torch_module.mean(forces.detach() ** 2)).cpu())
        significant_class = mode_class(frequencies, args.frequency_threshold_cm)
        stationary = force_rms <= args.force_rms_threshold
        one_significant_mode = significant_class == "1"
        output.update(
            {
                "energy_ev": float(energy.detach().reshape(-1)[0].cpu()),
                "force_rms_ev_per_angstrom": force_rms,
                "lowest_eigenvalue_ev_per_angstrom2_amu": float(eigenvalues[0]),
                "second_eigenvalue_ev_per_angstrom2_amu": float(eigenvalues[1]),
                "lowest_frequency_cm": float(frequencies[0]),
                "second_frequency_cm": float(frequencies[1]),
                "strict_negative_mode_class": mode_class(frequencies, 0.0),
                "significant_negative_mode_class": significant_class,
                "is_stationary": bool_text(stationary),
                "has_one_significant_imaginary_mode": bool_text(one_significant_mode),
                "is_ts_like": bool_text(stationary and one_significant_mode),
                "eigensolver_residual_1": float(residuals[0]),
                "eigensolver_residual_2": float(residuals[1]),
                "hvp_calls": hvp_calls,
            }
        )
    except Exception as error:  # Keep long screens resumable and surface per-candidate failures.
        output["error"] = f"{type(error).__name__}: {error}"
        output["hvp_calls"] = hvp_calls
    output["wall_seconds"] = time.perf_counter() - start_time
    return output


def main() -> None:
    args = parse_args()
    rows, settings = validate_inputs(args)
    if args.dry_run:
        print(json.dumps({**settings, "candidate_count": len(rows)}, indent=2, sort_keys=True))
        return

    import torch
    from torch_geometric.data import Data

    sys.path.insert(0, str(args.horm_repo.resolve()))
    from training_module import PotentialModule

    device = resolve_device(torch, args.device)
    settings.update(
        {
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python_version": platform.python_version(),
            "resolved_device": str(device),
            "torch_version": torch.__version__,
        }
    )
    existing_rows = prepare_output(args.output, settings, args.resume)
    result_by_key = {candidate_key(row): row for row in existing_rows}
    completed_keys = {candidate_key(row) for row in existing_rows if not row.get("error")}

    module = PotentialModule.load_from_checkpoint(str(args.checkpoint), map_location=device, strict=False)
    potential = module.potential.to(device).eval()
    potential.requires_grad_(False)

    for ordinal, row in enumerate(rows, start=1):
        key = candidate_key(row)
        if key in completed_keys:
            print(f"[{ordinal}/{len(rows)}] {key}: already complete", flush=True)
            continue
        result = screen_candidate(
            row,
            args.manifest,
            potential,
            torch,
            Data,
            device,
            str(settings["checkpoint_sha256"]),
            args,
        )
        result_by_key[key] = result
        ordered_results = [
            result_by_key[candidate_key(source_row)]
            for source_row in rows
            if candidate_key(source_row) in result_by_key
        ]
        write_csv_atomic(args.output, ordered_results)
        status = "ok" if not result["error"] else result["error"]
        print(f"[{ordinal}/{len(rows)}] {key}: {status} ({float(result['wall_seconds']):.2f}s)", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_results = [result_by_key[candidate_key(row)] for row in rows]
    write_csv_atomic(args.output, final_results)
    successful = sum(not row["error"] for row in final_results)
    print(f"Wrote {successful}/{len(final_results)} successful screens to {args.output}")


if __name__ == "__main__":
    main()
