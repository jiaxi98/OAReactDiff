"""Lightweight geometry and curvature diagnostics for OA-ReactDiff candidates.

The functions in this module intentionally depend only on NumPy so that existing
XYZ and Hessian artifacts can be audited without installing the historical
PyTorch/PyG training environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ATOMIC_MASSES = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
}

HARTREE_J = 4.3597447222071e-18
BOHR_M = 5.29177210903e-11
ELECTRON_VOLT_J = 1.602176634e-19
ANGSTROM_M = 1.0e-10
ATOMIC_MASS_KG = 1.66053906660e-27
LIGHT_SPEED_CM_S = 2.99792458e10


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read atom symbols and Cartesian coordinates from an XYZ file."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    n_atoms = int(lines[0].strip())
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms in {path}, found {len(atom_lines)}")

    symbols: List[str] = []
    coordinates = np.empty((n_atoms, 3), dtype=float)
    for index, line in enumerate(atom_lines):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ row in {path}: {line!r}")
        symbols.append(fields[0])
        coordinates[index] = [float(value) for value in fields[1:4]]
    return symbols, coordinates


def ordered_kabsch_rmsd(
    reference: np.ndarray,
    candidate: np.ndarray,
    allow_reflection: bool = True,
) -> float:
    """Return Kabsch RMSD for structures with an already matched atom order."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and candidate must both have shape [n_atoms, 3]")

    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=0, keepdims=True)
    covariance = candidate_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if not allow_reflection and np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    aligned = candidate_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))


def _linear_sum_assignment(cost: np.ndarray) -> np.ndarray:
    """Solve a square assignment problem with the Hungarian algorithm."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    size = cost.shape[0]
    row_potential = np.zeros(size + 1)
    col_potential = np.zeros(size + 1)
    matched_row = np.zeros(size + 1, dtype=int)
    predecessor = np.zeros(size + 1, dtype=int)

    for row in range(1, size + 1):
        matched_row[0] = row
        min_cost = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    cost[current_row - 1, candidate_column - 1]
                    - row_potential[current_row]
                    - col_potential[candidate_column]
                )
                if reduced_cost < min_cost[candidate_column]:
                    min_cost[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if min_cost[candidate_column] < delta:
                    delta = min_cost[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    col_potential[candidate_column] -= delta
                else:
                    min_cost[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = np.empty(size, dtype=int)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def _distance_fingerprints(symbols: Sequence[str], coordinates: np.ndarray) -> np.ndarray:
    species = sorted(set(symbols))
    groups = [
        [index for index, symbol in enumerate(symbols) if symbol == species_symbol]
        for species_symbol in species
    ]
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    return np.asarray(
        [
            np.concatenate([np.sort(distances[atom_index, group]) for group in groups])
            for atom_index in range(len(symbols))
        ]
    )


def _kabsch_alignment(
    reference: np.ndarray,
    candidate: np.ndarray,
    allow_reflection: bool,
) -> Tuple[np.ndarray, float]:
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=0, keepdims=True)
    left, _, right_t = np.linalg.svd(candidate_centered.T @ reference_centered)
    rotation = left @ right_t
    if not allow_reflection and np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    aligned = candidate_centered @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))
    return rotation, rmsd


def element_permuted_kabsch_rmsd(
    symbols: Sequence[str],
    reference: np.ndarray,
    candidate: np.ndarray,
    allow_reflection: bool = True,
    max_iterations: int = 20,
) -> float:
    """Approximate element-constrained permutation-invariant Kabsch RMSD.

    Distance fingerprints provide a rotation-invariant initial assignment. The
    correspondence and rigid alignment are then refined together. This avoids a
    heavy Pymatgen dependency while handling the atom-order ambiguity present in
    the bundled OA-ReactDiff demo.
    """
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.shape != (len(symbols), 3):
        raise ValueError("symbols, reference, and candidate have inconsistent shapes")

    reference_fingerprints = _distance_fingerprints(symbols, reference)
    candidate_fingerprints = _distance_fingerprints(symbols, candidate)
    initial_permutation = np.arange(len(symbols))
    for symbol in sorted(set(symbols)):
        indices = np.asarray([index for index, value in enumerate(symbols) if value == symbol])
        cost = np.linalg.norm(
            reference_fingerprints[indices, None, :] - candidate_fingerprints[indices[None, :], :],
            axis=-1,
        )
        initial_permutation[indices] = indices[_linear_sum_assignment(cost)]

    best_rmsd = np.inf
    for starting_permutation in (np.arange(len(symbols)), initial_permutation):
        permutation = starting_permutation.copy()
        for _ in range(max_iterations):
            rotation, rmsd = _kabsch_alignment(reference, candidate[permutation], allow_reflection)
            best_rmsd = min(best_rmsd, rmsd)
            aligned_all = (candidate - candidate.mean(axis=0, keepdims=True)) @ rotation
            reference_centered = reference - reference.mean(axis=0, keepdims=True)
            updated = permutation.copy()
            for symbol in sorted(set(symbols)):
                reference_indices = np.asarray(
                    [index for index, value in enumerate(symbols) if value == symbol]
                )
                candidate_indices = reference_indices
                cost = np.linalg.norm(
                    reference_centered[reference_indices, None, :]
                    - aligned_all[candidate_indices[None, :], :],
                    axis=-1,
                )
                updated[reference_indices] = candidate_indices[_linear_sum_assignment(cost)]
            if np.array_equal(updated, permutation):
                break
            permutation = updated
        _, rmsd = _kabsch_alignment(reference, candidate[permutation], allow_reflection)
        best_rmsd = min(best_rmsd, rmsd)
    return float(best_rmsd)


def canonical_cartesian_hessian(hessian: np.ndarray, n_atoms: int) -> np.ndarray:
    """Convert common PySCF Hessian layouts to a symmetric [3N, 3N] matrix."""
    hessian = np.asarray(hessian, dtype=float)
    if hessian.shape == (n_atoms, n_atoms, 3, 3):
        hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)
    elif hessian.shape == (n_atoms, 3, n_atoms, 3):
        hessian = hessian.reshape(3 * n_atoms, 3 * n_atoms)
    elif hessian.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(f"Unsupported Hessian shape {hessian.shape} for {n_atoms} atoms")
    return 0.5 * (hessian + hessian.T)


def vibrational_basis(symbols: Sequence[str], coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal vibrational basis and per-coordinate square-root masses.

    The basis is expressed in mass-weighted Cartesian coordinates. It removes
    three translations and either two or three rotations, as determined by the
    numerical rank of the rigid-body vectors (linear molecules therefore retain
    ``3N - 5`` vibrational dimensions).
    """
    masses = np.asarray([ATOMIC_MASSES[symbol] for symbol in symbols], dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    center_of_mass = np.average(coordinates, axis=0, weights=masses)
    centered = coordinates - center_of_mass
    sqrt_masses = np.sqrt(masses)

    rigid_vectors = []
    axes = np.eye(3)
    for axis in axes:
        rigid_vectors.append((sqrt_masses[:, None] * axis[None, :]).reshape(-1))
    for axis in axes:
        displacement = np.cross(axis[None, :], centered)
        rigid_vectors.append((sqrt_masses[:, None] * displacement).reshape(-1))

    rigid_matrix = np.column_stack(rigid_vectors)
    left, singular_values, _ = np.linalg.svd(rigid_matrix, full_matrices=True)
    tolerance = np.finfo(float).eps * max(rigid_matrix.shape) * singular_values[0]
    rigid_rank = int(np.sum(singular_values > tolerance))
    return left[:, rigid_rank:], np.repeat(sqrt_masses, 3)


def vibrational_eigenvalues(
    hessian: np.ndarray,
    symbols: Sequence[str],
    coordinates: np.ndarray,
) -> np.ndarray:
    """Return rigid-mode-projected, mass-weighted Hessian eigenvalues."""
    n_atoms = len(symbols)
    if coordinates.shape != (n_atoms, 3):
        raise ValueError("coordinates and symbols have inconsistent sizes")
    cartesian_hessian = canonical_cartesian_hessian(hessian, n_atoms)
    internal_basis, sqrt_mass_vector = vibrational_basis(symbols, coordinates)
    inverse_sqrt_mass = 1.0 / sqrt_mass_vector
    mass_weighted = inverse_sqrt_mass[:, None] * cartesian_hessian * inverse_sqrt_mass[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = internal_basis.T @ mass_weighted @ internal_basis
    if not np.isfinite(projected).all():
        raise ValueError("Projected Hessian contains non-finite values")
    return np.linalg.eigvalsh(0.5 * (projected + projected.T))


def eigenvalues_to_wavenumbers(eigenvalues: np.ndarray) -> np.ndarray:
    """Convert Hartree/(Bohr^2 amu) eigenvalues to signed cm^-1 values."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    angular_frequency_sq = eigenvalues * HARTREE_J / (BOHR_M**2 * ATOMIC_MASS_KG)
    wavenumbers = np.sqrt(np.abs(angular_frequency_sq)) / (2.0 * math.pi * LIGHT_SPEED_CM_S)
    return np.sign(eigenvalues) * wavenumbers


def ev_angstrom_eigenvalues_to_wavenumbers(eigenvalues: np.ndarray) -> np.ndarray:
    """Convert eV/(angstrom^2 amu) eigenvalues to signed cm^-1 values."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    angular_frequency_sq = eigenvalues * ELECTRON_VOLT_J / (ANGSTROM_M**2 * ATOMIC_MASS_KG)
    wavenumbers = np.sqrt(np.abs(angular_frequency_sq)) / (2.0 * math.pi * LIGHT_SPEED_CM_S)
    return np.sign(eigenvalues) * wavenumbers


def projected_mass_weighted_hvp(
    vector: np.ndarray,
    internal_basis: np.ndarray,
    sqrt_mass_vector: np.ndarray,
    cartesian_hvp,
) -> np.ndarray:
    """Apply a rigid-mode-projected, mass-weighted Hessian through a Cartesian HVP."""
    vector = np.asarray(vector, dtype=float)
    internal_basis = np.asarray(internal_basis, dtype=float)
    sqrt_mass_vector = np.asarray(sqrt_mass_vector, dtype=float)
    if vector.shape != (internal_basis.shape[1],):
        raise ValueError("vector size does not match the vibrational basis")
    if sqrt_mass_vector.shape != (internal_basis.shape[0],):
        raise ValueError("mass vector size does not match the vibrational basis")
    cartesian_vector = (internal_basis @ vector) / sqrt_mass_vector
    cartesian_result = np.asarray(cartesian_hvp(cartesian_vector), dtype=float)
    if cartesian_result.shape != cartesian_vector.shape:
        raise ValueError("Cartesian HVP returned an inconsistent shape")
    return internal_basis.T @ (cartesian_result / sqrt_mass_vector)


def _candidate_id(path: Path) -> int:
    match = re.search(r"(?:gen_|candidate_)(\d+)_ts", path.name)
    if match is None:
        raise ValueError(f"Cannot infer candidate id from {path}")
    return int(match.group(1))


def _load_cluster_labels(path: Path) -> Dict[int, str]:
    if not path.is_file():
        return {}
    raw_clusters: Mapping[str, Iterable[str]] = json.loads(path.read_text())
    labels: Dict[int, str] = {}
    for label, members in raw_clusters.items():
        for member in members:
            match = re.search(r"(\d+)\.ts\.opt\.xyz$", member)
            if match is not None:
                labels[int(match.group(1))] = str(label)
    return labels


def _load_candidate_metrics(path: Path) -> Dict[int, Dict[str, float]]:
    if not path.is_file():
        return {}
    metrics: Dict[int, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate_id = int(row["idx"])
            metrics[candidate_id] = {
                "force_rms_ev_per_angstrom": float(row["force_rms"]),
                "reported_imaginary_modes": int(row["imag_freq"]),
            }
    return metrics


def audit_demo_candidates(
    demo_dir: Path,
    imaginary_threshold_cm: float = 50.0,
    force_rms_threshold_ev_per_angstrom: float = 0.05,
    intended_cluster: str = "0",
    allow_reflection: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Audit the precomputed ``demo/example-3`` candidate set."""
    demo_dir = Path(demo_dir)
    reference_path = demo_dir / "ground_truth" / "sample_0_ts.xyz"
    reactant_path = demo_dir / "ground_truth" / "sample_0_react.xyz"
    product_path = demo_dir / "ground_truth" / "sample_0_prod.xyz"
    reference_symbols, reference_coordinates = read_xyz(reference_path)
    cluster_labels = _load_cluster_labels(demo_dir / "cluster.json")
    candidate_metrics = _load_candidate_metrics(demo_dir / "summary.csv")

    candidate_paths = sorted((demo_dir / "generated").glob("gen_*_ts.xyz"), key=_candidate_id)
    rows: List[Dict[str, object]] = []
    for candidate_path in candidate_paths:
        candidate_id = _candidate_id(candidate_path)
        symbols, coordinates = read_xyz(candidate_path)
        if symbols != reference_symbols:
            raise ValueError(f"Atom order/species mismatch for {candidate_path}")

        hessian_path = demo_dir / "hess" / f"hess_{candidate_id}.pkl"
        frequencies = np.asarray([], dtype=float)
        if hessian_path.is_file():
            import pickle

            with hessian_path.open("rb") as handle:
                hessian = pickle.load(handle)
            frequencies = eigenvalues_to_wavenumbers(vibrational_eigenvalues(hessian, symbols, coordinates))

        strict_imaginary_modes = int(np.sum(frequencies < 0.0))
        significant_imaginary_modes = int(np.sum(frequencies < -imaginary_threshold_cm))
        metrics = candidate_metrics.get(candidate_id, {})
        force_rms = metrics.get("force_rms_ev_per_angstrom")
        is_stationary = (
            force_rms <= force_rms_threshold_ev_per_angstrom
            if force_rms is not None
            else None
        )
        cluster = cluster_labels.get(candidate_id, "")
        optimized_path = demo_dir / "opt_ts" / f"{candidate_id}.ts.opt.xyz"
        optimized = optimized_path.is_file()
        rows.append(
            {
                "dataset_index": "demo-example-3",
                "candidate_id": candidate_id,
                "candidate_path": str(candidate_path),
                "reactant_path": str(reactant_path),
                "reference_ts_path": str(reference_path),
                "product_path": str(product_path),
                "ordered_kabsch_rmsd": ordered_kabsch_rmsd(
                    reference_coordinates,
                    coordinates,
                    allow_reflection=allow_reflection,
                ),
                "permutation_kabsch_rmsd": element_permuted_kabsch_rmsd(
                    symbols,
                    reference_coordinates,
                    coordinates,
                    allow_reflection=allow_reflection,
                ),
                "has_hessian": hessian_path.is_file(),
                "min_frequency_cm": float(frequencies.min()) if frequencies.size else "",
                "strict_imaginary_modes": strict_imaginary_modes if frequencies.size else "",
                "is_index_one": strict_imaginary_modes == 1 if frequencies.size else None,
                "significant_imaginary_modes": (
                    significant_imaginary_modes if frequencies.size else ""
                ),
                "has_one_significant_imaginary_mode": (
                    significant_imaginary_modes == 1 if frequencies.size else None
                ),
                "force_rms_ev_per_angstrom": force_rms if force_rms is not None else "",
                "reported_imaginary_modes": metrics.get("reported_imaginary_modes", ""),
                "is_stationary": is_stationary,
                "is_ts_like": (
                    is_stationary and strict_imaginary_modes == 1
                    if is_stationary is not None and frequencies.size
                    else None
                ),
                "optimized": optimized,
                "optimized_cluster": cluster,
                "optimized_to_intended_cluster": (
                    cluster == intended_cluster if optimized and cluster else None
                ),
            }
        )

    rmsds = np.asarray([float(row["permutation_kabsch_rmsd"]) for row in rows])
    force_rms_values = np.asarray(
        [
            float(row["force_rms_ev_per_angstrom"])
            for row in rows
            if row["force_rms_ev_per_angstrom"] != ""
        ]
    )
    cluster_counts = Counter(str(row["optimized_cluster"]) for row in rows if row["optimized_cluster"] != "")
    index_one_ids = {int(row["candidate_id"]) for row in rows if row["is_index_one"]}
    optimized_ids = {int(row["candidate_id"]) for row in rows if row["optimized"]}
    summary: Dict[str, object] = {
        "demo_dir": str(demo_dir),
        "candidate_count": len(rows),
        "hessian_count": sum(bool(row["has_hessian"]) for row in rows),
        "index_one_count": sum(bool(row["is_index_one"]) for row in rows),
        "one_significant_imaginary_mode_count": sum(
            bool(row["has_one_significant_imaginary_mode"]) for row in rows
        ),
        "reported_index_one_count": sum(row["reported_imaginary_modes"] == 1 for row in rows),
        "strict_mode_counts_match_reported": all(
            row["reported_imaginary_modes"] == ""
            or row["strict_imaginary_modes"] == row["reported_imaginary_modes"]
            for row in rows
        ),
        "stationary_count": sum(bool(row["is_stationary"]) for row in rows),
        "stationarity_available_count": sum(row["is_stationary"] is not None for row in rows),
        "ts_like_count": sum(bool(row["is_ts_like"]) for row in rows),
        "ts_like_available_count": sum(row["is_ts_like"] is not None for row in rows),
        "optimized_count": sum(bool(row["optimized"]) for row in rows),
        "index_one_ids_match_optimized_ids": index_one_ids == optimized_ids,
        "optimized_to_intended_cluster_count": sum(bool(row["optimized_to_intended_cluster"]) for row in rows),
        "best_permutation_kabsch_rmsd": float(rmsds.min()) if rmsds.size else None,
        "median_permutation_kabsch_rmsd": float(np.median(rmsds)) if rmsds.size else None,
        "coverage_rmsd_below_0_1": bool(rmsds.size and np.any(rmsds < 0.1)),
        "coverage_rmsd_below_0_2": bool(rmsds.size and np.any(rmsds < 0.2)),
        "imaginary_threshold_cm": imaginary_threshold_cm,
        "force_rms_threshold_ev_per_angstrom": force_rms_threshold_ev_per_angstrom,
        "median_force_rms_ev_per_angstrom": (
            float(np.median(force_rms_values)) if force_rms_values.size else None
        ),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "rmsd_note": (
            "Pure-NumPy iterative element-permutation Kabsch RMSD. Confirm headline values with the repository's "
            "Pymatgen matcher before publication."
        ),
    }
    return rows, summary


def write_audit(rows: Sequence[Mapping[str, object]], summary: Mapping[str, object], output_dir: Path) -> None:
    """Write machine-readable candidate rows and an aggregate summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _first_present(row: Mapping[str, object], names: Sequence[str]) -> Optional[object]:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _optional_float(value: Optional[object]) -> Optional[float]:
    return None if value is None or value == "" else float(value)


def _optional_bool(value: Optional[object]) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _status(value: Optional[bool]) -> str:
    if value is None:
        return "NA"
    return "true" if value else "false"


def summarize_candidate_rows(
    rows: Sequence[Mapping[str, object]],
    rmsd_threshold: float = 0.2,
    force_rms_threshold_ev_per_angstrom: float = 0.05,
    confidence_higher_is_better: bool = True,
) -> List[Dict[str, object]]:
    """Classify per-reaction failure modes while preserving missing evidence as NA."""
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        reaction_id = _first_present(row, ["dataset_index", "reaction_id"])
        grouped.setdefault(str(reaction_id if reaction_id is not None else "demo"), []).append(row)

    summaries: List[Dict[str, object]] = []
    for reaction_id, candidate_rows in grouped.items():
        rmsds = [
            _optional_float(
                _first_present(row, ["paper_style_rmsd", "permutation_kabsch_rmsd"])
            )
            for row in candidate_rows
        ]
        good_indices = [
            index
            for index, rmsd in enumerate(rmsds)
            if rmsd is not None and rmsd < rmsd_threshold
        ]
        if good_indices:
            coverage_failure: Optional[bool] = False
        elif all(rmsd is not None for rmsd in rmsds):
            coverage_failure = True
        else:
            coverage_failure = None

        confidences = [
            _optional_float(_first_present(row, ["confidence"])) for row in candidate_rows
        ]
        selected_index: Optional[int] = None
        ranking_failure: Optional[bool] = None
        if coverage_failure is False and all(value is not None for value in confidences):
            direction = 1.0 if confidence_higher_is_better else -1.0
            selected_index = max(
                range(len(candidate_rows)),
                key=lambda index: direction * float(confidences[index]),
            )
            if rmsds[selected_index] is not None:
                ranking_failure = bool(rmsds[selected_index] >= rmsd_threshold)

        geometry_physics_bad: Optional[bool]
        if coverage_failure is True:
            geometry_physics_bad = False
        elif coverage_failure is None:
            geometry_physics_bad = None
        else:
            physics_states: List[Optional[bool]] = []
            for index in good_indices:
                row = candidate_rows[index]
                stationary = _optional_bool(_first_present(row, ["is_stationary"]))
                if stationary is None:
                    force_rms = _optional_float(
                        _first_present(
                            row,
                            ["force_rms_ev_per_angstrom", "force_norm"],
                        )
                    )
                    if force_rms is not None:
                        stationary = force_rms <= force_rms_threshold_ev_per_angstrom

                index_one = _optional_bool(_first_present(row, ["is_index_one"]))
                if index_one is None:
                    mode_count = _optional_float(
                        _first_present(
                            row,
                            ["strict_imaginary_modes", "negative_modes"],
                        )
                    )
                    if mode_count is not None:
                        index_one = mode_count == 1

                if stationary is False or index_one is False:
                    physics_states.append(False)
                elif stationary is True and index_one is True:
                    physics_states.append(True)
                else:
                    physics_states.append(None)
            if any(state is False for state in physics_states):
                geometry_physics_bad = True
            elif physics_states and all(state is True for state in physics_states):
                geometry_physics_bad = False
            else:
                geometry_physics_bad = None

        irc_results = [
            value
            for value in (
                _optional_bool(_first_present(row, ["irc_connects_target"]))
                for row in candidate_rows
            )
            if value is not None
        ]
        if irc_results:
            alternate_saddle: Optional[bool] = any(value is False for value in irc_results)
            alternate_saddle_evidence = "irc"
        else:
            cluster_results = []
            for row in candidate_rows:
                if _optional_bool(_first_present(row, ["optimized"])) is not True:
                    continue
                cluster_result = _optional_bool(
                    _first_present(row, ["optimized_to_intended_cluster"])
                )
                if cluster_result is not None:
                    cluster_results.append(cluster_result)
            alternate_saddle = (
                any(value is False for value in cluster_results) if cluster_results else None
            )
            alternate_saddle_evidence = "optimized_cluster" if cluster_results else "NA"

        selected_row = candidate_rows[selected_index] if selected_index is not None else None
        selected_candidate_id = (
            _first_present(selected_row, ["candidate_id"]) if selected_row is not None else ""
        )
        summaries.append(
            {
                "reaction_id": reaction_id,
                "candidate_count": len(candidate_rows),
                "best_rmsd": min((value for value in rmsds if value is not None), default=""),
                "coverage_failure": _status(coverage_failure),
                "ranking_failure": _status(ranking_failure),
                "selected_candidate_id": selected_candidate_id,
                "geometry_good_physics_bad": _status(geometry_physics_bad),
                "alternate_saddle": _status(alternate_saddle),
                "alternate_saddle_evidence": alternate_saddle_evidence,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit precomputed OA-ReactDiff demo candidates.")
    parser.add_argument("--demo-dir", type=Path, default=Path("demo/example-3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imaginary-threshold-cm", type=float, default=50.0)
    parser.add_argument("--force-rms-threshold-ev-per-angstrom", type=float, default=0.05)
    parser.add_argument("--intended-cluster", default="0")
    parser.add_argument("--disallow-reflection", action="store_true")
    args = parser.parse_args()

    rows, summary = audit_demo_candidates(
        demo_dir=args.demo_dir,
        imaginary_threshold_cm=args.imaginary_threshold_cm,
        force_rms_threshold_ev_per_angstrom=args.force_rms_threshold_ev_per_angstrom,
        intended_cluster=args.intended_cluster,
        allow_reflection=not args.disallow_reflection,
    )
    write_audit(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
