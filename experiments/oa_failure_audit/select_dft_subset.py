#!/usr/bin/env python3
"""Select a deterministic, reaction-capped subset for DFT Hessian calculations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple


SELECTION_FIELDS = [
    "dft_selection_rank",
    "dft_stratum",
    "dft_rmsd_bin",
    "dft_force_bin",
    "dft_selection_reason",
]
STRATUM_ORDER = [
    "ef_vs_efh_curvature_disagreement",
    "coverage_failure_best",
    "geometry_good_physics_bad",
    "geometry_bad_index_one",
    "efh_index_one",
    "efh_multimode",
    "efh_minimum_like",
    "screen_failure",
    "unclassified",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=positive_int, default=96)
    parser.add_argument("--max-per-reaction", type=positive_int, default=2)
    parser.add_argument("--ef-prefix", default="horm_left_ef")
    parser.add_argument("--efh-prefix", default="horm_left_efh")
    parser.add_argument("--rmsd-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def rebase_path_fields(
    rows: Sequence[Mapping[str, str]],
    source_csv: Path,
    output_csv: Path,
) -> List[Dict[str, str]]:
    rebased_rows = []
    for source_row in rows:
        row = dict(source_row)
        for field, raw_value in source_row.items():
            if not field.endswith("_path") or not raw_value:
                continue
            raw_path = Path(raw_value)
            resolved = raw_path if raw_path.is_absolute() else source_csv.parent / raw_path
            if resolved.is_file():
                relative = os.path.relpath(resolved.resolve(), output_csv.parent.resolve())
                row[field] = Path(relative).as_posix()
        rebased_rows.append(row)
    return rebased_rows


def optional_float(value: Optional[object]) -> Optional[float]:
    return None if value is None or value == "" else float(value)


def optional_bool(value: Optional[object]) -> Optional[bool]:
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def candidate_key(row: Mapping[str, object]) -> Tuple[str, str]:
    return str(row.get("dataset_index", "demo")), str(row["candidate_id"])


def reaction_id(row: Mapping[str, object]) -> str:
    return str(row.get("dataset_index", row.get("reaction_id", "demo")))


def candidate_rmsd(row: Mapping[str, object]) -> Optional[float]:
    return optional_float(row.get("paper_style_rmsd") or row.get("permutation_kabsch_rmsd"))


def rmsd_bin(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value < 0.1:
        return "lt_0_1"
    if value < 0.2:
        return "0_1_to_0_2"
    if value < 0.5:
        return "0_2_to_0_5"
    return "ge_0_5"


def force_bin(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value < 0.05:
        return "lt_0_05"
    if value < 0.2:
        return "0_05_to_0_2"
    if value < 0.5:
        return "0_2_to_0_5"
    return "ge_0_5"


def stable_order_key(row: Mapping[str, object], seed: int) -> str:
    key = "|".join((*candidate_key(row), str(seed)))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def coverage_failure_best_keys(
    rows: Sequence[Mapping[str, object]],
    threshold: float,
) -> set[Tuple[str, str]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[reaction_id(row)].append(row)
    selected = set()
    for reaction_rows in grouped.values():
        observed = [(row, candidate_rmsd(row)) for row in reaction_rows]
        if any(value is None for _, value in observed):
            continue
        if any(float(value) < threshold for _, value in observed):
            continue
        best_row, _ = min(observed, key=lambda item: float(item[1]))
        selected.add(candidate_key(best_row))
    return selected


def classify_row(
    row: Mapping[str, object],
    ef_prefix: str,
    efh_prefix: str,
    coverage_best_keys: set[Tuple[str, str]],
    threshold: float,
) -> Tuple[str, str]:
    rmsd = candidate_rmsd(row)
    ef_error = str(row.get(f"{ef_prefix}_error", ""))
    efh_error = str(row.get(f"{efh_prefix}_error", ""))
    ef_class = str(row.get(f"{ef_prefix}_significant_negative_mode_class", ""))
    efh_class = str(row.get(f"{efh_prefix}_significant_negative_mode_class", ""))
    efh_ts_like = optional_bool(row.get(f"{efh_prefix}_is_ts_like"))

    if ef_error or efh_error or not ef_class or not efh_class:
        return "screen_failure", "At least one HORM screen is missing or failed."
    if ef_class != efh_class:
        return (
            "ef_vs_efh_curvature_disagreement",
            f"E/F predicts {ef_class} significant negative modes; E/F/H predicts {efh_class}.",
        )
    if candidate_key(row) in coverage_best_keys:
        return "coverage_failure_best", "Best-RMSD candidate from a reaction with no sample below the RMSD threshold."
    if rmsd is not None and rmsd < threshold and efh_ts_like is False:
        return (
            "geometry_good_physics_bad",
            "Low RMSD, but the E/F/H force-plus-curvature screen rejects the candidate.",
        )
    if rmsd is not None and rmsd >= threshold and efh_class == "1":
        return (
            "geometry_bad_index_one",
            "High RMSD with one significant E/F/H negative mode; possible alternate saddle.",
        )
    if efh_class == "1":
        return "efh_index_one", "E/F/H predicts exactly one significant negative mode."
    if efh_class == ">=2":
        return "efh_multimode", "E/F/H predicts at least two significant negative modes."
    if efh_class == "0":
        return "efh_minimum_like", "E/F/H predicts no significant negative modes."
    return "unclassified", "Insufficient information for a more specific stratum."


def interleave_cells(rows: Sequence[Dict[str, object]], seed: int) -> Deque[Dict[str, object]]:
    cells: Dict[Tuple[str, str], Deque[Dict[str, object]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: stable_order_key(item, seed)):
        cells[(str(row["dft_rmsd_bin"]), str(row["dft_force_bin"]))].append(row)
    ordered_cells = [cells[key] for key in sorted(cells)]
    interleaved: Deque[Dict[str, object]] = deque()
    while ordered_cells:
        remaining = []
        for cell in ordered_cells:
            interleaved.append(cell.popleft())
            if cell:
                remaining.append(cell)
        ordered_cells = remaining
    return interleaved


def select_rows(
    rows: Sequence[Mapping[str, str]],
    budget: int,
    max_per_reaction: int,
    ef_prefix: str,
    efh_prefix: str,
    threshold: float,
    seed: int,
) -> List[Dict[str, object]]:
    coverage_best = coverage_failure_best_keys(rows, threshold)
    classified: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for source_row in rows:
        row: Dict[str, object] = dict(source_row)
        stratum, reason = classify_row(source_row, ef_prefix, efh_prefix, coverage_best, threshold)
        force = optional_float(source_row.get(f"{efh_prefix}_force_rms_ev_per_angstrom"))
        row.update(
            {
                "dft_stratum": stratum,
                "dft_rmsd_bin": rmsd_bin(candidate_rmsd(source_row)),
                "dft_force_bin": force_bin(force),
                "dft_selection_reason": reason,
            }
        )
        classified[stratum].append(row)

    queues = {
        stratum: interleave_cells(classified[stratum], seed)
        for stratum in STRATUM_ORDER
        if classified[stratum]
    }
    reaction_counts: Counter[str] = Counter()
    selected: List[Dict[str, object]] = []
    while queues and len(selected) < budget:
        progress = False
        for stratum in STRATUM_ORDER:
            queue = queues.get(stratum)
            if queue is None:
                continue
            chosen = None
            while queue:
                candidate = queue.popleft()
                if reaction_counts[reaction_id(candidate)] < max_per_reaction:
                    chosen = candidate
                    break
            if not queue:
                queues.pop(stratum, None)
            if chosen is None:
                continue
            reaction_counts[reaction_id(chosen)] += 1
            chosen["dft_selection_rank"] = len(selected) + 1
            selected.append(chosen)
            progress = True
            if len(selected) >= budget:
                break
        if not progress:
            break
    return selected


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    fields, rows = read_csv(args.manifest)
    rows = rebase_path_fields(rows, args.manifest, args.output)
    if not rows:
        raise ValueError(f"Manifest is empty: {args.manifest}")
    required = {
        "candidate_id",
        f"{args.ef_prefix}_significant_negative_mode_class",
        f"{args.efh_prefix}_significant_negative_mode_class",
        f"{args.efh_prefix}_force_rms_ev_per_angstrom",
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Manifest is missing screening columns: {sorted(missing)}")
    if set(SELECTION_FIELDS).intersection(fields):
        raise ValueError("Manifest already contains DFT selection columns")
    selected = select_rows(
        rows,
        args.budget,
        args.max_per_reaction,
        args.ef_prefix,
        args.efh_prefix,
        args.rmsd_threshold,
        args.seed,
    )
    output_fields = SELECTION_FIELDS + fields
    write_csv_atomic(args.output, output_fields, selected)
    summary = {
        "budget": args.budget,
        "candidate_count": len(rows),
        "ef_prefix": args.ef_prefix,
        "efh_prefix": args.efh_prefix,
        "max_per_reaction": args.max_per_reaction,
        "rmsd_threshold": args.rmsd_threshold,
        "seed": args.seed,
        "selected_count": len(selected),
        "stratum_counts": dict(sorted(Counter(str(row["dft_stratum"]) for row in selected).items())),
        "unique_reaction_count": len({reaction_id(row) for row in selected}),
    }
    write_json_atomic(args.output.with_suffix(args.output.suffix + ".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
