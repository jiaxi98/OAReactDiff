#!/usr/bin/env python3
"""Turn merged DFT results into a physically gated optimization/IRC worklist."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


FOLLOWUP_FIELDS = ["followup_priority", "followup_action", "irc_eligible_now", "followup_reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dft-prefix", default="dft")
    parser.add_argument("--efh-prefix", default="horm_left_efh")
    parser.add_argument("--rmsd-threshold", type=float, default=0.2)
    parser.add_argument("--include-no-followup", action="store_true")
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


def classify_followup(
    row: Mapping[str, str],
    dft_prefix: str,
    efh_prefix: str,
    rmsd_threshold: float,
) -> Tuple[int, str, bool, str]:
    dft_error = row.get(f"{dft_prefix}_error", "")
    dft_index_one = optional_bool(row.get(f"{dft_prefix}_is_index_one"))
    dft_ts_like = optional_bool(row.get(f"{dft_prefix}_is_ts_like"))
    efh_class = row.get(f"{efh_prefix}_significant_negative_mode_class", "")
    rmsd = optional_float(row.get("paper_style_rmsd") or row.get("permutation_kabsch_rmsd"))
    stratum = row.get("dft_stratum", "")

    if dft_error:
        return 1, "repair_dft_calculation", False, f"DFT calculation failed: {dft_error}"
    if dft_ts_like is True:
        if rmsd is not None and rmsd >= rmsd_threshold:
            reason = "Stationary DFT index-1 point has high reference RMSD; test whether it is an alternate saddle."
        else:
            reason = "Stationary DFT index-1 point is ready for the connectivity certificate."
        return 1, "run_bidirectional_irc", True, reason
    if dft_index_one is True:
        return (
            2,
            "ts_optimize_then_rehessian",
            False,
            "DFT finds index-1 curvature, but the raw candidate is not stationary; optimize before any IRC.",
        )
    anomaly = (
        efh_class == "1"
        or stratum in {"ef_vs_efh_curvature_disagreement", "geometry_good_physics_bad"}
        or (rmsd is not None and rmsd < rmsd_threshold)
    )
    if anomaly:
        return (
            3,
            "review_or_optimize_then_rehessian",
            False,
            "DFT rejects index-1 curvature despite a geometric or MLIP signal; IRC is not yet well-defined.",
        )
    return 4, "no_followup", False, "DFT does not identify an index-1 or prespecified anomalous case."


def prepare_rows(
    rows: Sequence[Mapping[str, str]],
    dft_prefix: str,
    efh_prefix: str,
    rmsd_threshold: float,
    include_no_followup: bool,
) -> List[Dict[str, object]]:
    prepared = []
    for source_row in rows:
        if not source_row.get(f"{dft_prefix}_scf_converged") and not source_row.get(f"{dft_prefix}_error"):
            continue
        priority, action, eligible, reason = classify_followup(
            source_row,
            dft_prefix,
            efh_prefix,
            rmsd_threshold,
        )
        if action == "no_followup" and not include_no_followup:
            continue
        row: Dict[str, object] = dict(source_row)
        row.update(
            {
                "followup_priority": priority,
                "followup_action": action,
                "irc_eligible_now": "true" if eligible else "false",
                "followup_reason": reason,
            }
        )
        prepared.append(row)
    return sorted(
        prepared,
        key=lambda row: (
            int(row["followup_priority"]),
            str(row.get("dataset_index", "")),
            str(row.get("candidate_id", "")),
        ),
    )


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    fields, rows = read_csv(args.manifest)
    rows = rebase_path_fields(rows, args.manifest, args.output)
    required = {
        f"{args.dft_prefix}_error",
        f"{args.dft_prefix}_is_index_one",
        f"{args.dft_prefix}_is_ts_like",
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Manifest is missing DFT columns: {sorted(missing)}")
    prepared = prepare_rows(
        rows,
        args.dft_prefix,
        args.efh_prefix,
        args.rmsd_threshold,
        args.include_no_followup,
    )
    write_csv_atomic(args.output, FOLLOWUP_FIELDS + fields, prepared)
    irc_count = sum(row["irc_eligible_now"] == "true" for row in prepared)
    print(f"Wrote {len(prepared)} follow-ups ({irc_count} immediately IRC-eligible) to {args.output}")


if __name__ == "__main__":
    main()
