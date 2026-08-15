#!/usr/bin/env python3
"""Aggregate candidate diagnostics into per-reaction, three-state failure labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from oa_reactdiff.analyze.failure_audit import summarize_candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rmsd-threshold", type=float, default=0.2)
    parser.add_argument("--force-rms-threshold-ev-per-angstrom", type=float, default=0.05)
    parser.add_argument("--confidence-lower-is-better", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))
    if not candidate_rows:
        raise ValueError(f"No candidate rows found in {args.manifest}")

    reaction_rows = summarize_candidate_rows(
        candidate_rows,
        rmsd_threshold=args.rmsd_threshold,
        force_rms_threshold_ev_per_angstrom=args.force_rms_threshold_ev_per_angstrom,
        confidence_higher_is_better=not args.confidence_lower_is_better,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reaction_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reaction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reaction_rows)

    status_fields = [
        "coverage_failure",
        "ranking_failure",
        "geometry_good_physics_bad",
        "alternate_saddle",
    ]
    summary = {
        "candidate_manifest": str(args.manifest),
        "reaction_count": len(reaction_rows),
        "rmsd_threshold": args.rmsd_threshold,
        "force_rms_threshold_ev_per_angstrom": args.force_rms_threshold_ev_per_angstrom,
        "confidence_higher_is_better": not args.confidence_lower_is_better,
        "status_counts": {
            field: dict(sorted(Counter(str(row[field]) for row in reaction_rows).items()))
            for field in status_fields
        },
    }
    (args.output_dir / "failure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
