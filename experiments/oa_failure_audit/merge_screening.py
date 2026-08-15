#!/usr/bin/env python3
"""Merge one or more model-screen CSVs into a generated-candidate manifest."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


IDENTITY_FIELDS = {
    "dataset_index",
    "source_reaction_index",
    "candidate_id",
    "candidate_path",
    "paper_style_rmsd",
    "permutation_kabsch_rmsd",
    "model_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--screen",
        action="append",
        required=True,
        metavar="PREFIX=CSV",
        help="Screen result and output-column prefix; repeat for each model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def candidate_key(row: Mapping[str, object]) -> Tuple[str, str]:
    return str(row.get("dataset_index", "demo")), str(row["candidate_id"])


def parse_screen_specification(specification: str) -> Tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Expected PREFIX=CSV, found {specification!r}")
    prefix, raw_path = specification.split("=", 1)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError(f"Invalid screen prefix {prefix!r}")
    return prefix, Path(raw_path)


def rebase_path_fields(
    rows: Sequence[Mapping[str, str]],
    source_csv: Path,
    output_csv: Path,
) -> List[Dict[str, str]]:
    """Keep artifact paths valid when a derived manifest is written elsewhere."""
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


def unique_by_key(rows: Sequence[Mapping[str, str]], source_name: str) -> Dict[Tuple[str, str], Mapping[str, str]]:
    indexed: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = candidate_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate candidate key {key} in {source_name}")
        indexed[key] = row
    return indexed


def merge_rows(
    manifest_fields: Sequence[str],
    manifest_rows: Sequence[Mapping[str, str]],
    screens: Sequence[Tuple[str, Sequence[str], Sequence[Mapping[str, str]]]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    manifest_by_key = unique_by_key(manifest_rows, "manifest")
    output_fields = list(manifest_fields)
    screen_indexes = []
    for prefix, screen_fields, screen_rows in screens:
        screen_by_key = unique_by_key(screen_rows, prefix)
        unexpected = set(screen_by_key).difference(manifest_by_key)
        if unexpected:
            preview = sorted(unexpected)[:5]
            raise ValueError(f"Screen {prefix!r} contains candidates absent from the manifest: {preview}")
        metric_fields = [field for field in screen_fields if field not in IDENTITY_FIELDS]
        prefixed_fields = [f"{prefix}_{field}" for field in metric_fields]
        collisions = set(prefixed_fields).intersection(output_fields)
        if collisions:
            raise ValueError(f"Output columns already exist: {sorted(collisions)}")
        output_fields.extend(prefixed_fields)
        screen_indexes.append((prefix, metric_fields, screen_by_key))

    merged_rows: List[Dict[str, str]] = []
    for source_row in manifest_rows:
        key = candidate_key(source_row)
        merged = dict(source_row)
        for prefix, metric_fields, screen_by_key in screen_indexes:
            screen_row = screen_by_key.get(key, {})
            for field in metric_fields:
                merged[f"{prefix}_{field}"] = screen_row.get(field, "")
        merged_rows.append(merged)
    return output_fields, merged_rows


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
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
    manifest_fields, manifest_rows = read_csv(args.manifest)
    manifest_rows = rebase_path_fields(manifest_rows, args.manifest, args.output)
    screens = []
    prefixes = set()
    for specification in args.screen:
        prefix, screen_path = parse_screen_specification(specification)
        if prefix in prefixes:
            raise ValueError(f"Duplicate screen prefix: {prefix}")
        prefixes.add(prefix)
        screen_fields, screen_rows = read_csv(screen_path)
        screen_rows = rebase_path_fields(screen_rows, screen_path, args.output)
        screens.append((prefix, screen_fields, screen_rows))
    output_fields, output_rows = merge_rows(manifest_fields, manifest_rows, screens)
    write_csv_atomic(args.output, output_fields, output_rows)
    print(f"Wrote {len(output_rows)} candidates with {len(screens)} screens to {args.output}")


if __name__ == "__main__":
    main()
