#!/usr/bin/env python3
"""Generate repeated R/P-conditioned TS candidates with resumable manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


ATOM_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
MANIFEST_FIELDS = [
    "dataset_index",
    "source_reaction_index",
    "candidate_id",
    "paper_style_rmsd",
    "candidate_path",
    "reactant_path",
    "reference_ts_path",
    "product_path",
    "molecular_charge",
    "spin",
    "generation_wall_seconds",
    "generation_seconds_per_candidate",
    "confidence",
    "force_rms_ev_per_angstrom",
    "strict_imaginary_modes",
    "optimized",
    "optimized_to_intended_cluster",
    "irc_connects_target",
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
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained-ts1x-diff.ckpt"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("oa_reactdiff/data/transition1x/valid_addprop.pkl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/oa_failure_audit/outputs/pilot"))
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="Filtered dataset indices; mutually exclusive with --start-index.",
    )
    parser.add_argument(
        "--start-index",
        type=nonnegative_int,
        default=0,
        help="First filtered dataset index when --indices is omitted.",
    )
    parser.add_argument("--max-reactions", type=positive_int, default=8)
    parser.add_argument("--samples-per-reaction", type=positive_int, default=8)
    parser.add_argument("--timesteps", type=positive_int, default=150)
    parser.add_argument("--resamplings", type=positive_int, default=2)
    parser.add_argument("--jump-length", type=positive_int, default=2)
    parser.add_argument("--noise-schedule", default="polynomial_2")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--single-fragment-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-reaction manifests after validating settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and index mapping without importing PyTorch or writing outputs.",
    )
    return parser.parse_args()


def selected_source_indices(raw_dataset: Dict[str, object], single_fragment_only: bool) -> List[int]:
    use_indices = set(int(index) for index in raw_dataset["use_ind"])
    if single_fragment_only:
        eligible = {
            index
            for index, is_single_fragment in enumerate(raw_dataset["single_fragment"])
            if int(is_single_fragment) == 1
        }
    else:
        eligible = set(range(len(raw_dataset["single_fragment"])))
    # Match ProcessedTS1x, which materializes a set intersection as a list.
    return list(eligible.intersection(use_indices))


def select_dataset_indices(args: argparse.Namespace, available_count: int) -> List[int]:
    if args.indices is not None:
        if args.start_index != 0:
            raise ValueError("--indices and a non-zero --start-index cannot be used together")
        dataset_indices = list(args.indices)
    else:
        stop_index = min(args.start_index + args.max_reactions, available_count)
        dataset_indices = list(range(args.start_index, stop_index))
    if not dataset_indices:
        raise ValueError("No reactions were selected")
    if len(set(dataset_indices)) != len(dataset_indices):
        raise ValueError("Filtered dataset indices must be unique")
    invalid_indices = [index for index in dataset_indices if index < 0 or index >= available_count]
    if invalid_indices:
        raise IndexError(f"Filtered dataset indices out of range: {invalid_indices}")
    return dataset_indices


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch_module.device(requested)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_xyz(path: Path, atomic_numbers: Sequence[int], coordinates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atomic_numbers)}\n\n")
        for atomic_number, coordinate in zip(atomic_numbers, coordinates):
            symbol = ATOM_SYMBOLS[int(atomic_number)]
            handle.write(f"{symbol} {coordinate[0]:.10f} {coordinate[1]:.10f} {coordinate[2]:.10f}\n")
    temporary_path.replace(path)


def split_tensor(tensor, sizes) -> List[object]:
    split_points = sizes.cumsum(dim=0).cpu().tolist()[:-1]
    return list(tensor.tensor_split(split_points))


def relative_output_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reaction_is_complete(reaction_dir: Path, output_dir: Path, expected_count: int) -> bool:
    manifest_path = reaction_dir / "candidate_manifest.csv"
    if not manifest_path.is_file():
        return False
    rows = read_manifest(manifest_path)
    if len(rows) != expected_count:
        return False
    required_paths = ["candidate_path", "reactant_path", "reference_ts_path", "product_path"]
    return all((output_dir / row[field]).is_file() for row in rows for field in required_paths)


def merge_reaction_manifests(output_dir: Path, dataset_indices: Sequence[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for dataset_index in sorted(dataset_indices):
        manifest_path = output_dir / f"reaction_{dataset_index:05d}" / "candidate_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed reaction manifest: {manifest_path}")
        rows.extend(read_manifest(manifest_path))
    write_csv_atomic(output_dir / "candidate_manifest.csv", rows)
    return rows


def settings_for_comparison(settings: Mapping[str, object]) -> Dict[str, object]:
    ignored = {
        "gpu_name",
        "resolved_device",
    }
    return {key: value for key, value in settings.items() if key not in ignored}


def prepare_output(output_dir: Path, settings: Mapping[str, object], resume: bool) -> None:
    settings_path = output_dir / "settings.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory is not empty; use --resume or choose another path: {output_dir}")
        if not settings_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {settings_path}")
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings_for_comparison(previous) != settings_for_comparison(settings):
            raise ValueError("Resume settings differ from the existing settings.json")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(settings_path, settings)


def synchronize_cuda(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def set_sampling_schedule(model, timesteps: int, device, noise_schedule: str):
    from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule

    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=1.0e-5,
    )
    model.ddpm.schedule = DiffSchedule(
        gamma_module=gamma_module,
        norm_values=model.ddpm.norm_values,
    )
    model.ddpm.T = timesteps
    return model.to(device)


def inpaint_batch(batch, model, resamplings: int, jump_length: int):
    import torch

    from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING

    representations, conditions = batch
    fixed = [
        torch.cat([representation[feature] for feature in FEATURE_MAPPING], dim=1)
        for representation in representations
    ]
    fragment_nodes = [representation["size"] for representation in representations]
    samples, _ = model.ddpm.inpaint(
        n_samples=representations[0]["size"].size(0),
        fragments_nodes=fragment_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=resamplings,
        jump_length=jump_length,
        timesteps=None,
        xh_fixed=fixed,
        frag_fixed=[0, 2],
    )
    return samples[0], fixed, fragment_nodes


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    with args.dataset.open("rb") as handle:
        raw_dataset = pickle.load(handle)
    source_indices = selected_source_indices(raw_dataset, args.single_fragment_only)
    dataset_indices = select_dataset_indices(args, len(source_indices))

    base_settings: Dict[str, object] = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset": str(args.dataset),
        "dataset_indices": dataset_indices,
        "source_indices": [source_indices[index] for index in dataset_indices],
        "samples_per_reaction": args.samples_per_reaction,
        "timesteps": args.timesteps,
        "resamplings": args.resamplings,
        "jump_length": args.jump_length,
        "noise_schedule": args.noise_schedule,
        "seed": args.seed,
        "requested_device": args.device,
        "single_fragment_only": args.single_fragment_only,
    }
    if args.dry_run:
        print(json.dumps(base_settings, indent=2, sort_keys=True))
        return

    import numpy as np
    import torch

    from oa_reactdiff.analyze.rmsd import batch_rmsd
    from oa_reactdiff.dataset.transition1x import ProcessedTS1x
    from oa_reactdiff.trainer.pl_trainer import DDPMModule

    device = resolve_device(torch, args.device)
    settings = dict(base_settings)
    settings.update(
        {
            "python_version": platform.python_version(),
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
    )
    prepare_output(args.output_dir, settings, args.resume)

    model = DDPMModule.load_from_checkpoint(str(args.checkpoint), map_location=device)
    model = set_sampling_schedule(
        model,
        timesteps=args.timesteps,
        device=device,
        noise_schedule=args.noise_schedule,
    )
    model.eval()

    dataset = ProcessedTS1x(
        npz_path=str(args.dataset),
        center=True,
        pad_fragments=0,
        device=device,
        zero_charge=False,
        remove_h=False,
        single_frag_only=args.single_fragment_only,
        swapping_react_prod=False,
        use_by_ind=True,
    )

    completed = 0
    for ordinal, dataset_index in enumerate(dataset_indices, start=1):
        reaction_dir = args.output_dir / f"reaction_{dataset_index:05d}"
        if args.resume and reaction_is_complete(
            reaction_dir,
            args.output_dir,
            args.samples_per_reaction,
        ):
            completed += 1
            print(f"[{ordinal}/{len(dataset_indices)}] reaction {dataset_index}: already complete", flush=True)
            continue

        reaction_seed = args.seed + dataset_index
        torch.manual_seed(reaction_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(reaction_seed)

        repeated_items = [dataset[dataset_index] for _ in range(args.samples_per_reaction)]
        batch = dataset.collate_fn(repeated_items)
        synchronize_cuda(torch, device)
        start_time = time.perf_counter()
        with torch.inference_mode():
            out_samples, xh_fixed, fragment_nodes = inpaint_batch(
                batch,
                model,
                resamplings=args.resamplings,
                jump_length=args.jump_length,
            )
        synchronize_cuda(torch, device)
        generation_wall_seconds = time.perf_counter() - start_time
        rmsds = batch_rmsd(fragment_nodes, out_samples, xh_fixed, idx=1, threshold=0.5)

        reaction_dir.mkdir(parents=True, exist_ok=True)
        split_outputs = [
            split_tensor(fragment, fragment_nodes[fragment_index])
            for fragment_index, fragment in enumerate(out_samples)
        ]
        split_fixed = [
            split_tensor(fragment, fragment_nodes[fragment_index])
            for fragment_index, fragment in enumerate(xh_fixed)
        ]

        reference_paths = {}
        for fragment_index, name in ((0, "reactant"), (1, "reference_ts"), (2, "product")):
            structure = split_fixed[fragment_index][0].detach().cpu().numpy()
            reference_path = reaction_dir / f"{name}.xyz"
            write_xyz(
                reference_path,
                np.rint(structure[:, -1]).astype(int),
                structure[:, :3],
            )
            reference_paths[name] = relative_output_path(reference_path, args.output_dir)

        reaction_rows = []
        seconds_per_candidate = generation_wall_seconds / args.samples_per_reaction
        for candidate_id, (candidate, rmsd) in enumerate(zip(split_outputs[1], rmsds)):
            candidate_array = candidate.detach().cpu().numpy()
            candidate_path = reaction_dir / f"candidate_{candidate_id:03d}_ts.xyz"
            write_xyz(
                candidate_path,
                np.rint(candidate_array[:, -1]).astype(int),
                candidate_array[:, :3],
            )
            reaction_rows.append(
                {
                    "dataset_index": dataset_index,
                    "source_reaction_index": source_indices[dataset_index],
                    "candidate_id": candidate_id,
                    "paper_style_rmsd": float(rmsd),
                    "candidate_path": relative_output_path(candidate_path, args.output_dir),
                    "reactant_path": reference_paths["reactant"],
                    "reference_ts_path": reference_paths["reference_ts"],
                    "product_path": reference_paths["product"],
                    "molecular_charge": 0,
                    "spin": 0,
                    "generation_wall_seconds": generation_wall_seconds,
                    "generation_seconds_per_candidate": seconds_per_candidate,
                    "confidence": "",
                    "force_rms_ev_per_angstrom": "",
                    "strict_imaginary_modes": "",
                    "optimized": "",
                    "optimized_to_intended_cluster": "",
                    "irc_connects_target": "",
                }
            )

        write_csv_atomic(reaction_dir / "candidate_manifest.csv", reaction_rows)
        write_json_atomic(
            reaction_dir / "timing.json",
            {
                "candidate_count": args.samples_per_reaction,
                "dataset_index": dataset_index,
                "generation_seconds_per_candidate": seconds_per_candidate,
                "generation_wall_seconds": generation_wall_seconds,
                "source_reaction_index": source_indices[dataset_index],
            },
        )
        completed += 1
        print(
            f"[{ordinal}/{len(dataset_indices)}] reaction {dataset_index}: "
            f"{generation_wall_seconds:.2f}s ({seconds_per_candidate:.2f}s/candidate)",
            flush=True,
        )

    rows = merge_reaction_manifests(args.output_dir, dataset_indices)
    write_json_atomic(
        args.output_dir / "timing_summary.json",
        {
            "candidate_count": len(rows),
            "completed_reaction_count": completed,
            "mean_generation_seconds_per_candidate": sum(
                float(row["generation_seconds_per_candidate"]) for row in rows
            )
            / len(rows),
            "reaction_count": len(dataset_indices),
        },
    )
    print(f"Wrote {len(rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
