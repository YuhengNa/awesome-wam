#!/usr/bin/env python
"""Audit camera views and storage formats before mixed-dataset PV-VAE training.

The script is intentionally read-only with respect to the datasets. It accepts
one or more dataset roots and writes a compact audit report containing:

- discovered LeRobot dataset roots and declared video features
- candidate camera/video groups in unknown directory layouts
- per-view file counts, file-size statistics, and repeated small-file rates
- sampled decode, black/flat-frame, and temporal-motion statistics
- contact sheets for visually identifying each camera role

This is the required inventory step before assigning canonical view roles such
as ``external_fixed_primary``, ``head_primary``, or ``wrist`` and before
building a mixed-dataset sampler.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".svo", ".svo2"}
DATA_SUFFIXES = {
    ".avi",
    ".h5",
    ".hdf5",
    ".json",
    ".jsonl",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".svo",
    ".svo2",
    ".tfrecord",
    ".webm",
}
CAMERA_WORDS = (
    "camera",
    "cam",
    "image",
    "rgb",
    "video",
    "wrist",
    "hand",
    "gripper",
    "head",
    "ego",
    "exterior",
    "external",
    "interior",
    "primary",
    "static",
)
EPISODE_RE = re.compile(r"episode_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="Dataset root. Repeat for multiple roots.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=12, help="Maximum recursive discovery depth from each input root.")
    parser.add_argument("--max-files", type=int, default=0, help="Stop discovery after this many files per root; 0 means all.")
    parser.add_argument("--max-generic-video-groups", type=int, default=32)
    parser.add_argument("--decode-videos-per-view", type=int, default=12)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--contact-sheet-videos", type=int, default=4)
    parser.add_argument("--small-file-threshold-kb", type=int, default=16)
    parser.add_argument("--hash-max-bytes-kb", type=int, default=1024)
    parser.add_argument("--max-hash-files-per-view", type=int, default=4096)
    parser.add_argument("--inspect-hdf5-files", type=int, default=4)
    parser.add_argument("--rgb-black-mean-threshold", type=float, default=0.02)
    parser.add_argument("--rgb-flat-std-threshold", type=float, default=0.02)
    parser.add_argument("--rgb-motion-threshold", type=float, default=0.02)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_name(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return rendered or "unnamed"


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def role_hint(camera_key: str) -> tuple[str, str]:
    key = camera_key.lower()
    if any(word in key for word in ("wrist", "hand", "gripper", "interior")):
        return "wrist", "high"
    if any(word in key for word in ("head", "ego")):
        return "head_primary_candidate", "medium"
    if any(word in key for word in ("external", "exterior", "over_shoulder", "static", "primary")):
        return "external_primary_candidate", "medium"
    if re.search(r"(?:image|camera|cam)[._-]?0(?:\D|$)", key):
        return "primary_candidate", "low"
    if key in {"image", "rgb", "observation.images.image"}:
        return "primary_candidate", "low"
    return "unknown", "none"


def hash_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video_sample(path: Path, frames_per_video: int) -> torch.Tensor:
    if path.suffix.lower() in {".svo", ".svo2"}:
        raise RuntimeError("ZED SVO requires the ZED SDK and is not decoded by this audit script.")
    first_error: Exception | None = None
    try:
        from torchvision.io import read_video  # type: ignore

        video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
        if video.shape[0] == 0:
            raise RuntimeError("decoded zero frames")
        indices = torch.linspace(0, video.shape[0] - 1, steps=min(frames_per_video, video.shape[0])).long()
        return video[indices].float().div(255.0).clamp(0.0, 1.0)
    except Exception as exc:
        first_error = exc
    try:
        import imageio.v3 as iio  # type: ignore

        metadata = iio.immeta(path)
        num_frames = int(metadata.get("nframes") or metadata.get("n_frames") or 0)
        if num_frames > 0:
            indices = np.linspace(0, num_frames - 1, num=min(frames_per_video, num_frames), dtype=np.int64)
            arr = iio.imread(path, index=indices.tolist())
        else:
            arr = iio.imread(path)
            if arr.ndim == 3:
                arr = arr[None]
            indices = np.linspace(0, arr.shape[0] - 1, num=min(frames_per_video, arr.shape[0]), dtype=np.int64)
            arr = arr[indices]
        frames = torch.as_tensor(arr)
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        return frames.permute(0, 3, 1, 2).float().div(255.0).clamp(0.0, 1.0)
    except Exception as second_error:
        raise RuntimeError(f"torchvision={first_error}; imageio={second_error}") from second_error


def video_metrics(frames: torch.Tensor) -> dict[str, float | int]:
    delta = (frames[1:] - frames[:1]).abs().mean(dim=(1, 2, 3)) if frames.shape[0] > 1 else torch.zeros(1)
    adjacent = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3)) if frames.shape[0] > 1 else torch.zeros(1)
    return {
        "decoded_frames": int(frames.shape[0]),
        "height": int(frames.shape[-2]),
        "width": int(frames.shape[-1]),
        "rgb_mean": float(frames.mean()),
        "rgb_std": float(frames.std()),
        "rgb_max_delta_to_first": float(delta.max()),
        "rgb_adjacent_delta": float(adjacent.mean()),
    }


def tensor_to_pil(frame: torch.Tensor, size: int = 144) -> Image.Image:
    image = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)


def write_contact_sheet(
    output_path: Path,
    title: str,
    decoded: list[tuple[Path, torch.Tensor, dict[str, float | int]]],
) -> None:
    if not decoded:
        return
    cell = 144
    frames_per_row = max(item[1].shape[0] for item in decoded)
    title_h = 48
    row_label_h = 36
    width = frames_per_row * cell
    height = title_h + len(decoded) * (cell + row_label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)
    for row, (path, frames, metrics) in enumerate(decoded):
        y = title_h + row * (cell + row_label_h)
        for column, frame in enumerate(frames):
            canvas.paste(tensor_to_pil(frame, cell), (column * cell, y))
        label = (
            f"{path.name} mean={metrics['rgb_mean']:.3f} std={metrics['rgb_std']:.3f} "
            f"motion={metrics['rgb_max_delta_to_first']:.3f}"
        )
        draw.text((4, y + cell + 5), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def camera_candidate_label(path: Path, root: Path) -> str:
    try:
        parts = list(path.relative_to(root).parts)
    except ValueError:
        parts = list(path.parts)
    candidates = [path.stem, *reversed(parts[:-1])]
    for candidate in candidates:
        lowered = candidate.lower()
        if any(word in lowered for word in CAMERA_WORDS):
            return candidate
    return path.parent.name


def camera_like_keys(value: Any, prefix: str = "", limit: int = 128) -> list[str]:
    matches: list[str] = []

    def visit(item: Any, current: str) -> None:
        if len(matches) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{current}.{key}" if current else str(key)
                if any(word in str(key).lower() for word in CAMERA_WORDS):
                    matches.append(child_path)
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item[:16]):
                visit(child, f"{current}[{index}]")

    visit(value, prefix)
    return matches


def summarize_dataset_info(path: Path) -> dict[str, Any]:
    try:
        info = json.loads(path.read_text())
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "name": info.get("name"),
        "version": info.get("version"),
        "top_level_keys": sorted(info.keys()),
        "split_keys": sorted(info.get("splits", {}).keys()) if isinstance(info.get("splits"), dict) else [],
        "camera_like_keys": camera_like_keys(info),
    }


def summarize_hdf5(path: Path, max_datasets: int = 512) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        return {"path": str(path), "error": f"h5py import failed: {type(exc).__name__}: {exc}"}
    datasets: list[dict[str, Any]] = []
    camera_datasets: list[dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as handle:
            def visit(name: str, item: Any) -> None:
                if len(datasets) >= max_datasets or not isinstance(item, h5py.Dataset):
                    return
                record = {"name": name, "shape": list(item.shape), "dtype": str(item.dtype)}
                datasets.append(record)
                if any(word in name.lower() for word in CAMERA_WORDS):
                    camera_datasets.append(record)

            handle.visititems(visit)
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "dataset_count_inspected": len(datasets),
        "camera_like_datasets": camera_datasets[:128],
        "sample_datasets": datasets[:32],
    }


def discover_tree(root: Path, max_depth: int, max_files: int) -> dict[str, Any]:
    suffix_counts: Counter[str] = Counter()
    sample_files_by_suffix: dict[str, list[str]] = defaultdict(list)
    camera_video_groups: dict[str, list[Path]] = defaultdict(list)
    lerobot_info_paths: list[Path] = []
    dataset_info_paths: list[Path] = []
    total_files = 0
    total_bytes = 0
    truncated = False
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth >= max_depth:
            dirs[:] = []
        for name in files:
            path = current_path / name
            total_files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            suffix = path.suffix.lower()
            suffix_counts[suffix or "<none>"] += 1
            if suffix in DATA_SUFFIXES and len(sample_files_by_suffix[suffix]) < 8:
                sample_files_by_suffix[suffix].append(relative_or_absolute(path, root))
            if name == "info.json" and current_path.name == "meta":
                lerobot_info_paths.append(path)
            if name == "dataset_info.json":
                dataset_info_paths.append(path)
            if suffix in VIDEO_SUFFIXES:
                camera_video_groups[camera_candidate_label(path, root)].append(path)
            if max_files > 0 and total_files >= max_files:
                truncated = True
                dirs[:] = []
                break
        if truncated:
            break
    return {
        "root": str(root),
        "exists": root.exists(),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "truncated": truncated,
        "suffix_counts": dict(suffix_counts.most_common()),
        "sample_files_by_suffix": dict(sample_files_by_suffix),
        "top_level_entries": sorted(path.name for path in root.iterdir())[:128],
        "lerobot_info_paths": lerobot_info_paths,
        "dataset_info_paths": dataset_info_paths,
        "generic_video_groups": camera_video_groups,
    }


def lerobot_video_groups(dataset_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        return groups
    for path in videos_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            groups[path.parent.name].append(path)
    return groups


def audit_video_group(
    paths: list[Path],
    *,
    camera_key: str,
    dataset_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    paths = sorted(set(paths))
    sizes: list[int] = []
    small_paths: list[Path] = []
    small_threshold = args.small_file_threshold_kb * 1024
    hash_max_bytes = args.hash_max_bytes_kb * 1024
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        sizes.append(size)
        if size < small_threshold:
            small_paths.append(path)

    hash_counts: Counter[str] = Counter()
    hash_failures = 0
    hash_candidates = []
    for path in paths:
        try:
            if path.stat().st_size <= hash_max_bytes:
                hash_candidates.append(path)
        except OSError:
            hash_failures += 1
    rng.shuffle(hash_candidates)
    hash_candidates = hash_candidates[: args.max_hash_files_per_view]
    for path in hash_candidates:
        try:
            hash_counts[hash_file(path)] += 1
        except OSError:
            hash_failures += 1
    repeated_hash_files = sum(count for count in hash_counts.values() if count > 1)
    hashed_files = sum(hash_counts.values())

    decoded: list[tuple[Path, torch.Tensor, dict[str, float | int]]] = []
    decode_errors: list[str] = []
    decode_candidates = list(paths)
    rng.shuffle(decode_candidates)
    decode_candidates = decode_candidates[: args.decode_videos_per_view]
    if not args.skip_decode:
        for path in decode_candidates:
            try:
                frames = read_video_sample(path, args.frames_per_video)
                metrics = video_metrics(frames)
                decoded.append((path, frames, metrics))
            except Exception as exc:
                decode_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    rgb_means = [float(item[2]["rgb_mean"]) for item in decoded]
    rgb_stds = [float(item[2]["rgb_std"]) for item in decoded]
    rgb_motion = [float(item[2]["rgb_max_delta_to_first"]) for item in decoded]
    role, confidence = role_hint(camera_key)
    sheet_path = output_dir / f"{safe_name(camera_key)}.png"
    write_contact_sheet(
        sheet_path,
        f"{dataset_root.name} / {camera_key} / role_hint={role}",
        decoded[: args.contact_sheet_videos],
    )
    decoded_count = len(decoded)
    return {
        "camera_key": camera_key,
        "role_hint": role,
        "role_hint_confidence": confidence,
        "total_files": len(paths),
        "file_size_bytes": summarize(sizes),
        "small_file_threshold_bytes": small_threshold,
        "small_file_count": len(small_paths),
        "small_file_ratio": len(small_paths) / max(len(paths), 1),
        "hashed_files": hashed_files,
        "repeated_hash_files": repeated_hash_files,
        "repeated_hash_ratio": repeated_hash_files / max(hashed_files, 1),
        "hash_failures": hash_failures,
        "decode_attempted": len(decode_candidates) if not args.skip_decode else 0,
        "decode_succeeded": decoded_count,
        "decode_success_ratio": decoded_count / max(len(decode_candidates), 1) if not args.skip_decode else None,
        "black_ratio": (
            sum(value < args.rgb_black_mean_threshold for value in rgb_means) / decoded_count if decoded_count else None
        ),
        "flat_ratio": (
            sum(value < args.rgb_flat_std_threshold for value in rgb_stds) / decoded_count if decoded_count else None
        ),
        "motion_ratio": (
            sum(value >= args.rgb_motion_threshold for value in rgb_motion) / decoded_count if decoded_count else None
        ),
        "rgb_mean": summarize(rgb_means),
        "rgb_std": summarize(rgb_stds),
        "rgb_max_delta_to_first": summarize(rgb_motion),
        "decode_errors": decode_errors[:8],
        "contact_sheet": str(sheet_path) if decoded else None,
        "example_paths": [relative_or_absolute(path, dataset_root) for path in paths[:5]],
    }


def audit_lerobot_dataset(
    info_path: Path,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    dataset_root = info_path.parent.parent
    try:
        info = json.loads(info_path.read_text())
    except Exception as exc:
        return {"dataset_root": str(dataset_root), "error": f"Failed to read info.json: {exc}"}
    features = info.get("features", {})
    declared_video_features = {
        key: value
        for key, value in features.items()
        if isinstance(value, dict) and str(value.get("dtype", "")).lower() in {"video", "image"}
    }
    groups = lerobot_video_groups(dataset_root)
    dataset_output = output_dir / safe_name(dataset_root.name)
    views = [
        audit_video_group(
            paths,
            camera_key=key,
            dataset_root=dataset_root,
            output_dir=dataset_output,
            args=args,
            rng=rng,
        )
        for key, paths in sorted(groups.items())
    ]
    for view in views:
        declared = declared_video_features.get(view["camera_key"])
        view["declared_in_metadata"] = declared is not None
        view["declared_shape"] = declared.get("shape") if declared else None
    parquet_count = sum(1 for _ in (dataset_root / "data").rglob("*.parquet")) if (dataset_root / "data").exists() else 0
    return {
        "dataset_kind": "lerobot",
        "dataset_root": str(dataset_root),
        "repo_id": info.get("repo_id"),
        "fps": info.get("fps"),
        "video_path_template": info.get("video_path"),
        "data_path_template": info.get("data_path"),
        "declared_video_features": declared_video_features,
        "parquet_count": parquet_count,
        "views": views,
    }


def flatten_view_rows(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_audit in audits:
        input_root = root_audit["root"]
        for dataset in root_audit.get("lerobot_datasets", []):
            for view in dataset.get("views", []):
                rows.append(
                    {
                        "input_root": input_root,
                        "dataset_root": dataset.get("dataset_root"),
                        "dataset_kind": dataset.get("dataset_kind"),
                        "fps": dataset.get("fps"),
                        "camera_key": view.get("camera_key"),
                        "role_hint": view.get("role_hint"),
                        "role_hint_confidence": view.get("role_hint_confidence"),
                        "total_files": view.get("total_files"),
                        "small_file_ratio": view.get("small_file_ratio"),
                        "repeated_hash_ratio": view.get("repeated_hash_ratio"),
                        "decode_success_ratio": view.get("decode_success_ratio"),
                        "black_ratio": view.get("black_ratio"),
                        "flat_ratio": view.get("flat_ratio"),
                        "motion_ratio": view.get("motion_ratio"),
                        "contact_sheet": view.get("contact_sheet"),
                    }
                )
        for group in root_audit.get("generic_video_groups", []):
            rows.append(
                {
                    "input_root": input_root,
                    "dataset_root": input_root,
                    "dataset_kind": "generic_video_group",
                    "camera_key": group.get("camera_key"),
                    "role_hint": group.get("role_hint"),
                    "role_hint_confidence": group.get("role_hint_confidence"),
                    "total_files": group.get("total_files"),
                    "small_file_ratio": group.get("small_file_ratio"),
                    "repeated_hash_ratio": group.get("repeated_hash_ratio"),
                    "decode_success_ratio": group.get("decode_success_ratio"),
                    "black_ratio": group.get("black_ratio"),
                    "flat_ratio": group.get("flat_ratio"),
                    "motion_ratio": group.get("motion_ratio"),
                    "contact_sheet": group.get("contact_sheet"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_view_summary(rows: list[dict[str, Any]]) -> None:
    print("\n== Camera/view summary ==")
    if not rows:
        print("No video groups found.")
        return
    def format_ratio(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    for row in rows:
        print(
            f"dataset={Path(str(row['dataset_root'])).name} key={row['camera_key']} "
            f"role={row['role_hint']} files={row['total_files']} "
            f"small={format_ratio(row['small_file_ratio'])} "
            f"duplicate={format_ratio(row['repeated_hash_ratio'])} "
            f"decode={format_ratio(row['decode_success_ratio'])} black={format_ratio(row['black_ratio'])} "
            f"flat={format_ratio(row['flat_ratio'])} motion={format_ratio(row['motion_ratio'])}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    audits: list[dict[str, Any]] = []

    for raw_root in args.root:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            audits.append({"root": str(root), "error": "Root does not exist."})
            continue
        discovery = discover_tree(root, args.max_depth, args.max_files)
        lerobot_datasets = [
            audit_lerobot_dataset(info_path, output_dir=output_dir / safe_name(root.name), args=args, rng=rng)
            for info_path in discovery.pop("lerobot_info_paths")
        ]
        known_lerobot_video_roots = {
            Path(dataset["dataset_root"]) / "videos"
            for dataset in lerobot_datasets
            if "dataset_root" in dataset
        }
        generic_groups = []
        sorted_groups = sorted(
            discovery.pop("generic_video_groups").items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        for camera_key, paths in sorted_groups:
            if any(any(video_root == parent or video_root in parent.parents for video_root in known_lerobot_video_roots) for parent in [path.parent for path in paths[:1]]):
                continue
            generic_groups.append(
                audit_video_group(
                    paths,
                    camera_key=camera_key,
                    dataset_root=root,
                    output_dir=output_dir / safe_name(root.name) / "generic",
                    args=args,
                    rng=rng,
                )
            )
            if len(generic_groups) >= args.max_generic_video_groups:
                break
        dataset_info_paths = discovery.pop("dataset_info_paths")
        discovery["dataset_info_summaries"] = [summarize_dataset_info(path) for path in dataset_info_paths]
        hdf5_paths = []
        for suffix in (".h5", ".hdf5"):
            for relative_path in discovery["sample_files_by_suffix"].get(suffix, []):
                hdf5_paths.append(root / relative_path)
        discovery["hdf5_summaries"] = [
            summarize_hdf5(path) for path in hdf5_paths[: args.inspect_hdf5_files]
        ]
        discovery["lerobot_datasets"] = lerobot_datasets
        discovery["generic_video_groups"] = generic_groups
        audits.append(discovery)

    report_path = output_dir / "dataset_view_audit.json"
    report_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    rows = flatten_view_rows(audits)
    csv_path = output_dir / "view_summary.csv"
    write_csv(csv_path, rows)
    print_view_summary(rows)
    print(f"\nreport={report_path}")
    print(f"view_summary={csv_path}")
    print("Role hints are naming-based candidates only; confirm them from contact sheets before training.")


if __name__ == "__main__":
    main()
