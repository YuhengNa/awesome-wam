#!/usr/bin/env python
"""Inspect temporal motion statistics in a local LeRobot dataset.

This is the diagnostic step before PV-VAE training on OXE / Bridge. It checks
whether sampled clips actually contain motion signal, instead of letting a
static-copy baseline dominate the predictive objective.

The script reads local LeRobot `parquet + mp4` directories and reports:

- RGB temporal deltas
- action deltas / action norms when present in parquet
- static clip ratios under simple thresholds

It is read-only and does not load SVG-P/DINO teachers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch


EPISODE_RE = re.compile(r"episode_(\d+)")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--future-deltas", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    parser.add_argument("--max-episodes", type=int, default=256)
    parser.add_argument("--samples-per-episode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-static-threshold", type=float, default=0.005)
    parser.add_argument("--rgb-black-mean-threshold", type=float, default=0.01)
    parser.add_argument("--rgb-flat-std-threshold", type=float, default=0.01)
    parser.add_argument("--action-static-threshold", type=float, default=1e-4)
    return parser.parse_args()


def episode_index_from_path(path: Path) -> int:
    match = EPISODE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def episode_key_from_data_path(path: Path) -> tuple[str, int]:
    return path.parent.name, episode_index_from_path(path)


def episode_key_from_video_path(path: Path) -> tuple[str, int]:
    return path.parent.parent.name, episode_index_from_path(path)


def parquet_table(path: Path):
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq.read_table(path)
    except Exception as exc:
        raise RuntimeError(f"pyarrow is required to inspect parquet efficiently: {exc}") from exc


def table_num_rows(table: Any) -> int:
    return int(table.num_rows)


def table_column_names(table: Any) -> list[str]:
    return list(table.column_names)


def table_column_numpy(table: Any, name: str) -> np.ndarray | None:
    if name not in table.column_names:
        return None
    return np.asarray(table[name].to_pylist())


def read_video_frames(path: Path, frame_indices: list[int]) -> torch.Tensor:
    try:
        from torchvision.io import read_video  # type: ignore

        video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
        return video[torch.as_tensor(frame_indices, dtype=torch.long)].float().div(255.0).clamp(0.0, 1.0)
    except Exception as first_error:
        try:
            import imageio.v3 as iio  # type: ignore

            arr = iio.imread(path, index=frame_indices)
            frames = torch.as_tensor(arr)
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            return frames.permute(0, 3, 1, 2).float().div(255.0).clamp(0.0, 1.0)
        except Exception as second_error:
            raise RuntimeError(f"Failed to read {path}: {first_error}; fallback: {second_error}") from second_error


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
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


def print_summary(name: str, values: list[float]) -> None:
    stats = summarize(values)
    if not stats:
        print(f"{name}: no values")
        return
    rendered = " ".join(f"{key}={value:.6f}" for key, value in stats.items())
    print(f"{name}: {rendered}")


def main() -> None:
    args = parse_args()
    root = Path(args.lerobot_root).expanduser().resolve()
    future_deltas = parse_int_list(args.future_deltas)
    frame_offsets = [0, *future_deltas]
    rng = random.Random(args.seed)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    print(f"root={root}")
    print(f"fps={info.get('fps')}")
    print(f"video_key={args.video_key}")
    print(f"future_deltas={future_deltas}")

    video_paths = sorted((root / "videos").glob(f"chunk-*/{args.video_key}/episode_*.mp4"))
    if not video_paths:
        video_paths = sorted((root / "videos").glob(f"chunk-*/*{args.video_key}*/episode_*.mp4"))
    video_by_episode = {episode_key_from_video_path(path): path for path in video_paths}
    video_index_only = Counter(episode_index_from_path(path) for path in video_paths)

    parquet_paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    episodes: list[tuple[int, tuple[str, int], Path, Any]] = []
    action_columns: Counter[str] = Counter()
    for parquet_path in parquet_paths:
        episode_idx = episode_index_from_path(parquet_path)
        episode_key = episode_key_from_data_path(parquet_path)
        if episode_key not in video_by_episode:
            continue
        table = parquet_table(parquet_path)
        for name in table_column_names(table):
            if "action" in name:
                action_columns[name] += 1
        if table_num_rows(table) <= max(future_deltas):
            continue
        episodes.append((episode_idx, episode_key, parquet_path, table))
        if args.max_episodes > 0 and len(episodes) >= args.max_episodes:
            break

    print(f"episodes={len(episodes)} videos={len(video_by_episode)} parquet_files={len(parquet_paths)}")
    print(f"duplicate_episode_ids_across_video_chunks={sum(count > 1 for count in video_index_only.values())}")
    print(f"action_columns={dict(action_columns)}")
    if not episodes:
        raise ValueError("No usable episodes found.")

    rgb_first_last: list[float] = []
    rgb_adjacent: list[float] = []
    rgb_max_frame_delta: list[float] = []
    rgb_mean_values: list[float] = []
    rgb_std_values: list[float] = []
    action_first_last: list[float] = []
    action_adjacent: list[float] = []
    action_norm: list[float] = []
    sampled = 0
    failed_videos = 0
    max_delta = max(future_deltas)

    for episode_idx, episode_key, _, table in episodes:
        num_rows = table_num_rows(table)
        max_start = num_rows - 1 - max_delta
        starts = list(range(max_start + 1))
        if args.samples_per_episode > 0 and len(starts) > args.samples_per_episode:
            starts = rng.sample(starts, args.samples_per_episode)

        action = table_column_numpy(table, "action")
        for start in starts:
            frame_indices = [start + offset for offset in frame_offsets]
            try:
                frames = read_video_frames(video_by_episode[episode_key], frame_indices)
            except Exception:
                failed_videos += 1
                continue
            diff_to_first = (frames[1:] - frames[:1]).abs().mean(dim=(1, 2, 3))
            adjacent = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
            rgb_first_last.append(float(diff_to_first[-1]))
            rgb_adjacent.append(float(adjacent.mean()))
            rgb_max_frame_delta.append(float(diff_to_first.max()))
            rgb_mean_values.append(float(frames.mean()))
            rgb_std_values.append(float(frames.std()))

            if action is not None:
                action_clip = np.asarray(action[frame_indices])
                action_diff = np.linalg.norm(action_clip[1:] - action_clip[:1], axis=-1)
                action_adj = np.linalg.norm(action_clip[1:] - action_clip[:-1], axis=-1)
                action_first_last.append(float(action_diff[-1]))
                action_adjacent.append(float(action_adj.mean()))
                action_norm.append(float(np.linalg.norm(action_clip[:-1], axis=-1).mean()))
            sampled += 1

    print(f"sampled_clips={sampled} failed_videos={failed_videos}")
    print_summary("rgb_first_last_mean_abs", rgb_first_last)
    print_summary("rgb_adjacent_mean_abs", rgb_adjacent)
    print_summary("rgb_max_delta_to_first", rgb_max_frame_delta)
    print_summary("rgb_mean", rgb_mean_values)
    print_summary("rgb_std", rgb_std_values)
    print_summary("action_first_last_l2", action_first_last)
    print_summary("action_adjacent_l2", action_adjacent)
    print_summary("action_norm_l2", action_norm)

    if sampled > 0:
        rgb_static = sum(value < args.rgb_static_threshold for value in rgb_max_frame_delta) / sampled
        print(f"rgb_static_ratio@{args.rgb_static_threshold:.6f}={rgb_static:.6f}")
        rgb_black = sum(value < args.rgb_black_mean_threshold for value in rgb_mean_values) / sampled
        rgb_flat = sum(value < args.rgb_flat_std_threshold for value in rgb_std_values) / sampled
        print(f"rgb_black_ratio@mean<{args.rgb_black_mean_threshold:.6f}={rgb_black:.6f}")
        print(f"rgb_flat_ratio@std<{args.rgb_flat_std_threshold:.6f}={rgb_flat:.6f}")
    if action_norm:
        action_static = sum(value < args.action_static_threshold for value in action_norm) / len(action_norm)
        print(f"action_static_ratio@{args.action_static_threshold:.6f}={action_static:.6f}")


if __name__ == "__main__":
    main()
