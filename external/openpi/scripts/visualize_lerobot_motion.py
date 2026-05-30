#!/usr/bin/env python
"""Visualize temporal motion in a local LeRobot dataset.

This script supports the PV-VAE/OXE debugging path. It samples clips from a
local LeRobot `parquet + mp4` dataset and writes side-by-side RGB frames plus
frame-to-first difference maps. The goal is to inspect whether sampled clips
actually contain visible motion before training a predictive tokenizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


EPISODE_RE = re.compile(r"episode_(\d+)")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--future-deltas", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    parser.add_argument("--num-clips", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=512)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--min-rgb-delta", type=float, default=0.0)
    parser.add_argument("--max-resample-attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--diff-scale", type=float, default=6.0)
    return parser.parse_args()


def episode_index_from_path(path: Path) -> int:
    match = EPISODE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def parquet_table(path: Path):
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq.read_table(path)
    except Exception as exc:
        raise RuntimeError(f"pyarrow is required to inspect parquet: {exc}") from exc


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


def tensor_to_pil(frame: torch.Tensor, size: int) -> Image.Image:
    image = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    pil = Image.fromarray((image * 255.0).astype(np.uint8))
    return pil.resize((size, size), Image.Resampling.BILINEAR)


def diff_to_pil(frame: torch.Tensor, first: torch.Tensor, size: int, scale: float) -> Image.Image:
    diff = (frame - first).abs().mean(dim=0).detach().cpu().clamp(0, 1).numpy()
    diff = np.clip(diff * scale, 0.0, 1.0)
    heat = np.zeros((*diff.shape, 3), dtype=np.float32)
    heat[..., 0] = diff
    heat[..., 1] = np.sqrt(diff) * 0.35
    heat[..., 2] = 1.0 - diff
    pil = Image.fromarray((heat * 255.0).astype(np.uint8))
    return pil.resize((size, size), Image.Resampling.BILINEAR)


def rgb_delta_metrics(frames: torch.Tensor) -> tuple[float, float, list[float]]:
    deltas = (frames[1:] - frames[:1]).abs().mean(dim=(1, 2, 3))
    adjacent = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
    return float(deltas.max()), float(adjacent.mean()), [float(v) for v in deltas]


def action_metrics(action: np.ndarray | None, frame_indices: list[int]) -> tuple[float | None, float | None]:
    if action is None:
        return None, None
    clip = np.asarray(action[frame_indices])
    first_last = float(np.linalg.norm(clip[-1] - clip[0]))
    adjacent = float(np.linalg.norm(clip[1:] - clip[:-1], axis=-1).mean())
    return first_last, adjacent


def build_candidates(root: Path, video_key: str, future_deltas: tuple[int, ...], max_episodes: int, samples_per_episode: int, seed: int):
    rng = random.Random(seed)
    video_paths = sorted((root / "videos").glob(f"chunk-*/{video_key}/episode_*.mp4"))
    if not video_paths:
        video_paths = sorted((root / "videos").glob(f"chunk-*/*{video_key}*/episode_*.mp4"))
    videos = {episode_index_from_path(path): path for path in video_paths}
    parquet_paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    candidates = []
    max_delta = max(future_deltas)

    episodes_seen = 0
    for parquet_path in parquet_paths:
        episode_idx = episode_index_from_path(parquet_path)
        if episode_idx not in videos:
            continue
        table = parquet_table(parquet_path)
        if table.num_rows <= max_delta:
            continue
        max_start = int(table.num_rows) - 1 - max_delta
        starts = list(range(max_start + 1))
        if samples_per_episode > 0 and len(starts) > samples_per_episode:
            starts = rng.sample(starts, samples_per_episode)
        action = table_column_numpy(table, "action")
        for start in starts:
            candidates.append(
                {
                    "episode_idx": episode_idx,
                    "video_path": videos[episode_idx],
                    "start": start,
                    "action": action,
                }
            )
        episodes_seen += 1
        if max_episodes > 0 and episodes_seen >= max_episodes:
            break
    rng.shuffle(candidates)
    return candidates


def draw_clip(
    output_path: Path,
    frames: torch.Tensor,
    *,
    title: str,
    frame_labels: list[str],
    cell_size: int,
    diff_scale: float,
) -> None:
    num_frames = frames.shape[0]
    label_h = 34
    title_h = 52
    rows = 2
    width = num_frames * cell_size
    height = title_h + rows * (cell_size + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)

    first = frames[0]
    for idx in range(num_frames):
        x = idx * cell_size
        rgb = tensor_to_pil(frames[idx], cell_size)
        diff = diff_to_pil(frames[idx], first, cell_size, diff_scale)
        canvas.paste(rgb, (x, title_h))
        canvas.paste(diff, (x, title_h + cell_size + label_h))
        draw.text((x + 4, title_h + cell_size + 6), frame_labels[idx], fill=(0, 0, 0), font=font)
        draw.text((x + 4, title_h + 2 * cell_size + label_h + 6), f"diff {idx}", fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    root = Path(args.lerobot_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    future_deltas = parse_int_list(args.future_deltas)
    frame_offsets = [0, *future_deltas]
    info = json.loads((root / "meta" / "info.json").read_text())

    candidates = build_candidates(
        root,
        args.video_key,
        future_deltas,
        args.max_episodes,
        args.samples_per_episode,
        args.seed,
    )
    if not candidates:
        raise ValueError("No candidates found.")

    manifest = []
    selected = 0
    attempts = 0
    for candidate in candidates:
        attempts += 1
        frame_indices = [candidate["start"] + offset for offset in frame_offsets]
        frames = read_video_frames(candidate["video_path"], frame_indices)
        rgb_max, rgb_adj, per_frame_delta = rgb_delta_metrics(frames)
        if rgb_max < args.min_rgb_delta and attempts < args.max_resample_attempts:
            continue
        action_fl, action_adj = action_metrics(candidate["action"], frame_indices)

        title = (
            f"{root.name} ep={candidate['episode_idx']} start={candidate['start']} fps={info.get('fps')} "
            f"rgb_max={rgb_max:.4f} rgb_adj={rgb_adj:.4f} "
            f"action_fl={action_fl if action_fl is not None else 'NA'}"
        )
        labels = [f"t+{offset}" for offset in frame_offsets]
        filename = f"clip_{selected:04d}_ep{candidate['episode_idx']:06d}_s{candidate['start']:06d}_rgb{rgb_max:.4f}.png"
        draw_clip(
            output_dir / filename,
            frames,
            title=title,
            frame_labels=labels,
            cell_size=args.cell_size,
            diff_scale=args.diff_scale,
        )
        manifest.append(
            {
                "file": filename,
                "episode_idx": candidate["episode_idx"],
                "start": candidate["start"],
                "frame_offsets": frame_offsets,
                "rgb_max_delta_to_first": rgb_max,
                "rgb_adjacent_delta": rgb_adj,
                "per_frame_delta_to_first": per_frame_delta,
                "action_first_last_l2": action_fl,
                "action_adjacent_l2": action_adj,
            }
        )
        selected += 1
        if selected >= args.num_clips:
            break

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"root={root}")
    print(f"output_dir={output_dir}")
    print(f"future_deltas={future_deltas}")
    print(f"selected={selected} attempts={attempts} min_rgb_delta={args.min_rgb_delta}")
    if selected < args.num_clips:
        print("warning: selected fewer clips than requested")


if __name__ == "__main__":
    main()
