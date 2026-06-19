#!/usr/bin/env python
"""Visualize clips exactly as the PV-VAE LeRobot training loader samples them.

This is a loader sanity-check script.  It reuses
train_predictive_feature_vae_lerobot.build_clip_dataset so the sampled clips,
time offsets, motion filters, dataset spec, and image resizing match PV-VAE
training.  Use it before multi-dataset training when adding a new dataset or
camera source.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import train_predictive_feature_vae_lerobot as lerobot_train
from visualize_lerobot_motion import draw_clip, rgb_delta_metrics, rgb_quality_metrics


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument("--dataset-spec-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-keys", default="observation.images.image")
    parser.add_argument("--future-deltas", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    parser.add_argument("--time-sampling-mode", choices=("frame_deltas", "duration_sec"), default="frame_deltas")
    parser.add_argument("--clip-duration-sec", type=float, default=None)
    parser.add_argument("--num-future-frames", type=int, default=None)
    parser.add_argument("--num-clips", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=512)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--mixture-samples-per-epoch", type=int, default=0)
    parser.add_argument(
        "--mixture-source-batch-mode",
        choices=("sample", "homogeneous"),
        default="sample",
        help="Use 'homogeneous' when mixed sources have different numbers of camera views.",
    )
    parser.add_argument("--min-rgb-delta", type=float, default=0.0)
    parser.add_argument("--min-rgb-mean", type=float, default=0.0)
    parser.add_argument("--min-rgb-std", type=float, default=0.0)
    parser.add_argument("--max-resample-attempts", type=int, default=200)
    parser.add_argument("--teacher-image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--diff-scale", type=float, default=6.0)
    return parser.parse_args()


def item_from_batch(batch: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "images": batch["images"][index],
        "rgb_delta": float(batch["rgb_delta"][index]),
        "rgb_mean": float(batch["rgb_mean"][index]),
        "rgb_std": float(batch["rgb_std"][index]),
        "dataset_name": batch["dataset_name"][index],
        "source_name": batch["source_name"][index],
        "episode_id": batch["episode_id"][index],
        "episode_uid": batch["episode_uid"][index],
        "start_index": int(batch["start_index"][index]),
        "frame_offsets": batch["frame_offsets"][index],
        "future_deltas": batch["future_deltas"][index],
        "time_offsets_sec": batch["time_offsets_sec"][index],
        "fps": float(batch["fps"][index]),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    future_deltas = parse_int_list(args.future_deltas)
    dataset, dataset_summary, _, _, _ = lerobot_train.build_clip_dataset(
        args,
        fallback_future_deltas=future_deltas,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lerobot_train.collate_clip_batch,
    )

    manifest: list[dict[str, Any]] = []
    written = 0
    for batch in loader:
        for batch_index in range(batch["images"].shape[0]):
            item = item_from_batch(batch, batch_index)
            images = item["images"]
            for view_index in range(images.shape[0]):
                frames = images[view_index]
                rgb_max, rgb_adj, per_frame_delta = rgb_delta_metrics(frames)
                rgb_mean, rgb_std = rgb_quality_metrics(frames)
                offsets = list(item["frame_offsets"] or range(frames.shape[0]))
                labels = [f"t+{offset}" for offset in offsets]
                title = (
                    f"{item['source_name']} uid={item['episode_uid']} start={item['start_index']} "
                    f"view={view_index} fps={item['fps']:.3f} rgb_max={rgb_max:.4f} "
                    f"rgb_adj={rgb_adj:.4f} rgb_mean={rgb_mean:.4f} rgb_std={rgb_std:.4f}"
                )
                filename = (
                    f"clip_{written:04d}_{item['source_name']}_v{view_index}_"
                    f"s{item['start_index']:06d}_rgb{rgb_max:.4f}.png"
                )
                path = output_dir / filename
                draw_clip(
                    path,
                    frames,
                    title=title,
                    frame_labels=labels,
                    cell_size=args.cell_size,
                    diff_scale=args.diff_scale,
                )
                manifest.append(
                    {
                        "file": str(path),
                        "source_name": item["source_name"],
                        "dataset_name": item["dataset_name"],
                        "episode_id": item["episode_id"],
                        "episode_uid": item["episode_uid"],
                        "start_index": item["start_index"],
                        "view_index": view_index,
                        "fps": item["fps"],
                        "frame_offsets": offsets,
                        "future_deltas": list(item["future_deltas"] or []),
                        "time_offsets_sec": list(item["time_offsets_sec"] or []),
                        "rgb_max_delta_to_first": rgb_max,
                        "rgb_adjacent_mean_abs": rgb_adj,
                        "rgb_mean": rgb_mean,
                        "rgb_std": rgb_std,
                        "loader_rgb_delta": item["rgb_delta"],
                        "loader_rgb_mean": item["rgb_mean"],
                        "loader_rgb_std": item["rgb_std"],
                        "per_frame_delta_to_first": per_frame_delta,
                    }
                )
                written += 1
                if written >= args.num_clips:
                    (output_dir / "manifest.json").write_text(
                        json.dumps(
                            {"dataset_summary": dataset_summary, "clips": manifest},
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    logging.info("Wrote %d clips to %s", written, output_dir)
                    return

    (output_dir / "manifest.json").write_text(
        json.dumps({"dataset_summary": dataset_summary, "clips": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Wrote %d clips to %s", written, output_dir)


if __name__ == "__main__":
    main()
