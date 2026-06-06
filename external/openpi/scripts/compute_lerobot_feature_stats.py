#!/usr/bin/env python
"""Compute frozen-teacher feature statistics for local LeRobot clips.

This is the first PV-VAE sanity check for OXE / Bridge. It measures whether
SVG-P / DINO features have channel-scale imbalance before we train a feature
VAE with raw MSE. The produced `.pt` file can be passed to
`train_predictive_feature_vae_lerobot.py --feature-normalization channel_standard`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument(
        "--dataset-spec-json",
        default=None,
        help="Optional JSON list/dict of LeRobot sources; uses the same format as train_predictive_feature_vae_lerobot.py.",
    )
    parser.add_argument("--output", required=True, help="Output .pt path.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--teacher", choices=("svg_p", "dinov3_vits16"), default="svg_p")
    parser.add_argument("--dinov3-path", default="assets/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--video-keys", default="observation.images.image")
    parser.add_argument("--future-deltas", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    parser.add_argument("--time-sampling-mode", choices=("frame_deltas", "duration_sec"), default="frame_deltas")
    parser.add_argument("--clip-duration-sec", type=float, default=None)
    parser.add_argument("--num-future-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=512)
    parser.add_argument("--episode-offset", type=int, default=0, help="Skip this many valid episodes before sampling clips.")
    parser.add_argument("--samples-per-episode", type=int, default=64)
    parser.add_argument("--mixture-samples-per-epoch", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--min-rgb-delta", type=float, default=0.0)
    parser.add_argument("--min-rgb-mean", type=float, default=0.0)
    parser.add_argument("--min-rgb-std", type=float, default=0.0)
    parser.add_argument("--max-resample-attempts", type=int, default=200)
    parser.add_argument("--temporal-compression", type=int, default=4)
    parser.add_argument("--encoder-microbatch", type=int, default=16)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--svg-autoencoder-root", default=None)
    parser.add_argument("--svg-config", default=None)
    parser.add_argument("--svg-checkpoint", default=None)
    parser.add_argument("--svg-dinov3-weights", default=None)
    parser.add_argument("--svg-feature-dim", type=int, default=384)
    parser.add_argument("--teacher-image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def encode_clip(
    *,
    teacher: str,
    encoder: torch.nn.Module,
    images: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    import train_predictive_feature_vae_libero as pv_utils

    if teacher == "svg_p":
        return pv_utils.encode_svg_p_clip(
            encoder,
            images,
            microbatch=args.encoder_microbatch,
            precision=args.precision,
            image_size=args.teacher_image_size,
        )
    return pv_utils.encode_dino_clip(
        encoder,
        images,
        microbatch=args.encoder_microbatch,
        precision=args.precision,
    )


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().cpu(), q).item())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import train_lam_libero as lam_utils
    import train_predictive_feature_vae_lerobot as lerobot_train

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    future_deltas = parse_int_list(args.future_deltas)
    dataset, dataset_summary, _, _, video_keys = lerobot_train.build_clip_dataset(args, fallback_future_deltas=future_deltas)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lerobot_train.collate_clip_batch,
    )

    if args.teacher == "svg_p":
        svg_args = argparse.Namespace(**vars(args))
        svg_args.feature_normalization = "none"
        encoder = lam_utils.load_svg_decoder(svg_args, device)
    else:
        from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder

        encoder = load_dinov3_patch_encoder(args.dinov3_path).to(device).eval()

    sum_channels: torch.Tensor | None = None
    sumsq_channels: torch.Tensor | None = None
    total_count = 0
    token_norm_sum = 0.0
    token_norm_sumsq = 0.0
    token_count = 0
    rgb_delta_values: list[float] = []
    rgb_mean_values: list[float] = []
    rgb_std_values: list[float] = []

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > args.max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        features = encode_clip(teacher=args.teacher, encoder=encoder, images=images, args=args).float()
        flat = features.reshape(-1, features.shape[-1])
        channel_sum = flat.sum(dim=0).cpu()
        channel_sumsq = flat.pow(2).sum(dim=0).cpu()
        if sum_channels is None:
            sum_channels = channel_sum
            sumsq_channels = channel_sumsq
        else:
            sum_channels += channel_sum
            assert sumsq_channels is not None
            sumsq_channels += channel_sumsq
        total_count += flat.shape[0]

        norms = flat.norm(dim=-1).cpu()
        token_norm_sum += float(norms.sum())
        token_norm_sumsq += float(norms.pow(2).sum())
        token_count += int(norms.numel())
        rgb_delta_values.extend(float(x) for x in batch["rgb_delta"].flatten())
        rgb_mean_values.extend(float(x) for x in batch["rgb_mean"].flatten())
        rgb_std_values.extend(float(x) for x in batch["rgb_std"].flatten())

        if batch_idx % 20 == 0:
            logging.info("processed batches=%d feature_tokens=%d", batch_idx, total_count)

    if sum_channels is None or sumsq_channels is None or total_count == 0:
        raise RuntimeError("No feature batches were processed.")

    mean = sum_channels / total_count
    var = (sumsq_channels / total_count - mean.pow(2)).clamp_min(1e-12)
    std = var.sqrt()
    token_norm_mean = token_norm_sum / max(token_count, 1)
    token_norm_var = token_norm_sumsq / max(token_count, 1) - token_norm_mean**2
    token_norm_std = max(token_norm_var, 0.0) ** 0.5

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "std": std,
            "var": var,
            "count": total_count,
            "teacher": args.teacher,
            "future_deltas": future_deltas,
            "dataset_summary": dataset_summary,
            "video_keys": video_keys,
            "args": vars(args),
        },
        output,
    )

    summary: dict[str, Any] = {
        "output": str(output),
        "teacher": args.teacher,
        "dataset_summary": dataset_summary,
        "feature_dim": int(mean.numel()),
        "feature_tokens": int(total_count),
        "channel_mean_abs_mean": float(mean.abs().mean()),
        "channel_mean_abs_max": float(mean.abs().max()),
        "channel_std_mean": float(std.mean()),
        "channel_std_min": float(std.min()),
        "channel_std_max": float(std.max()),
        "channel_std_p01": percentile(std, 0.01),
        "channel_std_p10": percentile(std, 0.10),
        "channel_std_p50": percentile(std, 0.50),
        "channel_std_p90": percentile(std, 0.90),
        "channel_std_p99": percentile(std, 0.99),
        "channel_std_max_over_min": float(std.max() / std.min().clamp_min(1e-12)),
        "token_norm_mean": token_norm_mean,
        "token_norm_std": token_norm_std,
        "rgb_delta_mean": float(np.mean(rgb_delta_values)) if rgb_delta_values else 0.0,
        "rgb_mean_mean": float(np.mean(rgb_mean_values)) if rgb_mean_values else 0.0,
        "rgb_std_mean": float(np.mean(rgb_std_values)) if rgb_std_values else 0.0,
    }
    summary_path = Path(args.summary_json) if args.summary_json else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("wrote stats to %s", output)
    logging.info("summary=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
