#!/usr/bin/env python
"""Visualize a trained PV-VAE checkpoint on fixed LeRobot clips.

This is the post-training diagnostic path for OXE / Bridge:

    checkpoint -> fixed filtered clips -> teacher features -> PV-VAE prediction
    -> GT / Pred / Delta / Error visualization

It is meant to answer whether the model predicts enough temporal change, not
just whether the scalar training loss decreases.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from openpi.models_pytorch.predictive_feature_vae import PredictiveFeatureVAE, PredictiveFeatureVAEConfig


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument("--dataset-spec-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", choices=("svg_p", "dinov3_vits16"), default=None)
    parser.add_argument("--dinov3-path", default=None)
    parser.add_argument("--video-keys", default=None)
    parser.add_argument("--future-deltas", default=None)
    parser.add_argument("--time-sampling-mode", choices=("frame_deltas", "duration_sec"), default=None)
    parser.add_argument("--clip-duration-sec", type=float, default=None)
    parser.add_argument("--num-future-frames", type=int, default=None)
    parser.add_argument("--observed-groups", type=int, default=None)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument(
        "--allow-duplicate-clips",
        action="store_true",
        help="Allow repeated (episode_id, start_index) clips in the visualization set.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=512)
    parser.add_argument("--episode-offset", type=int, default=None)
    parser.add_argument("--samples-per-episode", type=int, default=64)
    parser.add_argument("--mixture-samples-per-epoch", type=int, default=0)
    parser.add_argument("--min-rgb-delta", type=float, default=None)
    parser.add_argument("--min-rgb-mean", type=float, default=None)
    parser.add_argument("--min-rgb-std", type=float, default=None)
    parser.add_argument("--max-resample-attempts", type=int, default=200)
    parser.add_argument("--feature-normalization", choices=("none", "l2", "token_layer_norm", "channel_standard"), default=None)
    parser.add_argument("--feature-stats", default=None)
    parser.add_argument("--encoder-microbatch", type=int, default=None)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default=None)
    parser.add_argument("--decode-svg-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svg-autoencoder-root", default=None)
    parser.add_argument("--svg-config", default=None)
    parser.add_argument("--svg-checkpoint", default=None)
    parser.add_argument("--svg-dinov3-weights", default=None)
    parser.add_argument("--svg-decode-grid", type=int, default=None)
    parser.add_argument("--teacher-image-size", type=int, default=None)
    parser.add_argument("--vis-max-frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def arg_or_checkpoint(args: argparse.Namespace, checkpoint_args: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return checkpoint_args.get(name.replace("_", "-"), checkpoint_args.get(name, default))


def load_checkpoint(path: Path, device: torch.device) -> tuple[PredictiveFeatureVAE, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = PredictiveFeatureVAEConfig(**checkpoint["model_config"])
    model = PredictiveFeatureVAE(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("args", {}), checkpoint


def prepare_model_args(args: argparse.Namespace, checkpoint_args: dict[str, Any]) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    for name, default in (
        ("teacher", "svg_p"),
        ("dinov3_path", "assets/dinov3-vits16-pretrain-lvd1689m"),
        ("video_keys", "observation.images.image"),
        ("future_deltas", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"),
        ("dataset_spec_json", None),
        ("time_sampling_mode", "frame_deltas"),
        ("clip_duration_sec", None),
        ("num_future_frames", None),
        ("observed_groups", 1),
        ("episode_offset", 0),
        ("min_rgb_delta", 0.0),
        ("min_rgb_mean", 0.0),
        ("min_rgb_std", 0.0),
        ("feature_normalization", "none"),
        ("feature_stats", None),
        ("encoder_microbatch", 16),
        ("precision", "bfloat16"),
        ("svg_autoencoder_root", None),
        ("svg_config", None),
        ("svg_checkpoint", None),
        ("svg_dinov3_weights", None),
        ("svg_decode_grid", 16),
        ("teacher_image_size", 256),
    ):
        setattr(resolved, name, arg_or_checkpoint(args, checkpoint_args, name, default))
    return resolved


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


def sample_metrics(features: torch.Tensor, pred: torch.Tensor, observed_frames: int) -> dict[str, float]:
    # features/pred are one view with shape [T,N,D]; temporal deltas are along T.
    pred_delta = pred[1:] - pred[:-1]
    target_delta = features[1:] - features[:-1]
    pred_d = float(pred_delta.float().norm(dim=-1).mean().cpu())
    gt_d = float(target_delta.float().norm(dim=-1).mean().cpu())
    future_sqerr = (pred[observed_frames:] - features[observed_frames:]).float().pow(2)
    future_mse_by_frame = future_sqerr.mean(dim=(1, 2))
    future_mse = float(future_mse_by_frame.mean().cpu())
    static = features[observed_frames - 1 : observed_frames].expand_as(features[observed_frames:])
    static_sqerr = (static - features[observed_frames:]).float().pow(2)
    static_mse_by_frame = static_sqerr.mean(dim=(1, 2))
    static_mse = float(static_mse_by_frame.mean().cpu())
    full_mse = float((pred.float() - features.float()).pow(2).mean().cpu())
    future_copy_ratio_by_frame = future_mse_by_frame / static_mse_by_frame.clamp_min(1e-6)
    pred_delta_norm_by_frame = pred_delta.float().norm(dim=-1).mean(dim=1)
    gt_delta_norm_by_frame = target_delta.float().norm(dim=-1).mean(dim=1)
    delta_ratio_by_frame = pred_delta_norm_by_frame / gt_delta_norm_by_frame.clamp_min(1e-6)
    return {
        "feature_mse": full_mse,
        "future_mse": future_mse,
        "static_future_mse": static_mse,
        "future_copy_ratio": future_mse / max(static_mse, 1e-6),
        "pred_d": pred_d,
        "gt_d": gt_d,
        "d_ratio": pred_d / max(gt_d, 1e-6),
        "future_mse_by_frame": [float(value.cpu()) for value in future_mse_by_frame],
        "static_future_mse_by_frame": [float(value.cpu()) for value in static_mse_by_frame],
        "future_copy_ratio_by_frame": [float(value.cpu()) for value in future_copy_ratio_by_frame],
        "pred_delta_norm_by_frame": [float(value.cpu()) for value in pred_delta_norm_by_frame],
        "gt_delta_norm_by_frame": [float(value.cpu()) for value in gt_delta_norm_by_frame],
        "delta_ratio_by_frame": [float(value.cpu()) for value in delta_ratio_by_frame],
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _aggregate_scalar(manifest: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(item[key]) for item in manifest]
    return {
        "mean": _mean(values),
        "median": _median(values),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def _aggregate_vector(manifest: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    vectors = [item[key] for item in manifest if item.get(key)]
    if not vectors:
        return {"mean": [], "median": [], "min": [], "max": []}
    width = min(len(vector) for vector in vectors)
    columns = [[float(vector[i]) for vector in vectors] for i in range(width)]
    return {
        "mean": [_mean(column) for column in columns],
        "median": [_median(column) for column in columns],
        "min": [float(min(column)) for column in columns],
        "max": [float(max(column)) for column in columns],
    }


def summarize_manifest(manifest: list[dict[str, Any]], observed_frames: int) -> dict[str, Any]:
    unique_keys = {(item["episode_id"], item["start_index"]) for item in manifest}
    ratio_values = [float(item["future_copy_ratio"]) for item in manifest]
    summary = {
        "num_clips": len(manifest),
        "num_unique_episode_start": len(unique_keys),
        "observed_frames": observed_frames,
        "copy_better_count": sum(value < 1.0 for value in ratio_values),
        "copy_better_fraction": _mean([1.0 if value < 1.0 else 0.0 for value in ratio_values]),
        "strong_copy_better_count": sum(value <= 0.75 for value in ratio_values),
        "copy_worse_count": sum(value > 1.0 for value in ratio_values),
        "copy_much_worse_count": sum(value > 1.5 for value in ratio_values),
        "scalars": {},
        "by_future_frame": {},
    }
    for key in (
        "feature_mse",
        "future_mse",
        "static_future_mse",
        "future_copy_ratio",
        "pred_d",
        "gt_d",
        "d_ratio",
        "rgb_delta",
        "rgb_mean",
        "rgb_std",
    ):
        summary["scalars"][key] = _aggregate_scalar(manifest, key)
    for key in (
        "future_mse_by_frame",
        "static_future_mse_by_frame",
        "future_copy_ratio_by_frame",
        "pred_delta_norm_by_frame",
        "gt_delta_norm_by_frame",
        "delta_ratio_by_frame",
    ):
        summary["by_future_frame"][key] = _aggregate_vector(manifest, key)
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import train_lam_libero as lam_utils
    import train_predictive_feature_vae_libero as pv_utils
    import train_predictive_feature_vae_lerobot as lerobot_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, checkpoint_args, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    args = prepare_model_args(args, checkpoint_args)
    future_deltas = parse_int_list(args.future_deltas)
    dataset, dataset_summary, num_future_frames, _, video_keys = lerobot_train.build_clip_dataset(
        args,
        fallback_future_deltas=future_deltas,
    )
    observed_groups = min(max(int(args.observed_groups), 1), 1 + num_future_frames // model.config.temporal_compression)
    observed_frames = 1 + (observed_groups - 1) * model.config.temporal_compression

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
        svg_model = encoder if args.decode_svg_rgb else None
    else:
        from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder

        encoder = load_dinov3_patch_encoder(args.dinov3_path, device)
        svg_model = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True), encoding="utf-8")
    feature_stats = lerobot_train.load_feature_stats(args.feature_stats, device)
    logging.info(
        "checkpoint=%s step=%s clips=%d teacher=%s observed_groups=%d observed_frames=%d sources=%s",
        args.checkpoint,
        checkpoint.get("step"),
        args.num_clips,
        args.teacher,
        observed_groups,
        observed_frames,
        json.dumps(dataset_summary, sort_keys=True),
    )

    manifest = []
    seen_clips = set()
    clip_index = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        features = encode_clip(teacher=args.teacher, encoder=encoder, images=images, args=args)
        features = lerobot_train.normalize_feature_clip(features, args.feature_normalization, feature_stats)
        with torch.no_grad():
            pred = model(features, observed_groups)["pred"].detach()
        vis_features = lerobot_train.denormalize_feature_clip(features, args.feature_normalization, feature_stats)
        vis_pred = lerobot_train.denormalize_feature_clip(pred, args.feature_normalization, feature_stats)

        batch_size = images.shape[0]
        for item_idx in range(batch_size):
            clip_key = (batch["episode_uid"][item_idx], int(batch["start_index"][item_idx]))
            if not args.allow_duplicate_clips and clip_key in seen_clips:
                continue
            seen_clips.add(clip_key)
            metrics = sample_metrics(features[item_idx, 0], pred[item_idx, 0], observed_frames)
            filename = f"clip_{clip_index:04d}_ep{batch['episode_id'][item_idx]}_s{int(batch['start_index'][item_idx]):06d}.png"
            pv_utils.save_visualization(
                output_dir / filename,
                images[item_idx, 0],
                vis_features[item_idx, 0],
                vis_pred[item_idx, 0],
                svg_model,
                grid_size=args.svg_decode_grid,
                observed_frames=observed_frames,
                max_frames=args.vis_max_frames,
            )
            manifest.append(
                {
                    "file": filename,
                    "source_name": batch["source_name"][item_idx],
                    "episode_uid": batch["episode_uid"][item_idx],
                    "episode_id": batch["episode_id"][item_idx],
                    "start_index": int(batch["start_index"][item_idx]),
                    "frame_offsets": batch["frame_offsets"][item_idx],
                    "time_offsets_sec": batch["time_offsets_sec"][item_idx],
                    "rgb_delta": float(batch["rgb_delta"][item_idx]),
                    "rgb_mean": float(batch["rgb_mean"][item_idx]),
                    "rgb_std": float(batch["rgb_std"][item_idx]),
                    **metrics,
                }
            )
            logging.info(
                "clip=%d episode=%s start=%d future_copy_ratio=%.3f d_ratio=%.3f pred_d=%.3f gt_d=%.3f",
                clip_index,
                batch["episode_id"][item_idx],
                int(batch["start_index"][item_idx]),
                metrics["future_copy_ratio"],
                metrics["d_ratio"],
                metrics["pred_d"],
                metrics["gt_d"],
            )
            clip_index += 1
            if clip_index >= args.num_clips:
                break
        if clip_index >= args.num_clips:
            break

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summarize_manifest(manifest, observed_frames), indent=2),
        encoding="utf-8",
    )
    logging.info("Wrote %d visualization clips to %s", len(manifest), output_dir)


if __name__ == "__main__":
    main()
