#!/usr/bin/env python
"""Export PV-VAE compressed semantic latents for downstream WAM checks.

This script is intentionally small and diagnostic. It reuses the same LeRobot
clip loader, frozen SVG-P/DINO teacher, and feature normalization path as the
PV-VAE training/visualization scripts, then exports the PV-VAE encoder output
as a shape-stable target for a WAM-style video-action model.

Expected contract:

    RGB clip                  [B,V,17,C,H,W]
    teacher feature clip      [B,V,17,N,D]
    PV-VAE encoder mu         [B,V,5,N,d]
    WAM target layout         [B,d,5,H_feat,W_feat_total]

The last layout concatenates camera views along feature width. It is only a
contract check/export format; it does not train FastWAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="PV-VAE checkpoint .pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for latents.pt and summary.json.")
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument("--dataset-spec-json", default=None)
    parser.add_argument("--teacher", choices=("svg_p", "dinov3_vits16"), default=None)
    parser.add_argument("--dinov3-path", default=None)
    parser.add_argument("--video-keys", default=None)
    parser.add_argument(
        "--wam-view-layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help=(
            "How to pack per-view semantic latent grids for WAM. FastWAM's LIBERO/OXE-style "
            "multi-camera RGB path uses horizontal camera concatenation."
        ),
    )
    parser.add_argument(
        "--require-num-views",
        type=int,
        default=0,
        help="If positive, fail unless the exported clips have exactly this many camera views.",
    )
    parser.add_argument("--future-deltas", default=None)
    parser.add_argument("--time-sampling-mode", choices=("frame_deltas", "duration_sec"), default=None)
    parser.add_argument("--clip-duration-sec", type=float, default=None)
    parser.add_argument("--num-future-frames", type=int, default=None)
    parser.add_argument(
        "--observed-groups",
        type=int,
        default=0,
        help="Groups to encode. 0 means encode the full clip into all latent groups.",
    )
    parser.add_argument("--num-clips", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
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
    parser.add_argument("--teacher-image-size", type=int, default=None)
    parser.add_argument("--svg-autoencoder-root", default=None)
    parser.add_argument("--svg-config", default=None)
    parser.add_argument("--svg-checkpoint", default=None)
    parser.add_argument("--svg-dinov3-weights", default=None)
    parser.add_argument("--svg-decode-grid", type=int, default=None)
    parser.add_argument("--save-features", action="store_true", help="Also save normalized teacher features.")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16", help="Storage dtype for exported tensors.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_square_grid(num_tokens: int) -> int:
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square feature grid from N={num_tokens}.")
    return side


def latents_to_wam_target(latents: torch.Tensor, *, view_layout: str) -> torch.Tensor:
    """Convert [B,V,G,N,d] latents into a FastWAM-style packed spatial layout.

    The original FastWAM RGB dataloader packs camera images spatially before
    feeding Wan-VAE. This function mirrors that contract in semantic-latent
    space: each camera owns one H_feat x W_feat grid, and views are concatenated
    along width or height.
    """
    if latents.ndim != 5:
        raise ValueError(f"Expected latents [B,V,G,N,d], got {tuple(latents.shape)}.")
    batch_size, num_views, num_groups, num_tokens, latent_dim = latents.shape
    side = infer_square_grid(num_tokens)
    grid = latents.contiguous().view(batch_size, num_views, num_groups, side, side, latent_dim)
    if view_layout == "horizontal":
        grid = grid.permute(0, 5, 2, 3, 1, 4).contiguous()
        return grid.view(batch_size, latent_dim, num_groups, side, num_views * side)
    if view_layout == "vertical":
        grid = grid.permute(0, 5, 2, 1, 3, 4).contiguous()
        return grid.view(batch_size, latent_dim, num_groups, num_views * side, side)
    raise ValueError(f"Unsupported view_layout={view_layout!r}.")


def validate_wam_target_shape(latents: torch.Tensor, wam_target: torch.Tensor, *, view_layout: str) -> dict[str, int | str]:
    """Return layout metadata and fail loudly on shape drift."""
    if latents.ndim != 5 or wam_target.ndim != 5:
        raise ValueError(f"Expected latents/wam_target to be 5D, got {tuple(latents.shape)} and {tuple(wam_target.shape)}.")
    batch_size, num_views, num_groups, num_tokens, latent_dim = latents.shape
    side = infer_square_grid(num_tokens)
    expected_hw = (side, num_views * side) if view_layout == "horizontal" else (num_views * side, side)
    expected = (batch_size, latent_dim, num_groups, expected_hw[0], expected_hw[1])
    if tuple(wam_target.shape) != expected:
        raise ValueError(
            "Packed WAM target shape mismatch: "
            f"expected {expected} from latents {tuple(latents.shape)} with layout={view_layout}, "
            f"got {tuple(wam_target.shape)}."
        )
    return {
        "view_layout": view_layout,
        "num_views": num_views,
        "groups": num_groups,
        "latent_dim": latent_dim,
        "tokens_per_view": num_tokens,
        "feature_grid_h": side,
        "feature_grid_w": side,
        "packed_h": expected_hw[0],
        "packed_w": expected_hw[1],
    }


def tensor_for_storage(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if dtype == "float16":
        return tensor.to(torch.float16)
    return tensor.to(torch.float32)


def main() -> None:
    cli_args = parse_args()
    user_forced_lerobot_root = cli_args.lerobot_root is not None and cli_args.dataset_spec_json is None
    args = cli_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import train_lam_libero as lam_utils
    import train_predictive_feature_vae_lerobot as lerobot_train
    import visualize_predictive_feature_vae_lerobot as vis_pvvae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, checkpoint_args, checkpoint = vis_pvvae.load_checkpoint(Path(args.checkpoint), device)
    args = vis_pvvae.prepare_model_args(args, checkpoint_args)
    if user_forced_lerobot_root:
        # User explicitly requests a local dataset root; don't inherit mixed-source spec from checkpoint args.
        args.dataset_spec_json = None
    if args.lerobot_root is None:
        args.lerobot_root = checkpoint_args.get("lerobot_root", checkpoint_args.get("lerobot-root"))

    future_deltas = vis_pvvae.parse_int_list(args.future_deltas)
    dataset, dataset_summary, num_future_frames, _, video_keys = lerobot_train.build_clip_dataset(
        args,
        fallback_future_deltas=future_deltas,
    )
    total_groups = 1 + num_future_frames // model.config.temporal_compression
    observed_groups = total_groups if int(args.observed_groups) <= 0 else min(int(args.observed_groups), total_groups)
    observed_frames = 1 + (observed_groups - 1) * model.config.temporal_compression
    if args.require_num_views > 0 and len(video_keys) != args.require_num_views:
        raise ValueError(f"--require-num-views={args.require_num_views} but dataset video_keys={video_keys}.")

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": lerobot_train.collate_clip_batch,
    }
    if hasattr(lerobot_train, "seed_worker"):
        loader_kwargs["worker_init_fn"] = lerobot_train.seed_worker
    loader = DataLoader(**loader_kwargs)

    if args.teacher == "svg_p":
        svg_args = argparse.Namespace(**vars(args))
        svg_args.feature_normalization = "none"
        encoder = lam_utils.load_svg_decoder(svg_args, device)
    else:
        from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder

        encoder = load_dinov3_patch_encoder(args.dinov3_path, device)

    feature_stats = lerobot_train.load_feature_stats(args.feature_stats, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_batches: list[torch.Tensor] = []
    wam_batches: list[torch.Tensor] = []
    feature_batches: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    collected = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.no_grad():
            features = vis_pvvae.encode_clip(teacher=args.teacher, encoder=encoder, images=images, args=args)
            features = lerobot_train.normalize_feature_clip(features, args.feature_normalization, feature_stats)
            mu, logvar = model.encode_observed(features, observed_groups)
            latents = mu
            if args.require_num_views > 0 and latents.shape[1] != args.require_num_views:
                raise ValueError(f"Expected {args.require_num_views} views, got latents shape {tuple(latents.shape)}.")
            wam_target = latents_to_wam_target(latents, view_layout=args.wam_view_layout)
            layout_metadata = validate_wam_target_shape(latents, wam_target, view_layout=args.wam_view_layout)

        take = min(images.shape[0], args.num_clips - collected)
        latent_batches.append(tensor_for_storage(latents[:take], args.dtype))
        wam_batches.append(tensor_for_storage(wam_target[:take], args.dtype))
        if args.save_features:
            feature_batches.append(tensor_for_storage(features[:take], args.dtype))

        for item_idx in range(take):
            metadata.append(
                {
                    "source_name": batch["source_name"][item_idx],
                    "episode_uid": batch["episode_uid"][item_idx],
                    "episode_id": batch["episode_id"][item_idx],
                    "start_index": int(batch["start_index"][item_idx]),
                    "frame_offsets": batch["frame_offsets"][item_idx],
                    "time_offsets_sec": batch["time_offsets_sec"][item_idx],
                    "fps": float(batch["fps"][item_idx]),
                    "rgb_delta": float(batch["rgb_delta"][item_idx]),
                    "rgb_mean": float(batch["rgb_mean"][item_idx]),
                    "rgb_std": float(batch["rgb_std"][item_idx]),
                }
            )
        collected += take
        if collected >= args.num_clips:
            break

    if not latent_batches:
        raise RuntimeError("No clips were exported. Check dataset path and filters.")

    latents = torch.cat(latent_batches, dim=0)
    wam_target = torch.cat(wam_batches, dim=0)
    final_layout_metadata = validate_wam_target_shape(latents, wam_target, view_layout=args.wam_view_layout)
    payload: dict[str, Any] = {
        "latents": latents,
        "wam_target": wam_target,
        "metadata": metadata,
        "model_config": checkpoint["model_config"],
        "checkpoint_step": checkpoint.get("step"),
        "dataset_summary": dataset_summary,
        "video_keys": video_keys,
        "view_layout": args.wam_view_layout,
        "feature_normalization": args.feature_normalization,
        "observed_groups": observed_groups,
        "observed_frames": observed_frames,
        "total_groups": total_groups,
        "layout_metadata": final_layout_metadata,
        "contract": {
            "latents": "[B,V,G,N,d]",
            "wam_target": "[B,d,G,H_packed,W_packed]",
            "view_layout": args.wam_view_layout,
            "packing_rule": (
                "horizontal: camera grids are concatenated along W_packed; "
                "vertical: camera grids are concatenated along H_packed"
            ),
        },
    }
    if args.save_features:
        payload["features"] = torch.cat(feature_batches, dim=0)

    torch.save(payload, output_dir / "latents.pt")
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "num_clips": int(latents.shape[0]),
        "teacher": args.teacher,
        "feature_normalization": args.feature_normalization,
        "latent_shape": list(latents.shape),
        "wam_target_shape": list(wam_target.shape),
        "layout_metadata": final_layout_metadata,
        "view_layout": args.wam_view_layout,
        "video_keys": video_keys,
        "latent_norm_mean": float(latents.float().norm(dim=-1).mean()),
        "latent_norm_std": float(latents.float().norm(dim=-1).std()),
        "wam_target_mean": float(wam_target.float().mean()),
        "wam_target_std": float(wam_target.float().std()),
        "observed_groups": observed_groups,
        "observed_frames": observed_frames,
        "total_groups": total_groups,
        "num_future_frames": num_future_frames,
        "dataset_summary": dataset_summary,
        "first_metadata": metadata[: min(4, len(metadata))],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote %s and %s", output_dir / "latents.pt", output_dir / "summary.json")
    logging.info("latent_shape=%s wam_target_shape=%s", tuple(latents.shape), tuple(wam_target.shape))


if __name__ == "__main__":
    main()
