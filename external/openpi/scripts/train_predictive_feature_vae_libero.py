#!/usr/bin/env python
"""Train a PV-VAE-style tokenizer over frozen LIBERO visual features.

The script reuses OpenPI's LIBERO LeRobot loader and SVG/DINO feature
teachers. It samples a dense clip, encodes it into patch-token features, and
trains a predictive feature VAE to reconstruct the full clip from an observed
prefix of temporal groups.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel

from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder
from openpi.models_pytorch.predictive_feature_vae import PredictiveFeatureVAE, PredictiveFeatureVAEConfig

import train_lam_libero as lam_utils


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-config", default="pi05_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", choices=("svg_p", "dinov3_vits16"), default="svg_p")
    parser.add_argument("--dinov3-path", default="assets/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--views", default="base_0_rgb")
    parser.add_argument(
        "--future-deltas",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
        help="Comma-separated LeRobot future frame indices. Current frame is prepended as its own group.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--temporal-compression", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--future-loss-weight", type=float, default=1.0)
    parser.add_argument("--observed-groups", type=int, default=0, help="0 means randomly sample a prefix length.")
    parser.add_argument("--min-observed-groups", type=int, default=1)
    parser.add_argument("--feature-normalization", choices=("none", "l2", "token_layer_norm"), default="none")
    parser.add_argument("--encoder-microbatch", type=int, default=16)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--vis-interval", type=int, default=1000)
    parser.add_argument("--vis-max-frames", type=int, default=12)
    parser.add_argument("--decode-svg-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svg-autoencoder-root", default=None)
    parser.add_argument("--svg-config", default=None)
    parser.add_argument("--svg-checkpoint", default=None)
    parser.add_argument("--svg-dinov3-weights", default=None)
    parser.add_argument("--svg-feature-dim", type=int, default=384)
    parser.add_argument("--svg-decode-grid", type=int, default=16)
    parser.add_argument("--teacher-image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-norm-stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_loader(args: argparse.Namespace, future_deltas: tuple[int, ...]):
    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    base_config = _config.get_config(args.openpi_config)
    data_factory = dataclasses.replace(base_config.data, future_image_delta_indices=future_deltas)
    train_config = dataclasses.replace(
        base_config,
        data=data_factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return _data.create_data_loader(
        train_config,
        framework="pytorch",
        shuffle=True,
        skip_norm_stats=args.skip_norm_stats,
    )


def prepare_image_clip(observation, future_images: dict, views: list[str], device: torch.device) -> torch.Tensor:
    per_view = []
    for view in views:
        if view not in observation.images:
            raise KeyError(f"Missing current image view: {view}")
        if view not in future_images:
            raise KeyError(f"Missing future image view: {view}")

        current = lam_utils.image_batch_to_chw_float(observation.images[view].to(device, non_blocking=True))
        future = future_images[view].to(device, non_blocking=True)
        if future.ndim != 5:
            raise ValueError(f"Future image view {view} must be [B,T,H,W,C] or [B,T,C,H,W], got {tuple(future.shape)}.")
        batch_size, num_future = future.shape[:2]
        future_flat = future.reshape(batch_size * num_future, *future.shape[2:])
        future_flat = lam_utils.image_batch_to_chw_float(future_flat)
        future = future_flat.view(batch_size, num_future, *future_flat.shape[1:])
        view_clip = torch.cat([current[:, None], future], dim=1)
        per_view.append(view_clip)
    return torch.stack(per_view, dim=1)


@torch.no_grad()
def encode_dino_clip(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    microbatch: int,
    precision: str,
) -> torch.Tensor:
    batch_size, num_views, num_frames, channels, height, width = images.shape
    flat = images.reshape(batch_size * num_views * num_frames, channels, height, width)
    outputs = []
    use_amp = precision != "float32" and flat.is_cuda
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    for start in range(0, flat.shape[0], microbatch):
        chunk = flat[start : start + microbatch]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs.append(encoder(chunk).float())
    features = torch.cat(outputs, dim=0)
    return features.view(batch_size, num_views, num_frames, features.shape[-2], features.shape[-1])


@torch.no_grad()
def encode_svg_p_clip(
    svg_model: torch.nn.Module,
    images: torch.Tensor,
    *,
    microbatch: int,
    precision: str,
    image_size: int,
) -> torch.Tensor:
    batch_size, num_views, num_frames, channels, height, width = images.shape
    flat = images.reshape(batch_size * num_views * num_frames, channels, height, width)
    if flat.shape[-2:] != (image_size, image_size):
        flat = F.interpolate(flat, size=(image_size, image_size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
    flat = (flat - mean) / std

    outputs = []
    use_amp = precision != "float32" and flat.is_cuda
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    for start in range(0, flat.shape[0], microbatch):
        chunk = flat[start : start + microbatch]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            z = svg_model.encode(chunk).float()
        outputs.append(z.flatten(2).transpose(1, 2).contiguous())
    features = torch.cat(outputs, dim=0)
    return features.view(batch_size, num_views, num_frames, features.shape[-2], features.shape[-1])


def choose_observed_groups(args: argparse.Namespace, total_groups: int) -> int:
    if args.observed_groups > 0:
        return min(args.observed_groups, total_groups)
    min_groups = min(max(args.min_observed_groups, 1), total_groups)
    return random.randint(min_groups, total_groups)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    model_config: PredictiveFeatureVAEConfig,
) -> None:
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "model_config": dataclasses.asdict(model_config),
        },
        path,
    )


def selected_frame_indices(num_frames: int, max_frames: int) -> list[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))

    candidates = [0, num_frames // 2 - 1, num_frames // 2, num_frames - 1]
    return sorted({idx for idx in candidates if 0 <= idx < num_frames})


def pca_feature_clip_images(
    target_features: torch.Tensor,
    pred_features: torch.Tensor,
    frame_ids: list[int],
    *,
    size: int = 224,
) -> tuple[list[Image.Image], list[Image.Image]]:
    features = torch.stack([target_features[frame_ids], pred_features[frame_ids]], dim=0)
    features = features.detach().float().cpu()
    num_sets, num_frames, num_tokens, dim = features.shape
    flat = features.reshape(num_sets * num_frames * num_tokens, dim)
    centered = flat - flat.mean(dim=0, keepdim=True)
    try:
        _, _, components = torch.pca_lowrank(centered, q=3, center=False)
        projected = centered @ components[:, :3]
    except RuntimeError:
        projected = centered[:, :3]
    lo = projected.quantile(0.01, dim=0, keepdim=True)
    hi = projected.quantile(0.99, dim=0, keepdim=True)
    projected = ((projected - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

    grid = int(num_tokens**0.5)
    if grid * grid != num_tokens:
        raise ValueError(f"Expected square token grid, got {num_tokens} tokens.")
    projected = projected.reshape(num_sets, num_frames, grid, grid, 3).numpy()
    images: list[list[Image.Image]] = []
    for set_idx in range(num_sets):
        images.append(
            [
                Image.fromarray((projected[set_idx, frame_idx] * 255.0).astype(np.uint8)).resize(
                    (size, size), Image.Resampling.BILINEAR
                )
                for frame_idx in range(num_frames)
            ]
        )
    return images[0], images[1]


@torch.no_grad()
def save_visualization(
    path: Path,
    image_clip: torch.Tensor,
    target_features: torch.Tensor,
    pred_features: torch.Tensor,
    svg_model: torch.nn.Module | None,
    *,
    grid_size: int,
    observed_frames: int,
    max_frames: int,
) -> None:
    frame_ids = selected_frame_indices(target_features.shape[0], max_frames)
    rows: list[tuple[str, list[Image.Image]]] = []

    rows.append(("target_rgb", [lam_utils.tensor_image_to_pil(image_clip[idx]) for idx in frame_ids]))

    if svg_model is not None:
        rows.append(
            (
                "pred_feat_svg_rgb",
                [
                    lam_utils.decode_svg_feature_image(svg_model, pred_features[idx], grid_size=grid_size)
                    for idx in frame_ids
                ],
            )
        )

    gt_pca, pred_pca = pca_feature_clip_images(target_features, pred_features, frame_ids)
    rows.append(("gt_feat", gt_pca))
    rows.append(("pred_feat", pred_pca))

    cell = 224
    label_h = 28
    width = cell * len(frame_ids)
    height = (cell + label_h) * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, (row_name, panels) in enumerate(rows):
        y = row_idx * (cell + label_h)
        for col_idx, panel in enumerate(panels):
            x = col_idx * cell
            canvas.paste(panel, (x, y))
            marker = "obs" if frame_ids[col_idx] < observed_frames else "pred"
            draw.text((x + 8, y + cell + 6), f"{row_name} f{frame_ids[col_idx]} {marker}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    use_ddp, local_rank, device = lam_utils.setup_ddp()
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)

    future_deltas = parse_int_list(args.future_deltas)
    num_future_frames = len(future_deltas)
    num_frames = 1 + num_future_frames
    if num_future_frames % args.temporal_compression != 0:
        raise ValueError(
            f"future frames = {num_future_frames}, which must be divisible by "
            f"--temporal-compression={args.temporal_compression}."
        )
    total_groups = 1 + num_future_frames // args.temporal_compression

    output_dir = Path(args.output_dir)
    if lam_utils.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    data_loader = build_loader(args, future_deltas)
    views = [view.strip() for view in args.views.split(",") if view.strip()]

    encoder = None
    svg_model = None
    if args.teacher == "svg_p":
        if args.feature_normalization != "none":
            raise ValueError("SVG-P teacher requires --feature-normalization none for matched decode.")
        svg_model = lam_utils.load_svg_decoder(args, device)
        feature_dim = args.svg_feature_dim
    else:
        encoder = load_dinov3_patch_encoder(args.dinov3_path).to(device).eval()
        feature_dim = int(encoder.config.hidden_size)
        if args.decode_svg_rgb and lam_utils.is_rank0():
            svg_model = lam_utils.load_svg_decoder(args, device)

    model_config = PredictiveFeatureVAEConfig(
        feature_dim=feature_dim,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        temporal_compression=args.temporal_compression,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.heads,
        max_views=max(4, len(views)),
        max_frames=max(32, num_frames),
        kl_weight=args.kl_weight,
        cosine_weight=args.cosine_weight,
        delta_weight=args.delta_weight,
        future_loss_weight=args.future_loss_weight,
    )
    model = PredictiveFeatureVAE(model_config).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if lam_utils.is_rank0():
        param_count = sum(p.numel() for p in (model.module if use_ddp else model).parameters())
        logging.info("PredictiveFeatureVAE config: %s", model_config)
        logging.info(
            "params=%.1fM teacher=%s views=%s future_deltas=%s latent_groups=1+%d/%d=%d output_dir=%s",
            param_count / 1e6,
            args.teacher,
            views,
            future_deltas,
            num_future_frames,
            args.temporal_compression,
            total_groups,
            output_dir,
        )

    iterator = iter(data_loader)
    start_time = time.time()
    last_log_time = start_time
    for step in range(1, args.max_steps + 1):
        batch = next(iterator)
        observation, future_images = lam_utils.unpack_batch(batch)
        image_clip = prepare_image_clip(observation, future_images, views, device)

        if args.teacher == "svg_p":
            features = encode_svg_p_clip(
                svg_model,
                image_clip,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
        else:
            features = encode_dino_clip(
                encoder,
                image_clip,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )
        features = lam_utils.normalize_features(features, args.feature_normalization)

        observed_groups = choose_observed_groups(args, total_groups)
        optimizer.zero_grad(set_to_none=True)
        use_amp = args.precision != "float32" and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss_dict = model.module.compute_loss(features, observed_groups) if use_ddp else model.compute_loss(features, observed_groups)
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if (step % args.log_interval == 0 or step == args.max_steps) and lam_utils.is_rank0():
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6)
            last_log_time = now
            logging.info(
                (
                    "step=%d loss=%.6f recon=%.6f cos=%.6f delta=%.6f kl=%.6f "
                    "obs_mse=%.6f future_mse=%.6f pred_d=%.6f gt_d=%.6f d_ratio=%.3f "
                    "static_future_mse=%.6f obs_groups=%d grad=%.3f steps/s=%.3f"
                ),
                step,
                float(loss_dict["loss"].detach().cpu()),
                float(loss_dict["recon_loss"].detach().cpu()),
                float(loss_dict["cosine_loss"].detach().cpu()),
                float(loss_dict["delta_loss"].detach().cpu()),
                float(loss_dict["kl_loss"].detach().cpu()),
                float(loss_dict["observed_mse"].detach().cpu()),
                float(loss_dict["future_mse"].detach().cpu()),
                float(loss_dict["pred_delta_norm"].detach().cpu()),
                float(loss_dict["target_delta_norm"].detach().cpu()),
                float(loss_dict["delta_ratio"].detach().cpu()),
                float(loss_dict["static_future_mse"].detach().cpu()),
                observed_groups,
                float(grad_norm.detach().cpu()),
                steps_per_sec,
            )

        if args.vis_interval > 0 and step % args.vis_interval == 0 and lam_utils.is_rank0():
            pred = loss_dict["pred"].detach()
            save_visualization(
                output_dir / "vis" / f"step_{step:06d}.png",
                image_clip[0, 0],
                features[0, 0],
                pred[0, 0],
                svg_model if args.decode_svg_rgb else None,
                grid_size=args.svg_decode_grid,
                observed_frames=1 + (observed_groups - 1) * args.temporal_compression,
                max_frames=args.vis_max_frames,
            )

        if args.save_interval > 0 and step % args.save_interval == 0 and lam_utils.is_rank0():
            save_checkpoint(output_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, step, args, model_config)

    if lam_utils.is_rank0():
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, args.max_steps, args, model_config)
        logging.info("Finished in %.1f minutes", (time.time() - start_time) / 60.0)
    lam_utils.cleanup_ddp()


if __name__ == "__main__":
    main()
