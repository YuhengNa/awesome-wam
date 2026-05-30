#!/usr/bin/env python
"""Train a DreamDojo-style latent-action model on LIBERO feature pairs.

The script uses OpenPI's LeRobot LIBERO loader to sample current images plus
one future image at a fixed stride, encodes both with a frozen feature teacher,
and trains a small VAE bottleneck to reconstruct future features.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel

from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder
from openpi.models_pytorch.latent_action import FeatureLatentActionConfig, FeatureLatentActionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-config", default="pi05_libero", help="Base OpenPI data config to reuse.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and visualizations.")
    parser.add_argument(
        "--teacher",
        choices=("dinov3_vits16", "svg_p"),
        default="dinov3_vits16",
        help="Feature teacher. svg_p uses SVG-P's paired DINOv3-S+ encoder/decoder.",
    )
    parser.add_argument("--dinov3-path", default="assets/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--views", default="base_0_rgb")
    parser.add_argument("--lam-stride", type=int, default=4, help="Future frame delta in LeRobot steps.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument(
        "--feature-normalization",
        choices=("token_layer_norm", "l2", "none"),
        default="none",
    )
    parser.add_argument("--encoder-microbatch", type=int, default=64)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--vis-interval", type=int, default=1000)
    parser.add_argument("--decode-svg-rgb", action="store_true", help="Decode predicted DINO features with SVG-P decoder.")
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


def setup_ddp() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_loader(args: argparse.Namespace):
    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    base_config = _config.get_config(args.openpi_config)
    if not hasattr(base_config.data, "future_image_delta_indices"):
        raise ValueError(f"{args.openpi_config} is not a LIBERO config with future image support.")
    data_config = dataclasses.replace(base_config.data, future_image_delta_indices=(args.lam_stride,))
    train_config = dataclasses.replace(
        base_config,
        data=data_config,
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


def unpack_batch(batch):
    if isinstance(batch, tuple | list) and len(batch) == 3:
        observation, _, future_images = batch
        return observation, future_images
    raise ValueError("LAM training requires future images; set future_image_delta_indices in the data config.")


def image_batch_to_chw_float(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.ndim != 4:
        raise ValueError(f"Expected image batch [B,H,W,C] or [B,C,H,W], got {tuple(images.shape)}.")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    elif images.shape[1] != 3:
        raise ValueError(f"Expected RGB image batch, got {tuple(images.shape)}.")
    if images.max() > 2.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = images * 0.5 + 0.5
    return images.clamp(0.0, 1.0)


def prepare_image_pair(observation, future_images: dict, views: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    current_per_view = []
    future_per_view = []
    for view in views:
        if view not in observation.images:
            raise KeyError(f"Missing current image view: {view}")
        if view not in future_images:
            raise KeyError(f"Missing future image view: {view}")

        current = observation.images[view].to(device, non_blocking=True)
        future = future_images[view].to(device, non_blocking=True)
        if future.ndim != 5:
            raise ValueError(f"Future image view {view} must be [B,T,H,W,C] or [B,T,C,H,W], got {tuple(future.shape)}.")
        future = future[:, 0]

        current_per_view.append(image_batch_to_chw_float(current))
        future_per_view.append(image_batch_to_chw_float(future))

    current_images = torch.stack(current_per_view, dim=1)
    future_images_tensor = torch.stack(future_per_view, dim=1)
    return current_images, future_images_tensor


@torch.no_grad()
def encode_dino_features(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    microbatch: int,
    precision: str,
) -> torch.Tensor:
    batch_size, num_views, channels, height, width = images.shape
    flat = images.reshape(batch_size * num_views, channels, height, width)
    outputs = []
    use_amp = precision != "float32" and flat.is_cuda
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    for start in range(0, flat.shape[0], microbatch):
        chunk = flat[start : start + microbatch]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs.append(encoder(chunk).float())
    features = torch.cat(outputs, dim=0)
    return features.view(batch_size, num_views, features.shape[-2], features.shape[-1])


@torch.no_grad()
def encode_svg_p_features(
    svg_model: torch.nn.Module,
    images: torch.Tensor,
    *,
    microbatch: int,
    precision: str,
    image_size: int,
) -> torch.Tensor:
    batch_size, num_views, channels, height, width = images.shape
    flat = images.reshape(batch_size * num_views, channels, height, width)
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
    return features.view(batch_size, num_views, features.shape[-2], features.shape[-1])


def normalize_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return features
    if mode == "l2":
        return F.normalize(features, dim=-1)
    if mode == "token_layer_norm":
        return F.layer_norm(features, (features.shape[-1],))
    raise ValueError(f"Unknown feature normalization: {mode}")


def pca_feature_images(*feature_sets: torch.Tensor, size: int = 224) -> list[Image.Image]:
    tokens = [features.detach().float().cpu() for features in feature_sets]
    combined = torch.cat(tokens, dim=0)
    centered = combined - combined.mean(dim=0, keepdim=True)
    try:
        _, _, components = torch.pca_lowrank(centered, q=3, center=False)
        projected = centered @ components[:, :3]
    except RuntimeError:
        projected = centered[:, :3]
    lo = projected.quantile(0.01, dim=0, keepdim=True)
    hi = projected.quantile(0.99, dim=0, keepdim=True)
    projected = ((projected - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

    images = []
    offset = 0
    for features in tokens:
        count = features.shape[0]
        grid = int(count**0.5)
        if grid * grid != count:
            raise ValueError(f"Expected square token grid, got {count} tokens.")
        image = projected[offset : offset + count].reshape(grid, grid, 3).numpy()
        images.append(Image.fromarray((image * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR))
        offset += count
    return images


def tensor_image_to_pil(image: torch.Tensor, size: int = 224) -> Image.Image:
    image = image.detach().float().cpu()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image = image.clamp(0.0, 1.0)
    pil = Image.fromarray((image.numpy() * 255.0).astype(np.uint8))
    return pil.resize((size, size), Image.Resampling.BILINEAR)


def install_svg_inference_stubs() -> None:
    try:
        import pytorch_lightning  # noqa: F401
    except ModuleNotFoundError:
        pl = types.ModuleType("pytorch_lightning")

        class LightningModule(torch.nn.Module):
            pass

        pl.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = pl

    if "ldm.hy3.autoencoder_kl_3d" not in sys.modules:
        hy3_autoencoder = types.ModuleType("ldm.hy3.autoencoder_kl_3d")

        class UnusedHYDecoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("HYDecoder is not required for SVG-P feature decoding.")

        hy3_autoencoder.Decoder = UnusedHYDecoder
        sys.modules["ldm.hy3.autoencoder_kl_3d"] = hy3_autoencoder

    if "ldm.models.swin_v2" not in sys.modules:
        swin_v2 = types.ModuleType("ldm.models.swin_v2")

        class UnusedSwinV2Encoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("SwinV2Encoder is not required for SVG-P feature decoding.")

        swin_v2.SwinV2Encoder = UnusedSwinV2Encoder
        sys.modules["ldm.models.swin_v2"] = swin_v2

    if "ldm.rope_vit.vit_rope" not in sys.modules:
        vit_rope = types.ModuleType("ldm.rope_vit.vit_rope")

        def _unused_rope_vit(*args, **kwargs):
            raise RuntimeError("RoPE ViT is not required for SVG-P feature decoding.")

        vit_rope.rope_mixed_deit_small_patch16_LS = _unused_rope_vit
        sys.modules["ldm.rope_vit.vit_rope"] = vit_rope

    if "xformers" not in sys.modules:
        xformers = types.ModuleType("xformers")
        xops = types.ModuleType("xformers.ops")

        def sparsify24(weight, *args, **kwargs):
            return weight

        xops.sparsify24 = sparsify24
        xformers.ops = xops
        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = xops


def load_svg_decoder(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.feature_normalization != "none":
        raise ValueError("--decode-svg-rgb requires --feature-normalization none so decoder inputs stay in feature space.")
    required = {
        "--svg-autoencoder-root": args.svg_autoencoder_root,
        "--svg-config": args.svg_config,
        "--svg-checkpoint": args.svg_checkpoint,
        "--svg-dinov3-weights": args.svg_dinov3_weights,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"--decode-svg-rgb is missing required args: {', '.join(missing)}")

    import yaml

    root = Path(args.svg_autoencoder_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    install_svg_inference_stubs()

    from ldm.models.dinov3_decoder_native_resolution import DinoDecoder

    with open(args.svg_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["model"]["params"])
    params["ckpt_path"] = str(Path(args.svg_checkpoint).expanduser().resolve())
    params["is_train"] = False
    params["dinoconfig"] = dict(params["dinoconfig"])
    params["dinoconfig"]["dinov3_location"] = str(root / "dinov3")
    params["dinoconfig"]["weights"] = str(Path(args.svg_dinov3_weights).expanduser().resolve())

    decoder = DinoDecoder(**params).to(device).eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder


@torch.no_grad()
def decode_svg_feature_image(
    decoder: torch.nn.Module,
    features: torch.Tensor,
    *,
    grid_size: int,
    size: int = 224,
) -> Image.Image:
    tokens = features.detach().float()
    token_grid = int(tokens.shape[0] ** 0.5)
    if token_grid * token_grid != tokens.shape[0]:
        raise ValueError(f"Expected square token grid, got {tokens.shape[0]} tokens.")
    z = tokens.T.reshape(1, tokens.shape[-1], token_grid, token_grid)
    z = z.to(device=next(decoder.parameters()).device, dtype=next(decoder.parameters()).dtype)
    if token_grid != grid_size:
        z = F.interpolate(z.float(), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    decoded = decoder.decode(z.float()).float()[0]
    decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0).cpu()
    return tensor_image_to_pil(decoded, size=size)


def save_visualization(
    path: Path,
    current_image: torch.Tensor,
    future_image: torch.Tensor,
    current_features: torch.Tensor,
    future_features: torch.Tensor,
    pred_features: torch.Tensor,
    pred_decoded_image: Image.Image | None = None,
) -> None:
    current_pca, target_pca, pred_pca = pca_feature_images(current_features, future_features, pred_features)
    current_rgb = tensor_image_to_pil(current_image)
    future_rgb = tensor_image_to_pil(future_image)
    panels = [current_rgb, future_rgb, current_pca, target_pca, pred_pca]
    labels = ["rgb_t", "rgb_t+k", "feat_t", "feat_t+k", "pred_feat"]
    if pred_decoded_image is not None:
        panels.insert(2, pred_decoded_image)
        labels.insert(2, "pred_rgb_svg")
    width, height = 224 * len(panels), 252
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x = idx * 224
        canvas.paste(panel, (x, 0))
        draw.text((x + 8, 230), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    lam_config: FeatureLatentActionConfig,
) -> None:
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "lam_config": dataclasses.asdict(lam_config),
        },
        path,
    )


def compute_lam_loss(
    model: torch.nn.Module,
    current_features: torch.Tensor,
    future_features: torch.Tensor,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    outputs = model(current_features, future_features)
    recon_loss = F.mse_loss(outputs["pred"], future_features)
    kl_loss = -0.5 * (1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()).mean()
    loss = recon_loss + kl_weight * kl_loss
    return {
        **outputs,
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }


def batch_pearson_corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() < 2:
        return left.new_tensor(float("nan"))
    left = left.float().flatten()
    right = right.float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denom = left.norm() * right.norm()
    return (left * right).sum() / denom.clamp_min(1e-6)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    use_ddp, local_rank, device = setup_ddp()
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    data_loader = build_loader(args)
    views = [view.strip() for view in args.views.split(",") if view.strip()]

    encoder = None
    svg_model = None
    if args.teacher == "svg_p":
        if args.feature_normalization != "none":
            raise ValueError("SVG-P teacher requires --feature-normalization none for matched encode/decode.")
        svg_model = load_svg_decoder(args, device)
        feature_dim = args.svg_feature_dim
    else:
        encoder = load_dinov3_patch_encoder(args.dinov3_path).to(device)
        feature_dim = int(encoder.config.hidden_size)
        encoder.eval()
        svg_model = load_svg_decoder(args, device) if args.decode_svg_rgb and is_rank0() else None

    lam_config = FeatureLatentActionConfig(
        feature_dim=feature_dim,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.heads,
        max_views=max(4, len(views)),
        kl_weight=args.kl_weight,
    )
    model = FeatureLatentActionModel(lam_config).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if is_rank0():
        logging.info("LAM config: %s", lam_config)
        logging.info(
            "teacher=%s views=%s stride=%d feature_dim=%d teacher_image_size=%d output_dir=%s",
            args.teacher,
            views,
            args.lam_stride,
            feature_dim,
            args.teacher_image_size,
            output_dir,
        )

    iterator = iter(data_loader)
    start_time = time.time()
    last_log_time = start_time
    for step in range(1, args.max_steps + 1):
        batch = next(iterator)
        observation, future_images = unpack_batch(batch)
        current_images, future_images_tensor = prepare_image_pair(observation, future_images, views, device)

        if args.teacher == "svg_p":
            current_features = encode_svg_p_features(
                svg_model,
                current_images,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
            future_features = encode_svg_p_features(
                svg_model,
                future_images_tensor,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
        else:
            current_features = encode_dino_features(
                encoder,
                current_images,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )
            future_features = encode_dino_features(
                encoder,
                future_images_tensor,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )
        current_features = normalize_features(current_features, args.feature_normalization)
        future_features = normalize_features(future_features, args.feature_normalization)

        optimizer.zero_grad(set_to_none=True)
        use_amp = args.precision != "float32" and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss_dict = compute_lam_loss(model, current_features, future_features, args.kl_weight)
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if (step % args.log_interval == 0 or step == args.max_steps) and is_rank0():
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6)
            last_log_time = now
            with torch.no_grad():
                pred = loss_dict["pred"].detach()
                mse = F.mse_loss(pred.float(), future_features.float())
                cosine_metric = 1.0 - F.cosine_similarity(pred.float(), future_features.float(), dim=-1).mean()
                copy_mse = F.mse_loss(current_features.float(), future_features.float())
                copy_ratio = mse / copy_mse.clamp_min(1e-6)
                raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                zero_z = torch.zeros_like(loss_dict["z"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    zero_pred = raw_model.decode(current_features, zero_z)
                    if loss_dict["z"].shape[0] > 1:
                        shuffled_z = loss_dict["z"][torch.randperm(loss_dict["z"].shape[0], device=loss_dict["z"].device)]
                        shuffled_pred = raw_model.decode(current_features, shuffled_z)
                    else:
                        shuffled_pred = zero_pred
                zero_z_mse = F.mse_loss(zero_pred.float(), future_features.float())
                shuffle_z_mse = F.mse_loss(shuffled_pred.float(), future_features.float())
                target_delta = future_features.float() - current_features.float()
                pred_delta = pred.float() - current_features.float()
                delta_ratio = pred_delta.norm() / target_delta.norm().clamp_min(1e-6)
                z_norm = loss_dict["z"].float().norm(dim=-1).mean()
                mu_norm = loss_dict["mu"].float().norm(dim=-1).mean()
                target_delta_norm = target_delta.norm(dim=-1).mean()
                mu_norm_per_sample = loss_dict["mu"].float().norm(dim=-1)
                target_delta_norm_per_sample = target_delta.norm(dim=-1).mean(dim=(1, 2))
                z_delta_corr = batch_pearson_corr(mu_norm_per_sample, target_delta_norm_per_sample)
            logging.info(
                (
                    "step=%d loss=%.6f recon=%.6f cos=%.6f kl=%.6f copy_mse=%.6f "
                    "copy_ratio=%.3f zero_z_mse=%.6f shuffle_z_mse=%.6f "
                    "delta_ratio=%.3f z_norm=%.4f mu_norm=%.4f "
                    "target_delta_norm=%.4f z_delta_corr=%.3f grad=%.3f steps/s=%.3f"
                ),
                step,
                float(loss_dict["loss"].detach().cpu()),
                float(loss_dict["recon_loss"].detach().cpu()),
                float(cosine_metric.detach().cpu()),
                float(loss_dict["kl_loss"].detach().cpu()),
                float(copy_mse.detach().cpu()),
                float(copy_ratio.detach().cpu()),
                float(zero_z_mse.detach().cpu()),
                float(shuffle_z_mse.detach().cpu()),
                float(delta_ratio.detach().cpu()),
                float(z_norm.detach().cpu()),
                float(mu_norm.detach().cpu()),
                float(target_delta_norm.detach().cpu()),
                float(z_delta_corr.detach().cpu()),
                float(grad_norm.detach().cpu()),
                steps_per_sec,
            )

        if args.vis_interval > 0 and step % args.vis_interval == 0 and is_rank0():
            pred = loss_dict["pred"].detach()
            pred_decoded = (
                decode_svg_feature_image(
                    svg_model,
                    pred[0, 0],
                    grid_size=args.svg_decode_grid,
                )
                if svg_model is not None and args.decode_svg_rgb
                else None
            )
            save_visualization(
                output_dir / "vis" / f"step_{step:06d}.png",
                current_images[0, 0],
                future_images_tensor[0, 0],
                current_features[0, 0],
                future_features[0, 0],
                pred[0, 0],
                pred_decoded,
            )

        if args.save_interval > 0 and step % args.save_interval == 0 and is_rank0():
            save_checkpoint(output_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, step, args, lam_config)

    if is_rank0():
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, args.max_steps, args, lam_config)
        logging.info("Finished in %.1f minutes", (time.time() - start_time) / 60.0)
    cleanup_ddp()


if __name__ == "__main__":
    main()
