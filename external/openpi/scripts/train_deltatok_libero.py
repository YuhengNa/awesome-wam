#!/usr/bin/env python
"""Train a DeltaTok-style tokenizer on LIBERO feature pairs.

This script reuses the existing LAM LIBERO data and SVG/DINO feature-teacher
helpers, but trains a deterministic transition tokenizer:

    encode(x_t, x_t+k) -> z_delta [B, M, d]
    decode(x_t, z_delta) -> x_hat_t+k
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel

from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder
from openpi.models_pytorch.feature_delta_tokenizer import FeatureDeltaTokenizer, FeatureDeltaTokenizerConfig

import train_lam_libero as lam_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-config", default="pi05_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", choices=("dinov3_vits16", "svg_p"), default="svg_p")
    parser.add_argument("--dinov3-path", default="assets/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--views", default="base_0_rgb")
    parser.add_argument("--delta-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=0, help="0 keeps model dim equal to feature dim.")
    parser.add_argument("--token-dim", type=int, default=0, help="0 keeps delta token dim equal to model dim.")
    parser.add_argument("--num-delta-tokens", type=int, default=1, help="M in z_delta [B,M,d].")
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--loss-fn", choices=("mse", "log_cosh", "smooth_l1"), default="log_cosh")
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--cosine-weight", type=float, default=0.0)
    parser.add_argument("--decode-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-normalization", choices=("token_layer_norm", "l2", "none"), default="none")
    parser.add_argument("--encoder-microbatch", type=int, default=64)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--vis-interval", type=int, default=500)
    parser.add_argument("--decode-svg-rgb", action="store_true")
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


def build_loader(args: argparse.Namespace):
    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    base_config = _config.get_config(args.openpi_config)
    data_config = dataclasses.replace(base_config.data, future_image_delta_indices=(args.delta_stride,))
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


def feature_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    diff = pred.float() - target.detach().float()
    if args.loss_fn == "mse":
        return diff.square().mean()
    if args.loss_fn == "smooth_l1":
        return F.smooth_l1_loss(pred.float(), target.detach().float(), beta=args.smooth_l1_beta)
    # Stable log(cosh(x)).
    return (diff + F.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, device=diff.device))).mean()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    model_config: FeatureDeltaTokenizerConfig,
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    use_ddp, local_rank, device = lam_utils.setup_ddp()
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    output_dir = Path(args.output_dir)
    if lam_utils.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    data_loader = build_loader(args)
    views = [view.strip() for view in args.views.split(",") if view.strip()]

    encoder = None
    svg_model = None
    if args.teacher == "svg_p":
        if args.feature_normalization != "none":
            raise ValueError("SVG-P teacher requires --feature-normalization none for matched encode/decode.")
        svg_model = lam_utils.load_svg_decoder(args, device)
        feature_dim = args.svg_feature_dim
    else:
        encoder = load_dinov3_patch_encoder(args.dinov3_path).to(device)
        feature_dim = int(encoder.config.hidden_size)
        encoder.eval()
        svg_model = lam_utils.load_svg_decoder(args, device) if args.decode_svg_rgb and lam_utils.is_rank0() else None

    model_dim = feature_dim if args.model_dim <= 0 else args.model_dim
    model_config = FeatureDeltaTokenizerConfig(
        feature_dim=feature_dim,
        model_dim=model_dim,
        token_dim=args.token_dim,
        num_delta_tokens=args.num_delta_tokens,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.heads,
        max_views=max(4, len(views)),
        decode_residual=args.decode_residual,
        cosine_weight=args.cosine_weight,
    )
    model = FeatureDeltaTokenizer(model_config).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if lam_utils.is_rank0():
        logging.info("DeltaTok config: %s", model_config)
        logging.info(
            "teacher=%s views=%s stride=%d feature_dim=%d model_dim=%d M=%d token_dim=%d residual=%s output_dir=%s",
            args.teacher,
            views,
            args.delta_stride,
            feature_dim,
            model_dim,
            args.num_delta_tokens,
            model.module.token_dim if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.token_dim,
            args.decode_residual,
            output_dir,
        )

    iterator = iter(data_loader)
    last_log_time = time.time()
    start_time = last_log_time
    for step in range(1, args.max_steps + 1):
        batch = next(iterator)
        observation, future_images = lam_utils.unpack_batch(batch)
        current_images, future_images_tensor = lam_utils.prepare_image_pair(observation, future_images, views, device)

        if args.teacher == "svg_p":
            current_features = lam_utils.encode_svg_p_features(
                svg_model,
                current_images,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
            future_features = lam_utils.encode_svg_p_features(
                svg_model,
                future_images_tensor,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
        else:
            current_features = lam_utils.encode_dino_features(
                encoder,
                current_images,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )
            future_features = lam_utils.encode_dino_features(
                encoder,
                future_images_tensor,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )

        current_features = lam_utils.normalize_features(current_features, args.feature_normalization)
        future_features = lam_utils.normalize_features(future_features, args.feature_normalization)

        optimizer.zero_grad(set_to_none=True)
        use_amp = args.precision != "float32" and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(current_features, future_features)
            loss = feature_loss(outputs["pred"], future_features, args)
            if args.cosine_weight > 0:
                cosine_loss = 1.0 - F.cosine_similarity(
                    outputs["pred"].float(),
                    future_features.detach().float(),
                    dim=-1,
                ).mean()
                loss = loss + args.cosine_weight * cosine_loss
            else:
                cosine_loss = outputs["pred"].new_tensor(0.0)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        pred = outputs["pred"].detach()
        if (step % args.log_interval == 0 or step == args.max_steps) and lam_utils.is_rank0():
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6)
            last_log_time = now
            with torch.no_grad():
                mse = F.mse_loss(pred.float(), future_features.float())
                copy_mse = F.mse_loss(current_features.float(), future_features.float())
                target_delta = future_features.float() - current_features.float()
                pred_delta = pred.float() - current_features.float()
                delta_ratio = pred_delta.norm() / target_delta.norm().clamp_min(1e-6)
                token_norm = outputs["z_delta"].float().norm(dim=-1).mean()
                target_delta_norm = target_delta.norm(dim=-1).mean()
            logging.info(
                (
                    "step=%d loss=%.6f mse=%.6f cos=%.6f copy_mse=%.6f "
                    "delta_ratio=%.3f z_norm=%.4f target_delta_norm=%.4f grad=%.3f steps/s=%.3f"
                ),
                step,
                float(loss.detach().cpu()),
                float(mse.detach().cpu()),
                float(cosine_loss.detach().cpu()),
                float(copy_mse.detach().cpu()),
                float(delta_ratio.detach().cpu()),
                float(token_norm.detach().cpu()),
                float(target_delta_norm.detach().cpu()),
                float(grad_norm.detach().cpu()),
                steps_per_sec,
            )

        if args.vis_interval > 0 and step % args.vis_interval == 0 and lam_utils.is_rank0():
            pred_decoded = (
                lam_utils.decode_svg_feature_image(
                    svg_model,
                    pred[0, 0],
                    grid_size=args.svg_decode_grid,
                )
                if svg_model is not None and args.decode_svg_rgb
                else None
            )
            lam_utils.save_visualization(
                output_dir / "vis" / f"step_{step:06d}.png",
                current_images[0, 0],
                future_images_tensor[0, 0],
                current_features[0, 0],
                future_features[0, 0],
                pred[0, 0],
                pred_decoded,
            )

        if args.save_interval > 0 and step % args.save_interval == 0 and lam_utils.is_rank0():
            save_checkpoint(output_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, step, args, model_config)

    if lam_utils.is_rank0():
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, args.max_steps, args, model_config)
        logging.info("Finished in %.1f minutes", (time.time() - start_time) / 60.0)
    lam_utils.cleanup_ddp()


if __name__ == "__main__":
    main()
