#!/usr/bin/env python
"""Train a per-frame semantic feature VAE on LIBERO teacher features.

This is the simplest feature-tokenizer baseline:

    image clip -> teacher features [B,V,F,N,D]
    flatten frames -> [B*F,V,N,D]
    S-VAE -> latent tokens [B*F,V,N,d] -> reconstructed features

It intentionally does not learn temporal prediction. Use it as the channel
compression baseline against PV-VAE and DeltaTok.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.parallel

# Allow running this script from the outer monorepo without installing openpi.
_OPENPI_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))

from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder
from openpi.models_pytorch.semantic_feature_vae import SemanticFeatureVAE, SemanticFeatureVAEConfig

import train_lam_libero as lam_utils
import train_predictive_feature_vae_libero as clip_utils


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-config", default="pi05_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", choices=("svg_p", "dinov3_vits16"), default="svg_p")
    parser.add_argument("--dinov3-path", default="assets/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--views", default="base_0_rgb")
    parser.add_argument("--future-deltas", default="1,3,6,9")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--feature-normalization", choices=("none", "l2", "token_layer_norm"), default="none")
    parser.add_argument("--encoder-microbatch", type=int, default=16)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--vis-interval", type=int, default=1000)
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    model_config: SemanticFeatureVAEConfig,
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

    future_deltas = parse_int_list(args.future_deltas)
    if not future_deltas:
        raise ValueError("--future-deltas must contain at least one future frame index.")

    output_dir = Path(args.output_dir)
    if lam_utils.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    data_loader = clip_utils.build_loader(args, future_deltas)
    views = [view.strip() for view in args.views.split(",") if view.strip()]

    encoder = None
    svg_model = None
    if args.teacher == "svg_p":
        if args.feature_normalization != "none":
            raise ValueError("SVG-P teacher requires --feature-normalization none for matched encode/decode.")
        svg_model = lam_utils.load_svg_decoder(args, device)
        feature_dim = args.svg_feature_dim
    else:
        encoder = load_dinov3_patch_encoder(args.dinov3_path).to(device).eval()
        feature_dim = int(encoder.config.hidden_size)
        if args.decode_svg_rgb and lam_utils.is_rank0():
            svg_model = lam_utils.load_svg_decoder(args, device)

    model_config = SemanticFeatureVAEConfig(
        feature_dim=feature_dim,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_views=max(4, len(views)),
        kl_weight=args.kl_weight,
        cosine_weight=args.cosine_weight,
    )
    model = SemanticFeatureVAE(model_config).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if lam_utils.is_rank0():
        raw_model = model.module if use_ddp else model
        param_count = sum(param.numel() for param in raw_model.parameters())
        logging.info("SemanticFeatureVAE config: %s", model_config)
        logging.info(
            "params=%.1fM teacher=%s views=%s future_deltas=%s latent_dim=%d output_dir=%s",
            param_count / 1e6,
            args.teacher,
            views,
            future_deltas,
            args.latent_dim,
            output_dir,
        )

    iterator = iter(data_loader)
    start_time = time.time()
    last_log_time = start_time
    for step in range(1, args.max_steps + 1):
        batch = next(iterator)
        observation, future_images = lam_utils.unpack_batch(batch)
        image_clip = clip_utils.prepare_image_clip(observation, future_images, views, device)

        if args.teacher == "svg_p":
            features = clip_utils.encode_svg_p_clip(
                svg_model,
                image_clip,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
        else:
            features = clip_utils.encode_dino_clip(
                encoder,
                image_clip,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )
        features = lam_utils.normalize_features(features, args.feature_normalization)

        batch_size, num_views, num_frames, num_tokens, dim = features.shape
        frame_features = features.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, num_views, num_tokens, dim)

        optimizer.zero_grad(set_to_none=True)
        use_amp = args.precision != "float32" and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss_dict = model.module.compute_loss(frame_features) if use_ddp else model.compute_loss(frame_features)
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if (step % args.log_interval == 0 or step == args.max_steps) and lam_utils.is_rank0():
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6)
            last_log_time = now
            logging.info(
                "step=%d loss=%.6f recon=%.6f cos=%.6f kl=%.6f z_norm=%.4f target_norm=%.4f grad=%.3f steps/s=%.3f",
                step,
                float(loss_dict["loss"].detach().cpu()),
                float(loss_dict["recon_loss"].detach().cpu()),
                float(loss_dict["cosine_loss"].detach().cpu()),
                float(loss_dict["kl_loss"].detach().cpu()),
                float(loss_dict["latent_norm"].detach().cpu()),
                float(loss_dict["target_norm"].detach().cpu()),
                float(grad_norm.detach().cpu()),
                steps_per_sec,
            )

        if args.vis_interval > 0 and step % args.vis_interval == 0 and lam_utils.is_rank0():
            pred_flat = loss_dict["pred"]
            pred_clip = pred_flat.view(batch_size, num_frames, num_views, num_tokens, dim).permute(0, 2, 1, 3, 4)
            frame_idx = min(num_frames - 1, 1)
            pred_decoded = None
            if args.decode_svg_rgb and svg_model is not None:
                pred_decoded = lam_utils.decode_svg_feature_image(
                    svg_model,
                    pred_clip[0, 0, frame_idx],
                    grid_size=args.svg_decode_grid,
                )
            lam_utils.save_visualization(
                output_dir / "vis" / f"step_{step:06d}.png",
                image_clip[0, 0, 0],
                image_clip[0, 0, frame_idx],
                features[0, 0, 0],
                features[0, 0, frame_idx],
                pred_clip[0, 0, frame_idx],
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
