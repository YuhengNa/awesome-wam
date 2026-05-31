#!/usr/bin/env python
"""Train PV-VAE on a local LeRobot dataset directory.

This is the OXE / Bridge-v2 path. It bypasses OpenPI's named LIBERO config and
reads a local LeRobot layout directly:

    root/
      meta/info.json
      data/chunk-xxx/episode_yyyyyy.parquet
      videos/chunk-xxx/<video_key>/episode_yyyyyy.mp4

The batch contract is a clip tensor `[B,V,T,C,H,W]`, which is then encoded by a
frozen SVG-P or DINO teacher into `[B,V,T,N,D]` for `PredictiveFeatureVAE`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import random
import re
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader, Dataset

from openpi.models_pytorch.predictive_feature_vae import PredictiveFeatureVAE, PredictiveFeatureVAEConfig


EPISODE_RE = re.compile(r"episode_(\d+)")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", required=True, help="Local LeRobot dataset root, e.g. bridge_orig_lerobot.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", choices=("svg_p", "dinov3_vits16"), default="svg_p")
    parser.add_argument("--dinov3-path", default="assets/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--video-keys", default="observation.images.image")
    parser.add_argument("--future-deltas", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=512)
    parser.add_argument("--episode-offset", type=int, default=0, help="Skip this many valid episodes before sampling clips.")
    parser.add_argument("--samples-per-episode", type=int, default=64)
    parser.add_argument("--dry-run-loader", action="store_true", help="Build dataset and print one batch without loading teachers.")
    parser.add_argument("--overfit-first-batch", action="store_true", help="Reuse the first DataLoader batch for every step.")
    parser.add_argument("--min-rgb-delta", type=float, default=0.0, help="Reject clips whose max mean RGB delta to frame 0 is below this.")
    parser.add_argument("--min-rgb-mean", type=float, default=0.0, help="Reject near-black clips whose mean RGB is below this.")
    parser.add_argument("--min-rgb-std", type=float, default=0.0, help="Reject flat clips whose RGB std is below this.")
    parser.add_argument("--max-resample-attempts", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
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
    parser.add_argument("--observed-groups", type=int, default=0)
    parser.add_argument("--min-observed-groups", type=int, default=1)
    parser.add_argument("--feature-normalization", choices=("none", "l2", "token_layer_norm", "channel_standard"), default="none")
    parser.add_argument("--feature-stats", default=None, help="Path to stats .pt produced by compute_lerobot_feature_stats.py.")
    parser.add_argument("--encoder-microbatch", type=int, default=16)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--vis-interval", type=int, default=500)
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


def parquet_num_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        import pandas as pd  # type: ignore

        return int(len(pd.read_parquet(path, columns=["frame_index"])))


def read_video_frames(path: Path, frame_indices: list[int]) -> torch.Tensor:
    """Return selected frames as float tensor [T,C,H,W] in [0,1]."""
    try:
        from torchvision.io import read_video  # type: ignore

        video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
        frames = video[torch.as_tensor(frame_indices, dtype=torch.long)]
        return frames.float().div(255.0).clamp(0.0, 1.0)
    except Exception as first_error:
        try:
            import imageio.v3 as iio  # type: ignore

            arr = iio.imread(path, index=frame_indices)
            frames = torch.as_tensor(arr)
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            frames = frames.permute(0, 3, 1, 2)
            return frames.float().div(255.0).clamp(0.0, 1.0)
        except Exception as second_error:
            raise RuntimeError(f"Failed to read {path}: {first_error}; fallback: {second_error}") from second_error


class LocalLeRobotClipDataset(Dataset):
    def __init__(
        self,
        root: Path,
        *,
        video_keys: list[str],
        future_deltas: tuple[int, ...],
        max_episodes: int,
        episode_offset: int,
        samples_per_episode: int,
        seed: int,
        min_rgb_delta: float,
        min_rgb_mean: float,
        min_rgb_std: float,
        max_resample_attempts: int,
    ):
        self.root = root
        self.video_keys = video_keys
        self.future_deltas = future_deltas
        self.frame_offsets = [0, *future_deltas]
        self.rng = random.Random(seed)
        self.min_rgb_delta = min_rgb_delta
        self.min_rgb_mean = min_rgb_mean
        self.min_rgb_std = min_rgb_std
        self.max_resample_attempts = max(max_resample_attempts, 1)
        self.info = self._load_info()
        self.video_paths = self._index_video_paths(video_keys)
        self.episodes = self._index_episodes(max_episodes, episode_offset)
        self.samples = self._build_samples(samples_per_episode)
        if not self.samples:
            raise ValueError(f"No valid clips found in {root}")

    def _load_info(self) -> dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(info_path)
        return json.loads(info_path.read_text())

    def _index_video_paths(self, video_keys: list[str]) -> dict[str, dict[tuple[str, int], Path]]:
        result: dict[str, dict[tuple[str, int], Path]] = {}
        for key in video_keys:
            paths = sorted((self.root / "videos").glob(f"chunk-*/{key}/episode_*.mp4"))
            if not paths:
                paths = sorted((self.root / "videos").glob(f"chunk-*/*{key}*/episode_*.mp4"))
            mapping = {episode_key_from_video_path(path): path for path in paths}
            if not mapping:
                raise FileNotFoundError(f"No videos found for key={key} under {self.root / 'videos'}")
            if len(paths) > len(mapping):
                logging.warning(
                    "video key %s matched %d files but only %d unique episodes; "
                    "this usually means the key is ambiguous across camera views. "
                    "Prefer an explicit key such as observation.images.image_0.",
                    key,
                    len(paths),
                    len(mapping),
                )
            result[key] = mapping
        return result

    def _index_episodes(self, max_episodes: int, episode_offset: int) -> list[dict[str, Any]]:
        parquet_paths = sorted((self.root / "data").glob("chunk-*/episode_*.parquet"))
        episodes = []
        valid_seen = 0
        for parquet_path in parquet_paths:
            episode_index = episode_index_from_path(parquet_path)
            episode_key = episode_key_from_data_path(parquet_path)
            if any(episode_key not in self.video_paths[key] for key in self.video_keys):
                continue
            num_rows = parquet_num_rows(parquet_path)
            if num_rows <= max(self.future_deltas):
                continue
            if valid_seen < episode_offset:
                valid_seen += 1
                continue
            episodes.append({"episode_index": episode_index, "episode_key": episode_key, "num_rows": num_rows, "parquet": parquet_path})
            valid_seen += 1
            if max_episodes > 0 and len(episodes) >= max_episodes:
                break
        return episodes

    def _build_samples(self, samples_per_episode: int) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        max_delta = max(self.future_deltas)
        for ep_idx, episode in enumerate(self.episodes):
            max_start = episode["num_rows"] - 1 - max_delta
            starts = list(range(max_start + 1))
            if samples_per_episode > 0 and len(starts) > samples_per_episode:
                starts = sorted(self.rng.sample(starts, samples_per_episode))
            samples.extend((ep_idx, start) for start in starts)
        self.rng.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def clip_rgb_delta(images: torch.Tensor) -> float:
        # images: [V,T,C,H,W]
        delta = (images[:, 1:] - images[:, :1]).abs().mean(dim=(2, 3, 4))
        return float(delta.max())

    @staticmethod
    def clip_rgb_quality(images: torch.Tensor) -> tuple[float, float]:
        return float(images.mean()), float(images.std())

    def _passes_rgb_filter(self, sample: dict[str, Any]) -> bool:
        return (
            sample["rgb_delta"] >= self.min_rgb_delta
            and sample["rgb_mean"] >= self.min_rgb_mean
            and sample["rgb_std"] >= self.min_rgb_std
        )

    def _get_sample(self, index: int) -> dict[str, Any]:
        episode_list_index, start = self.samples[index]
        episode = self.episodes[episode_list_index]
        frame_indices = [start + offset for offset in self.frame_offsets]
        per_view = []
        for key in self.video_keys:
            frames = read_video_frames(self.video_paths[key][episode["episode_key"]], frame_indices)
            per_view.append(frames)
        images = torch.stack(per_view, dim=0)
        rgb_delta = self.clip_rgb_delta(images)
        rgb_mean, rgb_std = self.clip_rgb_quality(images)
        return {
            "images": images,
            "rgb_delta": rgb_delta,
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
            "dataset_name": self.root.name,
            "episode_id": str(episode["episode_index"]),
            "episode_key": episode["episode_key"],
            "start_index": start,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._get_sample(index)
        if self._passes_rgb_filter(sample):
            return sample
        for _ in range(self.max_resample_attempts - 1):
            new_index = random.randrange(len(self.samples))
            sample = self._get_sample(new_index)
            if self._passes_rgb_filter(sample):
                break
        return sample


def collate_clip_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "rgb_delta": torch.as_tensor([item["rgb_delta"] for item in batch], dtype=torch.float32),
        "rgb_mean": torch.as_tensor([item["rgb_mean"] for item in batch], dtype=torch.float32),
        "rgb_std": torch.as_tensor([item["rgb_std"] for item in batch], dtype=torch.float32),
        "dataset_name": [item["dataset_name"] for item in batch],
        "episode_id": [item["episode_id"] for item in batch],
        "start_index": torch.as_tensor([item["start_index"] for item in batch], dtype=torch.long),
    }


def choose_observed_groups(args: argparse.Namespace, total_groups: int) -> int:
    if args.observed_groups > 0:
        return min(args.observed_groups, total_groups)
    min_groups = min(max(args.min_observed_groups, 1), total_groups)
    return random.randint(min_groups, total_groups)


def load_feature_stats(path: str | None, device: torch.device) -> dict[str, torch.Tensor] | None:
    if path is None:
        return None
    stats = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "mean": stats["mean"].to(device=device, dtype=torch.float32).view(1, 1, 1, 1, -1),
        "std": stats["std"].to(device=device, dtype=torch.float32).clamp_min(1e-6).view(1, 1, 1, 1, -1),
    }


def normalize_feature_clip(features: torch.Tensor, mode: str, stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    import train_lam_libero as lam_utils

    if mode == "channel_standard":
        if stats is None:
            raise ValueError("--feature-normalization channel_standard requires --feature-stats.")
        return (features - stats["mean"]) / stats["std"]
    return lam_utils.normalize_features(features, mode)


def denormalize_feature_clip(features: torch.Tensor, mode: str, stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if mode == "channel_standard":
        if stats is None:
            raise ValueError("--feature-normalization channel_standard requires --feature-stats.")
        return features * stats["std"] + stats["mean"]
    return features


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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import train_lam_libero as lam_utils
    import train_predictive_feature_vae_libero as pv_utils

    use_ddp, local_rank, device = lam_utils.setup_ddp()
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)

    future_deltas = parse_int_list(args.future_deltas)
    if len(future_deltas) % args.temporal_compression != 0:
        raise ValueError("Number of future frames must be divisible by temporal compression.")
    num_frames = 1 + len(future_deltas)
    total_groups = 1 + len(future_deltas) // args.temporal_compression
    video_keys = [key.strip() for key in args.video_keys.split(",") if key.strip()]

    dataset = LocalLeRobotClipDataset(
        Path(args.lerobot_root),
        video_keys=video_keys,
        future_deltas=future_deltas,
        max_episodes=args.max_episodes,
        episode_offset=args.episode_offset,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        min_rgb_delta=args.min_rgb_delta,
        min_rgb_mean=args.min_rgb_mean,
        min_rgb_std=args.min_rgb_std,
        max_resample_attempts=args.max_resample_attempts,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_clip_batch,
    )

    output_dir = Path(args.output_dir)
    if lam_utils.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
        logging.info(
            "LeRobot dataset=%s episodes=%d samples=%d fps=%s video_keys=%s",
            args.lerobot_root,
            len(dataset.episodes),
            len(dataset),
            dataset.info.get("fps"),
            video_keys,
        )
        logging.info(
            "motion filter min_rgb_delta=%.6f min_rgb_mean=%.6f min_rgb_std=%.6f max_resample_attempts=%d",
            args.min_rgb_delta,
            args.min_rgb_mean,
            args.min_rgb_std,
            args.max_resample_attempts,
        )

    if args.dry_run_loader:
        batch = next(iter(loader))
        images = batch["images"]
        if lam_utils.is_rank0():
            logging.info(
                (
                    "dry_run batch images shape=%s dtype=%s min=%.4f max=%.4f "
                    "rgb_delta_shape=%s rgb_delta_mean=%.6f rgb_delta_max=%.6f "
                    "rgb_mean=%.6f rgb_std=%.6f "
                    "dataset=%s episode=%s start=%s"
                ),
                tuple(images.shape),
                images.dtype,
                float(images.min()),
                float(images.max()),
                tuple(batch["rgb_delta"].shape),
                float(batch["rgb_delta"].mean()),
                float(batch["rgb_delta"].max()),
                float(batch["rgb_mean"].mean()),
                float(batch["rgb_std"].mean()),
                batch["dataset_name"][:4],
                batch["episode_id"][:4],
                batch["start_index"][:4].tolist(),
            )
        lam_utils.cleanup_ddp()
        return

    encoder = None
    svg_model = None
    if args.teacher == "svg_p":
        svg_args = argparse.Namespace(**vars(args))
        svg_args.feature_normalization = "none"
        svg_model = lam_utils.load_svg_decoder(svg_args, device)
        feature_dim = args.svg_feature_dim
    else:
        from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder

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
        max_views=max(4, len(video_keys)),
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
    feature_stats = load_feature_stats(args.feature_stats, device)
    if args.feature_normalization == "channel_standard" and lam_utils.is_rank0():
        logging.info("Using channel_standard feature stats from %s", args.feature_stats)

    if lam_utils.is_rank0():
        param_count = sum(p.numel() for p in (model.module if use_ddp else model).parameters())
        logging.info("PredictiveFeatureVAE config: %s", model_config)
        logging.info(
            "params=%.1fM teacher=%s video_keys=%s future_deltas=%s latent_groups=%d output_dir=%s",
            param_count / 1e6,
            args.teacher,
            video_keys,
            future_deltas,
            total_groups,
            output_dir,
        )

    iterator = iter(loader)
    fixed_batch = None
    if args.overfit_first_batch:
        fixed_batch = next(iterator)
        if lam_utils.is_rank0():
            logging.info(
                "overfit_first_batch enabled: reusing episode=%s start=%s",
                fixed_batch["episode_id"],
                fixed_batch["start_index"].tolist(),
            )
    start_time = time.time()
    last_log_time = start_time
    for step in range(1, args.max_steps + 1):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

        image_clip = batch["images"].to(device, non_blocking=True)
        if args.teacher == "svg_p":
            features = pv_utils.encode_svg_p_clip(
                svg_model,
                image_clip,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
                image_size=args.teacher_image_size,
            )
        else:
            features = pv_utils.encode_dino_clip(
                encoder,
                image_clip,
                microbatch=args.encoder_microbatch,
                precision=args.precision,
            )
        features = normalize_feature_clip(features, args.feature_normalization, feature_stats)

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
            static_future_mse = loss_dict["static_future_mse"].detach()
            future_mse = loss_dict["future_mse"].detach()
            future_copy_ratio = future_mse / static_future_mse.clamp_min(1e-6)
            logging.info(
                (
                    "step=%d loss=%.6f recon=%.6f cos=%.6f delta=%.6f kl=%.6f "
                    "obs_mse=%.6f future_mse=%.6f static_future_mse=%.6f "
                    "future_copy_ratio=%.3f pred_d=%.6f gt_d=%.6f d_ratio=%.3f "
                    "rgb_delta_mean=%.6f rgb_delta_max=%.6f rgb_mean=%.6f rgb_std=%.6f "
                    "obs_groups=%d grad=%.3f steps/s=%.3f"
                ),
                step,
                float(loss_dict["loss"].detach().cpu()),
                float(loss_dict["recon_loss"].detach().cpu()),
                float(loss_dict["cosine_loss"].detach().cpu()),
                float(loss_dict["delta_loss"].detach().cpu()),
                float(loss_dict["kl_loss"].detach().cpu()),
                float(loss_dict["observed_mse"].detach().cpu()),
                float(future_mse.cpu()),
                float(static_future_mse.cpu()),
                float(future_copy_ratio.cpu()),
                float(loss_dict["pred_delta_norm"].detach().cpu()),
                float(loss_dict["target_delta_norm"].detach().cpu()),
                float(loss_dict["delta_ratio"].detach().cpu()),
                float(batch["rgb_delta"].mean().cpu()),
                float(batch["rgb_delta"].max().cpu()),
                float(batch["rgb_mean"].mean().cpu()),
                float(batch["rgb_std"].mean().cpu()),
                observed_groups,
                float(grad_norm.detach().cpu()),
                steps_per_sec,
            )

        if args.vis_interval > 0 and step % args.vis_interval == 0 and lam_utils.is_rank0():
            pred = loss_dict["pred"].detach()
            vis_features = denormalize_feature_clip(features, args.feature_normalization, feature_stats)
            vis_pred = denormalize_feature_clip(pred, args.feature_normalization, feature_stats)
            pv_utils.save_visualization(
                output_dir / "vis" / f"step_{step:06d}.png",
                image_clip[0, 0],
                vis_features[0, 0],
                vis_pred[0, 0],
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
