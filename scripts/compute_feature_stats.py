#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


DEFAULT_DATASET_ROOT = "/data/LFT-W02_data/junjie/data/libero_mask_depth"
DEFAULT_DATASETS = (
    "libero_spatial_lerobot_mask_depth",
    "libero_object_lerobot_mask_depth",
    "libero_goal_lerobot_mask_depth",
    "libero_10_lerobot_mask_depth",
)
DEFAULT_DINOV3_VITS = "/home/LFT-W02/.cache/modelscope/hub/models/facebook/dinov3-vits16-pretrain-lvd1689m"


os.environ.setdefault("HF_DATASETS_CACHE", str((Path.cwd() / "runs" / "hf_cache").resolve()))


class FastWAMFeatureStatsDataset(Dataset):
    def __init__(
        self,
        fastwam_root: Path,
        dataset_root: Path,
        dataset_names: list[str],
        camera_keys: list[str],
        num_frames: int,
        action_video_freq_ratio: int,
        video_size: int,
    ) -> None:
        sys.path.insert(0, str(fastwam_root / "src"))
        from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset

        shape_meta = {
            "images": [
                {"key": key, "raw_shape": [3, 256, 256], "shape": [3, video_size, video_size]}
                for key in camera_keys
            ],
            "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
            "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
        }
        dataset_dirs = [str(dataset_root / name) for name in dataset_names]
        self.base = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=0.0,
            is_training_set=True,
            global_sample_stride=1,
        )
        self.base._set_return_images(True)
        self.camera_keys = camera_keys
        self.video_indices = list(range(0, num_frames, action_video_freq_ratio))
        self.video_size = int(video_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        videos = []
        for key in self.camera_keys:
            frames = sample["images"][key][self.video_indices]  # [T,C,H,W], uint8
            frames = frames.to(torch.float32) / 255.0
            if frames.shape[-2:] != (self.video_size, self.video_size):
                frames = F.interpolate(
                    frames,
                    size=(self.video_size, self.video_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            videos.append(frames)
        video = torch.stack(videos, dim=0).contiguous()  # [V,T,C,H,W], range [0,1]
        image_is_pad = sample["image_is_pad"][self.video_indices].to(torch.bool)
        return {"video": video, "image_is_pad": image_is_pad, "idx": torch.tensor(idx, dtype=torch.long)}


def build_indices(total: int, sample_stride: int, max_samples: int | None) -> list[int]:
    indices = list(range(0, total, max(1, sample_stride)))
    if max_samples is not None and len(indices) > max_samples:
        lin = torch.linspace(0, len(indices) - 1, max_samples).round().to(torch.long)
        indices = [indices[i] for i in lin.unique().tolist()]
    return indices


def update_stats(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    sum_chw: torch.Tensor | None,
    sumsq_chw: torch.Tensor | None,
    sum_c: torch.Tensor | None,
    sumsq_c: torch.Tensor | None,
    count_bt: int,
    count_c: int,
):
    # features: [B,C,T,H,W], valid_mask: [B,T]
    features = features.float()
    valid = (~valid_mask.bool()).to(device=features.device, dtype=features.dtype)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return sum_chw, sumsq_chw, sum_c, sumsq_c, count_bt, count_c

    valid = valid[:, None, :, None, None]
    masked = features * valid
    batch_sum_chw = masked.sum(dim=(0, 2)).cpu()
    batch_sumsq_chw = (features.square() * valid).sum(dim=(0, 2)).cpu()
    batch_sum_c = batch_sum_chw.sum(dim=(1, 2))
    batch_sumsq_c = batch_sumsq_chw.sum(dim=(1, 2))

    if sum_chw is None:
        sum_chw = torch.zeros_like(batch_sum_chw)
        sumsq_chw = torch.zeros_like(batch_sumsq_chw)
        sum_c = torch.zeros_like(batch_sum_c)
        sumsq_c = torch.zeros_like(batch_sumsq_c)

    sum_chw += batch_sum_chw
    sumsq_chw += batch_sumsq_chw
    sum_c += batch_sum_c
    sumsq_c += batch_sumsq_c
    count_bt += valid_count
    count_c += valid_count * features.shape[-2] * features.shape[-1]
    return sum_chw, sumsq_chw, sum_c, sumsq_c, count_bt, count_c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, default=Path("external/FastWAM"))
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-names", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--camera-keys", nargs="+", default=["image", "wrist_image"])
    parser.add_argument("--model", default=DEFAULT_DINOV3_VITS)
    parser.add_argument("--output", type=Path, default=Path("runs/feature_stats/dinov3_vits16_libero_2cam224_stat.pt"))
    parser.add_argument("--video-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--action-video-freq-ratio", type=int, default=2)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=72)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    args.fastwam_root = args.fastwam_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = FastWAMFeatureStatsDataset(
        fastwam_root=args.fastwam_root,
        dataset_root=args.dataset_root,
        dataset_names=args.dataset_names,
        camera_keys=args.camera_keys,
        num_frames=args.num_frames,
        action_video_freq_ratio=args.action_video_freq_ratio,
        video_size=args.video_size,
    )
    indices = build_indices(len(dataset), args.sample_stride, args.max_samples)
    subset = Subset(dataset, indices)

    sys.path.insert(0, str(args.fastwam_root / "src"))
    from fastwam.models.vision import DINOv3FeatureEncoder

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    encoder = DINOv3FeatureEncoder(
        model_name_or_path=args.model,
        image_size=args.video_size,
        microbatch_size=args.microbatch_size,
        torch_dtype=dtype,
        device=args.device,
        trust_remote_code=True,
    ).eval()

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    sum_chw = sumsq_chw = sum_c = sumsq_c = None
    count_bt = 0
    count_c = 0
    start = time.time()
    last_shape = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="feature stats", total=len(loader)):
            video = batch["video"].to(args.device, non_blocking=True)
            features = encoder(video)
            last_shape = tuple(features.shape)
            sum_chw, sumsq_chw, sum_c, sumsq_c, count_bt, count_c = update_stats(
                features=features,
                valid_mask=batch["image_is_pad"],
                sum_chw=sum_chw,
                sumsq_chw=sumsq_chw,
                sum_c=sum_c,
                sumsq_c=sumsq_c,
                count_bt=count_bt,
                count_c=count_c,
            )

    if count_bt == 0 or sum_chw is None or sumsq_chw is None or sum_c is None or sumsq_c is None:
        raise RuntimeError("No valid frames were processed; cannot write stats.")

    mean = sum_chw / count_bt
    var = (sumsq_chw / count_bt - mean.square()).clamp_min(0.0)
    channel_mean = sum_c / count_c
    channel_var = (sumsq_c / count_c - channel_mean.square()).clamp_min(0.0)
    payload = {
        "mean": mean,
        "var": var,
        "std": torch.sqrt(var + 1e-6),
        "channel_mean": channel_mean,
        "channel_var": channel_var,
        "channel_std": torch.sqrt(channel_var + 1e-6),
        "count_bt": count_bt,
        "count_c": count_c,
        "feature_shape_last_batch": last_shape,
        "normalization_axes": "mean/var are over batch and time, shape [C,H,W_concat]; channel_* are over batch,time,height,width.",
        "config": vars(args) | {
            "fastwam_root": str(args.fastwam_root),
            "dataset_root": str(args.dataset_root),
            "output": str(args.output),
            "num_dataset_samples": len(dataset),
            "num_selected_samples": len(indices),
            "elapsed_seconds": time.time() - start,
        },
    }
    torch.save(payload, args.output)
    summary = {
        "output": str(args.output),
        "mean_shape": list(mean.shape),
        "channel_mean_shape": list(channel_mean.shape),
        "count_bt": count_bt,
        "count_c": count_c,
        "selected_samples": len(indices),
        "elapsed_seconds": round(payload["config"]["elapsed_seconds"], 2),
        "mean_abs": round(float(mean.abs().mean()), 6),
        "std_mean": round(float(payload["std"].mean()), 6),
        "channel_std_mean": round(float(payload["channel_std"].mean()), 6),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
