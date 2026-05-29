#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoModel


def read_frame(video_path: Path, frame_idx: int) -> Image.Image:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(frame_idx)
    finally:
        reader.close()
    return Image.fromarray(frame).convert("RGB")


def prepare_images(images: list[Image.Image], image_size: int) -> torch.Tensor:
    arrays = []
    for image in images:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arrays.append(arr)
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def pca_rgb(features: torch.Tensor) -> torch.Tensor:
    # features: [N, H, W, D]. Fit one PCA basis across all samples so colors are comparable.
    n, h, w, d = features.shape
    tokens = torch.nan_to_num(features.reshape(-1, d).float())
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:3].T
    lo = projected.amin(dim=0, keepdim=True)
    hi = projected.amax(dim=0, keepdim=True)
    projected = ((projected - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    return projected.reshape(n, h, w, 3)


def to_uint8_image(rgb: torch.Tensor) -> Image.Image:
    arr = (rgb.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def add_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 20), "white")
    canvas.paste(image, (0, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill=(0, 0, 0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--episode", default="episode_000005.mp4")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 8, 16])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_root = Path(args.video_root)
    camera_videos = {
        "image": video_root / "observation.images.image" / args.episode,
        "wrist": video_root / "observation.images.wrist_image" / args.episode,
    }
    samples: list[tuple[str, int, Image.Image]] = []
    for camera, video_path in camera_videos.items():
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        for frame_idx in args.frames:
            samples.append((camera, frame_idx, read_frame(video_path, frame_idx)))

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval()
    config = model.config
    image_size = int(getattr(config, "image_size", 224))
    patch_size = int(getattr(config, "patch_size", 16))
    hidden_size = int(getattr(config, "hidden_size"))
    grid = image_size // patch_size
    num_patch_tokens = grid * grid

    pixel_values = prepare_images([sample[2] for sample in samples], image_size=image_size)
    with torch.no_grad():
        output = model(pixel_values=pixel_values)
    patch_tokens = output.last_hidden_state[:, -num_patch_tokens:, :]
    features = patch_tokens.reshape(len(samples), grid, grid, hidden_size)
    pca = pca_rgb(features)

    rows = []
    for idx, (camera, frame_idx, image) in enumerate(samples):
        rgb = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        lowres = to_uint8_image(pca[idx])
        nearest = lowres.resize((image_size, image_size), Image.Resampling.NEAREST)
        bilinear = lowres.resize((image_size, image_size), Image.Resampling.BILINEAR)

        stem = f"{camera}_frame{frame_idx:04d}"
        rgb.save(output_dir / f"{stem}_rgb.png")
        lowres.save(output_dir / f"{stem}_pca_14x14.png")
        nearest.save(output_dir / f"{stem}_pca_224_nearest.png")
        bilinear.save(output_dir / f"{stem}_pca_224_bilinear.png")

        cells = [
            add_label(rgb, f"{camera} f{frame_idx} RGB"),
            add_label(nearest, "PCA nearest"),
            add_label(bilinear, "PCA bilinear"),
        ]
        row = Image.new("RGB", (sum(cell.width for cell in cells), cells[0].height), "white")
        x = 0
        for cell in cells:
            row.paste(cell, (x, 0))
            x += cell.width
        rows.append(row)

    contact = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        contact.paste(row, (0, y))
        y += row.height
    contact.save(output_dir / "dinov3_vits_feature_probe_contact.png")

    print(f"model={args.model}")
    print(f"feature_shape=[{len(samples)}, {grid}, {grid}, {hidden_size}]")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
