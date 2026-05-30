# PV-VAE OXE Execution Plan

## Current Decision

The active line is now PV-VAE on OXE, not DeltaTok on LIBERO.

Important correction:

- OXE data path is known: `/data/user/jhe724/workspace/data/OXE`
- This OXE subset is expected to include Bridge and `fractal20220817`.
- The missing item is the pretrained DeltaTok checkpoint, not the OXE data.
- DeltaTok should not be retrained on LIBERO for the current plan.

## Goal

Train and evaluate the PV-VAE-style temporal predictive feature tokenizer on OXE clips:

```text
RGB clip
  -> frozen SVG-P / DINO feature teacher
  -> x [B,V,17,N,D]
  -> PV-VAE encoder
  -> z [B,V,5,N,d]
  -> PV-VAE decoder
  -> x_hat [B,V,17,N,D]
```

The first diagnostic target is not policy rollout. It is whether PV-VAE predicts future feature clips better than static copy.

## PV-VAE Contract

For `temporal_compression=4` and 16 future frames:

```text
input:  [B,V,17,N,D]
latent: [B,V,5,N,d]
output: [B,V,17,N,D]
```

Groups:

```text
group 0: x_0
group 1: x_1 ... x_4
group 2: x_5 ... x_8
group 3: x_9 ... x_12
group 4: x_13 ... x_16
```

Training samples a prefix length `observed_groups`. The encoder sees only the observed prefix; the decoder fills missing future groups with learned pad latents and reconstructs the full clip.

## Metrics To Trust First

Primary:

- `future_mse`: MSE on dropped / unobserved future frames.
- `static_future_mse`: baseline from copying the last observed frame.
- `future_mse / static_future_mse`: must be below 1 to beat static copy.
- `delta_ratio = pred_delta_norm / target_delta_norm`: should not collapse toward 0.

Secondary:

- `cosine_loss`
- `observed_mse`
- SVG decoded RGB visualization
- PCA feature visualization

## Current Code Status

Implemented locally:

- `external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py`
- `external/openpi/scripts/train_predictive_feature_vae_libero.py`

The model is dataset-agnostic once it receives feature clips `[B,V,F,N,D]`.

The current training script is LIBERO-oriented because its loader path uses OpenPI LIBERO configs. OXE support depends on the server-side OpenPI RLDS / TFDS loader and the actual OXE directory layout.

## Confirmed OXE Layout

Current hpc-3 inspection found:

```text
/data/user/jhe724/workspace/data/OXE/bridge_orig_lerobot
/data/user/jhe724/workspace/data/OXE/fractal20220817_data_lerobot
```

The confirmed `fractal20220817_data_lerobot/meta/info.json` uses local LeRobot `parquet + mp4` layout:

```text
fps: 3
video_path: videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
data_path: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
image key: observation.images.image, video, [256, 320, 3]
state: float32 [8]
action: float32 [7]
```

Therefore the next implementation target is not a raw OXE/RLDS dataloader. It is a local LeRobot clip dataloader that can read `bridge_orig_lerobot` first and `fractal20220817_data_lerobot` later.

## Immediate Server Step: Inspect Bridge

Run this first:

```bash
python external/openpi/scripts/inspect_oxe_dataset.py \
  --root /data/user/jhe724/workspace/data/OXE/bridge_orig_lerobot \
  --max-depth 4
```

Then inspect Bridge metadata:

```bash
python - <<'PY'
import json
p="/data/user/jhe724/workspace/data/OXE/bridge_orig_lerobot/meta/info.json"
info=json.load(open(p))
print("repo_id:", info.get("repo_id"))
print("fps:", info.get("fps"))
print("video_path:", info.get("video_path"))
print("data_path:", info.get("data_path"))
print("features:")
for k,v in info.get("features", {}).items():
    print(" ", k, v.get("dtype"), v.get("shape"))
PY
```

## Current Training Entry

Added:

```text
external/openpi/scripts/train_predictive_feature_vae_lerobot.py
```

This script reads local LeRobot directories directly and returns clip batches:

```text
images [B,V,T,C,H,W]
```

Start with loader-only dry run:

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/train_predictive_feature_vae_lerobot.py \
  --lerobot-root /data/user/jhe724/workspace/data/OXE/bridge_orig_lerobot \
  --output-dir runs/pvvae/bridgev2_loader_dryrun \
  --video-keys observation.images.image \
  --future-deltas 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --batch-size 2 \
  --num-workers 0 \
  --max-episodes 8 \
  --samples-per-episode 4 \
  --dry-run-loader
```

## What We Need From The Inspection

For each OXE subset, especially Bridge and `fractal20220817`, identify:

- dataset directory name and version
- split names
- image keys
- action key and shape
- language / instruction key
- episode / steps nesting
- image dtype and shape
- whether frames are already decoded or stored as encoded bytes

## Likely Next Code Step

After inspection, add an OXE clip loader that returns the same minimal clip batch contract:

```python
{
    "images": Tensor[B, V, T, C, H, W],
    "actions": Tensor[B, T - 1, A] | None,
    "instruction": list[str] | None,
    "dataset_name": list[str],
    "episode_id": list[str],
}
```

Then reuse the existing PV-VAE feature encoder:

```text
images -> encode_svg_p_clip / encode_dino_clip -> features [B,V,T,N,D]
```

## First PV-VAE OXE Smoke

Once OXE loading works, use a small model first:

```text
teacher: SVG-P or DINOv3
views: first available camera
T: 17 frames
temporal_compression: 4
batch_size: 4 or 8
max_steps: 100
observed_groups: random prefix
```

Acceptable smoke result:

- no dataloader errors
- no NaN
- checkpoints saved
- `future_mse`, `static_future_mse`, `delta_ratio` logged
- visualization saved if SVG-P decode is available

Research success requires more:

- `future_mse < static_future_mse`
- `delta_ratio` not collapsed
- predicted future visualization is not pure copy / average
