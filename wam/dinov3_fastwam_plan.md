# FastWAM-DINOv3 Implementation Plan

## Goal

Implement a DINOv3 variant of FastWAM that changes only the world prediction target:

- Baseline target: image-like future frames encoded by Wan VAE.
- New target: future RGB frames encoded online by frozen DINOv3 patch features.

The first version must keep the original FastWAM behavior otherwise: `action_conditioned=false`, unchanged action branch, unchanged MoT attention mask, same aligned LIBERO data, same video/action horizons.

## Non-Goals For The First Version

- Do not cache DINO features.
- Do not train a feature adapter.
- Do not make video prediction action-conditioned.
- Do not let the action branch attend to future predicted DINO tokens.
- Do not add pixel/video decoding metrics for DINO features.

These are later ablations, not part of the representation-only baseline.

## Target Data Shape

Use online DINOv3-S with `patch_size=16` and `224x224` camera images.

For the current aligned LIBERO setting:

- input RGB clips: `[B, V, T, 3, 224, 224]`
- cameras: `V=2`
- video frames: `T=9`
- DINOv3 patch grid per camera: `14 x 14`
- feature dim: `D=384`
- model feature grid: `[B, D, T, 14, 28]` after concatenating the two camera grids along width

This mirrors the old video latent interface `[B, C, T, H, W]`, allowing reuse of the Wan video DiT tokenization path with new input/output channel dimensions.

## Implementation Steps

### 1. Preserve Multi-Camera RGB Inputs

Add a DINO data path that does not horizontally concatenate camera images before feature extraction. The dataset or model input builder should expose RGB as `[B, V, T, C, H, W]`.

Keep the existing RGB/depth/seg VAE path unchanged.

### 2. Add Online DINOv3 Encoder

Create a frozen encoder module, for example `src/fastwam/models/vision/dinov3.py`.

Requirements:

- load DINOv3-S from a configurable local/Hugging Face path;
- call `eval()` and `requires_grad_(False)`;
- run under `torch.no_grad()`;
- use the official DINOv3 image processor or equivalent ImageNet preprocessing;
- support microbatching over flattened `B * V * T` images;
- return patch tokens only, excluding CLS/register tokens unless explicitly enabled later.

### 3. Add `FastWAMDINO`

Create a new model class, for example `src/fastwam/models/wan22/fastwam_dino.py`.

Reuse from `FastWAM`:

- text/context handling;
- proprio token handling;
- action expert;
- MoT execution;
- action loss;
- flow-matching scheduler.

Replace:

- `vae.encode(...)` with online DINOv3 feature extraction;
- VAE temporal mask alignment with direct frame-level DINO mask alignment;
- decode/video visualization with feature-space diagnostics.

The first frame should still be the conditioning frame. Video loss should exclude frame 0, matching the current FastWAM behavior when first-frame latent is fused.

### 4. Reuse Wan DiT Blocks With New Projections

Use Wan-initialized transformer blocks but replace incompatible feature input/output layers.

Initial config:

```yaml
video_dit_config:
  in_dim: 384
  out_dim: 384
  patch_size: [1, 2, 2]
  action_conditioned: false
  fuse_vae_embedding_in_latents: true
```

The loader must tolerate shape mismatches for:

- `patch_embedding.*`
- `head.*`

All compatible transformer, time, text, and attention weights should load from Wan.

### 5. Add Configs

Add separate configs so the baseline cannot be confused with the VAE path:

- `configs/model/fastwam_dinov3.yaml`
- `configs/data/libero_dinov3_2cam.yaml`
- `configs/task/libero_dinov3_2cam224_1e-4.yaml`

The task config should match the aligned RGB baseline:

- `num_frames: 17`
- `action_video_freq_ratio: 2`
- video frames: `9`
- action horizon: `16`
- `batch_size: 16` for initial smoke if memory allows

### 6. Add Verification Scripts

Extend or add scripts for:

- static config validation;
- one-batch shape check;
- DINO feature statistics logging;
- one-step smoke training;
- checkpoint prediction visualization via PCA.

Required diagnostics:

- DINO feature shape;
- feature mean/std;
- video target velocity MSE;
- feature cosine similarity;
- action loss.

### 7. Smoke Training

Run a short training job before full 2k-step runs.

Exit criteria:

- no DINO preprocessing dtype/range error;
- no Wan weight loading mismatch beyond expected input/output layers;
- no NaN in feature velocity loss;
- GPU memory and throughput are measured;
- `loss_video` decreases or is at least numerically stable over the smoke window.

### 8. Evaluation

For the first DINO baseline, evaluate in feature space:

- future-only feature MSE;
- future-only feature cosine similarity;
- flow velocity MSE;
- action L1/L2, using the same action eval path as RGB/depth/seg.

Do not compare DINO feature prediction with RGB/depth/seg via PSNR/SSIM. Those metrics are pixel-space and are not comparable to DINO targets.

## Fairness Rules

The first DINOv3 experiment should differ from the aligned RGB baseline only in the world target representation.

Keep fixed:

- dataset;
- temporal horizon;
- action horizon;
- batch size target;
- training steps;
- optimizer and LR;
- action branch;
- `action_conditioned=false`;
- MoT action/video attention topology.

Later ablations can separately test feature normalization, action-conditioned feature dynamics, action attention to predicted future features, larger DINO encoders, or offline feature caching.
