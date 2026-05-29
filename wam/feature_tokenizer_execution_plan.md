# Feature Tokenizer Execution Plan

This document turns the current tokenizer discussion into an executable plan.
It should be read together with `tokenizer_methods.md`, `current_fastwam_experiments.md`,
and `research_refine.md`.

## Goal

Build and evaluate visual feature tokenizers for robot world models. The immediate
target is not full FastWAM replacement, but a reliable tokenizer layer that can
compress teacher visual features while preserving information useful for action
prediction and future-state modeling.

The current research question is:

> Should the world model latent space be optimized for reconstruction, semantics,
> temporal prediction, or action-relevant transitions?

## Current Status

Three tokenizer directions are active:

1. Per-frame S-VAE / channel adapter
   - Contract: `[B,V,T,N,D] -> [B,V,T,N,d]`.
   - Compresses each frame independently.
   - Main purpose: baseline for semantic feature compression.
   - Current gap: no standalone model/training script existed before this pass.

2. PV-VAE-style temporal predictive feature VAE
   - Contract: `[B,V,1+T,N,D] -> [B,V,1+T/4,N,d]`.
   - Compresses channels and temporal groups.
   - Already implemented in `predictive_feature_vae.py` and
     `train_predictive_feature_vae_libero.py`.
   - Main risk: the model may learn static-copy reconstruction instead of useful
     predictive dynamics.

3. DeltaTok / transition tokenizer
   - Contract target: `x_t, x_{t+k} -> z_delta [B,V,M,d]`.
   - Current implementation is a deterministic global transition token:
     `x_t, x_{t+k} -> z [B,d]`.
   - Already implemented in `delta_tokenizer.py` and `train_deltatok_libero.py`.
   - Main purpose: learn action-salient state changes instead of full scene
     reconstruction.

## Data Format

Training scripts should consume batches from the OpenPI LIBERO data path and
convert image observations into teacher feature tensors.

Image tensors:

```text
current_images: [B,V,C,H,W]
future_images:  [B,V,T,C,H,W]
image_clip:     [B,V,1+T,C,H,W]
```

Teacher feature tensors:

```text
features:       [B,V,F,N,D]
B: batch size
V: camera views
F: frame count, usually 1 + number of future deltas
N: spatial tokens per view, for example 16 * 16
D: teacher feature width, for example 384 or 1024
d: compressed latent/token width
```

The first usable data target is LIBERO because existing OpenPI scripts already
provide dataloaders, camera handling, and teacher feature extraction. Larger
datasets such as DROID, OXE, RoboCasa, and Behavior1K should be added after the
tokenizer contracts and evaluation scripts are stable.

## Evaluation

Tokenizer evaluation should avoid relying on reconstruction loss alone.

Core metrics:

```text
recon_mse: feature reconstruction error
cosine_loss: semantic direction preservation
future_mse: future-frame reconstruction or prediction error
static_future_mse: error from copying current features
delta_ratio: model future error divided by static-copy error
kl_loss: VAE regularization when enabled
```

Action-relevance metrics:

```text
LARY-style action probe:
  freeze tokenizer latents/features
  train a small action head to predict robot actions
  compare action loss/success proxy across tokenizer variants

FastWAM-style downstream check:
  plug tokenizer output into the world/action branch
  compare action decoding, temporal prediction, and rollout quality
```

The rule of thumb:

```text
good tokenizer != lowest reconstruction loss
good tokenizer = compact latent + semantic preservation + action-predictive signal
```

## Immediate Implementation Steps

1. Add a standalone Per-frame S-VAE model.
   - Input: `[B,V,N,D]`.
   - Output: reconstructed features `[B,V,N,D]` and latent tokens `[B,V,N,d]`.
   - Train it over clips by flattening the frame axis: `[B,V,F,N,D] -> [B*F,V,N,D]`.

2. Add a LIBERO training script for S-VAE.
   - Reuse existing OpenPI LIBERO dataloader utilities.
   - Reuse SVG-P and DINO feature extraction utilities.
   - Log reconstruction, cosine, KL, latent norm, and target norm.

3. Run a small smoke experiment.
   - Teacher: `svg_p` first, because current scripts already support SVG-P
     visualization.
   - Views: `base_0_rgb`.
   - Future deltas: `1,3,6,9`.
   - Steps: 100 to 500 for sanity.

4. Compare against existing PV-VAE and DeltaTok runs.
   - S-VAE answers: can we compress semantics frame-by-frame?
   - PV-VAE answers: does temporal grouping help?
   - DeltaTok answers: is transition-only information more action-relevant?

## Suggested Smoke Commands

Run these on the machine that has OpenPI dependencies and data access:

```bash
python external/openpi/scripts/train_svae_libero.py \
  --output-dir runs/svae/smoke_svgp \
  --teacher svg_p \
  --views base_0_rgb \
  --future-deltas 1,3,6,9 \
  --batch-size 8 \
  --max-steps 100 \
  --latent-dim 96 \
  --model-dim 384
```

When using `--teacher svg_p`, also provide the local SVG-P paths used by the
existing LAM/PV-VAE scripts:

```bash
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

If those paths are not ready, run the first smoke with `--teacher dinov3_vits16
--decode-svg-rgb false`.

Then compare with:

```bash
python external/openpi/scripts/train_predictive_feature_vae_libero.py ...
python external/openpi/scripts/train_deltatok_libero.py ...
```

## Decision Criteria

Continue a tokenizer direction only if it satisfies at least one of these:

1. It beats static-copy baselines on temporal metrics.
2. It preserves teacher semantic structure under high compression.
3. It improves frozen-latent action probing.
4. It plugs into FastWAM/OpenPI without awkward shape conversions.

If a method only gives pretty reconstructions but weak action probing, it should
be treated as an auxiliary baseline rather than the main research line.
