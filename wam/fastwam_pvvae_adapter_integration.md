# FastWAM PV-VAE Adapter Integration Plan

## Position

This plan is for integrating the trained PV-VAE-style adapter into the FastWAM / Wan2.2 world-action model path, not into OpenPI.

OpenPI scripts were only used as lightweight utilities for:

- reading LeRobot/OXE clips;
- extracting frozen SVG-P / DINO teacher features;
- training and diagnosing the adapter.

The downstream research target is FastWAM-style video/action joint training. OpenPI is a VLA/VLM policy stack and should not be treated as the final WAM integration path.

## Correct Role Split

```text
RGB / observation clip
  -> frozen semantic teacher encoder
       SVG-P / DINO / SigLIP
  -> semantic feature clip x [B,V,17,N,D]
  -> PV-VAE adapter / tokenizer
       temporal + channel compression
  -> compressed semantic latent z [B,V,5,N,d]
  -> FastWAM / Wan2.2 video DiT target
       video-action joint denoising / world-action training
```

The semantic encoder is SVG-P / DINO / SigLIP. PV-VAE is an adapter/tokenizer over those semantic features.

The PV-VAE decoder is mainly for adapter pretraining and visualization. It is not assumed to be part of the final FastWAM inference path.

## Why PV-VAE Fits FastWAM Future4

Current SVG-DINO-P FastWAM future4 uses:

```text
num_frames = 17
action_horizon = 16
action_video_freq_ratio = 4
video target indices = [0, 4, 8, 12, 16]
```

PV-VAE temporal grouping uses:

```text
group 0: x_0
group 1: x_1  ... x_4
group 2: x_5  ... x_8
group 3: x_9  ... x_12
group 4: x_13 ... x_16
```

So PV-VAE naturally produces 5 latent groups:

```text
z_0, z_1, z_2, z_3, z_4
```

This matches the 1 condition + 4 future target structure of the existing SVG future4 FastWAM experiments, but the target is now compressed semantic latent rather than raw SVG-P feature.

## Proposed FastWAM Contract

The adapter-backed feature encoder should expose the same interface as existing FastWAM feature encoders, but return compressed latent groups:

```text
input video:
  [B,V,17,C,H,W]

frozen teacher features:
  [B,V,17,N,D]

PV-VAE latent:
  [B,V,5,N,d]

FastWAM target format:
  [B,d,5,H_feat,W_feat_total]
```

For two camera views with SVG-P `16x16` tokens:

```text
N = 16 * 16
V = 2
[B,V,5,N,d]
  -> [B,d,5,16,32]
```

For one view:

```text
[B,1,5,16*16,d]
  -> [B,d,5,16,16]
```

The FastWAM video DiT config should set:

```yaml
video_dit_config:
  in_dim: d
  out_dim: d
```

For current PV-VAE runs, likely first value:

```yaml
d: 128
```

## First Implementation Target

Add a new FastWAM feature encoder wrapper, conceptually:

```python
PVVAEAdaptedFeatureEncoder(
    base_encoder=SVGFeatureEncoder(...),
    pvvae_checkpoint=...,
    feature_stats=...,
    temporal_compression=4,
)
```

Expected behavior:

1. Receive raw RGB clip from FastWAM dataset.
2. Encode all 17 frames with frozen SVG-P / DINO.
3. Apply the same feature normalization used in PV-VAE training.
4. Run PV-VAE encoder only.
5. Return compressed latent groups in FastWAM feature-target layout.
6. Freeze both teacher encoder and PV-VAE adapter for the first downstream shape / loss smoke.

Do not treat this as the first expensive full FastWAM training run. Joint tuning is a later ablation, and full-scale FastWAM training should only happen after the adapter passes standalone quality probes.

The first target is a cheap integration proof:

```text
Can FastWAM's video branch consume PV-VAE-compressed semantic targets
with the expected shape, finite loss, and unchanged action path?
```

This is different from claiming that the final model architecture must be
`history + action -> PV-VAE latent -> PV-VAE decoder`. The PV-VAE decoder remains a diagnostic tool unless a later paper-grounded design explicitly uses it.

## Data Requirement

The existing FastWAM future4 data path may only pass subsampled video frames `[0,4,8,12,16]` into the model. PV-VAE needs all 17 frames to produce groups:

```text
x_0 ... x_16 -> z_0 ... z_4
```

Therefore the first integration must verify one of these two routes:

### Preferred Route

Configure the dataset/model path so the feature encoder receives full 17-frame video:

```text
raw clip [B,V,17,C,H,W]
  -> PV-VAE feature encoder returns [B,d,5,H,W]
```

The model then trains video DiT over 5 latent groups.

### Fallback Route

If the dataset currently cannot pass full 17 frames into the feature encoder, add a separate field for full-resolution feature-target clips, instead of abusing `action_video_freq_ratio`.

Avoid training PV-VAE target from only `[0,4,8,12,16]`; that no longer matches the adapter pretraining contract.

## Training Loss in FastWAM

The FastWAM video loss should be computed on compressed semantic latent groups:

```text
L_video = MSE / flow target loss over z groups
```

Not on decoded RGB.

Decoded RGB is only for visualization:

```text
pred z
  -> PV-VAE decoder
  -> reconstructed SVG-P/DINO feature clip
  -> SVG decoder / PCA visualization
```

## First Baselines

Use matched FastWAM settings:

1. Raw SVG-DINO-P target:
   - existing `libero_svg_dino_p_2cam256_future4_1e-4`

2. PV-VAE-compressed SVG-DINO-P target:
   - same action horizon
   - same cameras
   - same data
   - same optimizer
   - same Wan/ActionDiT backbone
   - only target representation changes from `D=384` raw feature to `d=128` compressed latent

Optional later:

3. No-world-loss baseline.
4. Adapter joint-finetune ablation.
5. Adapter with decoder reconstruction regularizer during WAM training.

## First Smoke Checks

Before long training, run a shape-only check:

```text
batch video: [B,V,17,C,H,W]
teacher x:   [B,V,17,N,D]
adapter z:   [B,V,5,N,d]
target:      [B,d,5,H_feat,W_feat_total]
```

Then run a short FastWAM smoke:

```text
20-100 training steps
```

Accept only if:

- no shape mismatch;
- no NaN;
- video loss finite;
- action loss still computed;
- eval can save PCA visualization of predicted/GT latent;
- optional SVG decode path works for qualitative check.

This smoke is not meant to prove final control improvement. It only checks that the adapter can be used as a world-target representation in the FastWAM code path.

## LARY-Style Tokenizer Quality Probe

Before committing large compute to FastWAM, add a LARY-style action probe over frozen representations. The point is to test whether the compressed representation preserves action-relevant transition information, not just reconstruction quality.

Suggested probe contract:

```text
input:
  z_t, z_{t+1:t+k}, optional instruction/state

probe target:
  action chunk a_{t:t+k-1}

metrics:
  action L1 / L2
  direction / gripper accuracy when applicable
  comparison against raw SVG-P / DINO feature probe
  comparison against per-frame S-VAE and PV-VAE latents
```

Use the same frozen dataset splits and the same lightweight probe architecture across targets. If PV-VAE compression badly hurts action prediction compared with raw SVG-P / DINO, then it is not ready as a FastWAM target even if decoded visualization looks acceptable.

This probe should be run before a full FastWAM training job. It is cheaper, closer to the representation question in `research_refine.md`, and aligns the tokenizer evaluation with action relevance.

## Decision Criteria

The adapter is useful only if the downstream FastWAM comparison improves or preserves action-side metrics under lower-dimensional world targets:

- action L1/L2;
- rollout success if available;
- feature/latent prediction metrics;
- speed/memory improvement from `D=384` to `d=128`;
- robustness or shift behavior.

Do not decide based on PV-VAE decoded RGB alone.

## Immediate Next Coding Tasks on FastWAM

0. Add / run the LARY-style probe on frozen features and PV-VAE latents.

1. Locate existing feature encoder registry:

```text
src/fastwam/models/vision/feature_encoders.py
```

2. Add a PV-VAE adapted encoder implementation:

```text
src/fastwam/models/vision/pvvae_adapter.py
```

3. Register config name, e.g.

```yaml
feature_encoder_config:
  name: svg_pvvae
  base_name: svg
  pvvae_checkpoint: ...
  feature_stats: ...
  latent_dim: 128
  temporal_compression: 4
```

4. Add model config:

```text
configs/model/fastwam_svg_pvvae.yaml
```

5. Add task config:

```text
configs/task/libero_svg_pvvae_2cam256_future4_1e-4.yaml
```

6. Add verification script:

```text
scripts/verify_pvvae_feature_encoder.py
```

7. Run smoke before long training.

## Current Local Blocker

The local `E:/awesome_wam/external/FastWAM` checkout has missing core files under `src/fastwam/models/...` and a corrupted `.git` object store. The FastWAM integration code should be edited in the valid server-side FastWAM checkout, or after refreshing the local FastWAM tree.
