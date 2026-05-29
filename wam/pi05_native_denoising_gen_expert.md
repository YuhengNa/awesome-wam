# Pi0.5-Native Denoising Generation Expert

Date: 2026-05-16

## Idea Summary

This idea adds a training-time future representation objective to a pi0.5-style VLA without requiring test-time future imagination. The model keeps pi0.5's policy-first behavior: at deployment it consumes current observations and language, then outputs an action chunk. During training, a third generation expert predicts future visual representations, forcing the shared MoT backbone to learn dynamics-aware and interaction-aware features.

## Core Claim

Policy-first VLAs such as pi0.5 are mainly supervised by action flow matching. They can learn strong action policies, but they do not receive a dense future-state constraint. A native denoising generation expert can add this constraint while preserving the original pi0.5 inference path. If the auxiliary future loss improves action success or robustness, the contribution is not "imagine at test time", but "learn better policy representations from future prediction at train time".

## Architecture

Use a three-expert Mixture-of-Transformer layout:

```text
Expert 0: VLM / understanding expert   # PaliGemma/SigLIP, pretrained
Expert 1: action expert                # pi0.5 action expert, load trained weights
Expert 2: generation expert            # Gemma-300M-style, random init
```

Keep the expert order as `[vlm, action, gen]` so existing pi0.5 action-expert parameter names remain stable. The generation expert should match the action expert's transformer shape, e.g. the OpenPI `gemma_300m`-style width/depth/head configuration, so it can participate in the same layer-wise MoT attention.

## Initialization Decision

The main experiment uses random initialization for the generation expert:

```yaml
gen_expert_init: random
gen_expert_checkpoint: null
use_adarms: [false, true, true]
```

Do not copy the action expert into the generation expert for the mainline result. Copying action weights would mix in action-specific priors and make the ablation less clean. It can be kept as a later ablation:

```yaml
gen_expert_init: copy_action_expert
```

We also do not use Gemma-3-270M or EmbeddingGemma-300M as strict initialization, because their shapes do not match pi0.5's `gemma_300m` expert configuration. If an exact matching generic Gemma-300M checkpoint is found later, it can be added as an ablation, not the default.

## Future Target

The preferred target is the native pi0.5 visual representation, but the target encoder must be stable. The first version should use a frozen teacher image encoder for future frames:

```text
future RGB frames
  -> frozen pi0.5/PaliGemma SigLIP image encoder / projector
  -> future visual tokens
  -> stop-gradient target
```

The online current-observation path remains trainable:

```text
current RGB frames
  -> trainable online pi0.5 SigLIP/VLM prefix
  -> action expert and gen expert condition on this prefix
```

This keeps the auxiliary world objective aligned with the VLA's own observation space while avoiding a moving target. Do not use the same trainable SigLIP instance as both the online encoder and the future target encoder in the main experiment; otherwise the generation loss can partly optimize the target space itself instead of learning prediction. If visualization is needed, start with PCA over predicted and target tokens. A learned decoder or external SigLIP2/Scale-RAE decoder can be tested later, but should not be required for the first control experiment.

## Denoising Objective

Train the generation expert with rectified-flow style denoising over future visual tokens:

```text
target = normalize(future_visual_tokens)
noise  = randn_like(target)
t      ~ action-flow timestep distribution
x_t    = t * noise + (1 - t) * target
u_t    = noise - target

gen expert predicts u_hat_t from x_t, current context, and timestep t
L_gen = mse(u_hat_t, u_t)
L_total = L_action + lambda_gen * L_gen
```

Recommended first setting:

```yaml
lambda_gen: 0.01
lambda_gen_warmup_steps: 1000
future_frames: 4
future_target: pi05_native_visual_tokens
future_target_encoder: frozen_pi05_siglip
future_target_stop_gradient: true
online_prefix_encoder_trainable: true
```

Use the same timestep-conditioning mechanism as the action expert where possible. The target normalization should be explicit and logged, because feature scale can dominate the loss.

## Attention Mask

The action path must not attend to future generation tokens. Otherwise training would leak future information that is unavailable at inference. Use a sibling-branch mask:

```text
query \ key   prefix/current   action   gen
prefix/current       1           0       0
action               1           1       0
gen                  1           0       1
```

This is the key difference from a test-time imagination model. The generation branch is a training-only auxiliary branch. The action branch can be evaluated exactly like the pi0.5 baseline.

## Positioning

This idea is closer to FastWAM's training-only future supervision than to imagine-then-act planning. The difference is that the implementation is pi0.5-native:

- it uses pi0.5's own visual tokens as the future target;
- it preserves the pi0.5 action inference path;
- it adds a third MoT expert instead of attaching an external DiT head;
- the generation expert is randomly initialized, matching the clean initialization style used for new expert branches.

## First Validation Plan

Target benchmark: **LIBERO**. The first implementation should compare against the OpenPI `pi05_libero` action-only baseline, using the same official `physical-intelligence/libero` dataset, action normalization, two-camera observation convention, and `action_horizon=10`.

1. Implement the three-expert config and verify that loading the original pi0.5 action checkpoint is unchanged.
2. Add LIBERO future RGB loading and native visual-token target extraction with a frozen pi0.5/PaliGemma SigLIP teacher and `stop_gradient`.
3. Implement the sibling attention mask so action tokens cannot see generation tokens.
4. Train a short smoke run with action loss plus a warmed-up, small `lambda_gen`.
5. Compare against `pi05_libero` on action loss, LIBERO rollout success, and robustness.
6. Add PCA visualization of predicted versus target future tokens for debugging.

## Implementation Status

Initial implementation started in `external/openpi` using OpenPI's PyTorch pi0.5 path. `external/InternVLA-A1` is also checked out as the three-expert MoT reference.

Implemented:

- optional `enable_gen_expert` config fields in `Pi0Config`;
- `PaliGemmaWithExpertModel` support for `[VLM, action, gen]` experts while preserving the existing `gemma_expert` action-branch parameter names;
- random-init `gen_expert` with `use_adarms=[false, true, true]`;
- sibling branch attention mask `[prefix/current, action, gen]`, where action cannot attend to gen tokens;
- `forward_with_future_token_targets(...)` for action loss plus future-token rectified-flow loss;
- `encode_future_images_with_teacher(...)` for generic `[B,T,C,H,W]` or `[B,V,T,C,H,W]` future RGB encoding using a frozen teacher vision tower;
- relaxed PyTorch checkpoint loading when `enable_gen_expert=true`, so old pi0.5 weights can load while new gen-expert keys stay randomly initialized.
- LIBERO future-frame data plumbing for the PyTorch path:
  - `LeRobotLiberoDataConfig.future_image_delta_indices`;
  - current/future frame splitting for `image` and `wrist_image`;
  - `future_images` resize and batch return from the data loader;
  - trainer branch that online-encodes future RGB targets and logs `action_loss`, `gen_loss`, and `gen_loss_weight`.
- `pi05_libero_gen` config:
  - dataset: `physical-intelligence/libero`;
  - action horizon: `10`;
  - future visual targets: 4 frames at dataset offsets `(1, 3, 6, 9)`;
  - cameras: `base_0_rgb` and `left_wrist_0_rgb`;
  - `lambda_gen=0.01`, warmup `1000` steps.

Verified so far:

```bash
python -m py_compile \
  external/openpi/src/openpi/policies/libero_policy.py \
  external/openpi/src/openpi/transforms.py \
  external/openpi/src/openpi/training/config.py \
  external/openpi/src/openpi/training/data_loader.py \
  external/openpi/src/openpi/models_pytorch/gemma_pytorch.py \
  external/openpi/src/openpi/models_pytorch/pi0_pytorch.py \
  external/openpi/src/openpi/models/pi0_config.py \
  external/openpi/scripts/train_pytorch.py
```

Not implemented yet:

- real checkpoint load / model instantiation smoke in an OpenPI environment with full dependencies;
- PCA visualization for predicted/target future tokens.
- LIBERO rollout eval comparison between `pi05_libero` and `pi05_libero_gen`;
- ablation on future-frame offsets and `lambda_gen`.

## Required Ablations

| Variant | Purpose |
| --- | --- |
| pi0.5 baseline | action-only reference |
| random-init gen expert | main method |
| no action sees gen | confirms no future leakage |
| gen attends action vs not | tests whether action-conditioned future target matters |
| deterministic MSE head | checks whether denoising is necessary |
| trainable target encoder | moving-target ablation, not mainline |
| copy-action init | optional ablation, not mainline |

## Risks

- The future-token count can be large. Start with 4 future frames or pooled visual tokens if memory is tight.
- Native pi0.5 visual tokens may not have a good RGB decoder, so early debugging should rely on feature loss and PCA.
- If the future target encoder is trainable, the target distribution can drift. The first version should keep a frozen teacher encoder for target extraction.
- If `lambda_gen` is too high, the auxiliary objective may hurt action learning. Warmup and loss-scale logging are required.
- A generic Gemma-300M checkpoint should not be used unless its architecture exactly matches the pi0.5 expert.
