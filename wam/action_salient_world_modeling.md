# Interaction-Saliency World Loss

Date: 2026-05-14

## Idea Summary

Robot world models should not optimize future prediction uniformly over all visual tokens. Most tokens describe background, texture, lighting, or static scene content; only a subset is interaction-relevant, such as gripper, target object, contact, and controllable motion regions. This method uses an IDM or interaction probe to estimate token-level saliency and reweights the world-model loss toward these tokens.

## Core Claim

Uniform world loss can produce visually or semantically plausible futures while under-weighting the regions that matter for interaction. A learned saliency map can identify which future tokens are useful for recovering the action and interaction that caused the transition. Training the WAM with this saliency-weighted world loss should improve action prediction and control success, even when average feature MSE is similar.

## Architecture

The method adds a training-only saliency probe. The FastWAM backbone and inference path remain unchanged.

```text
GT video frames -> frozen encoder E -> GT latents z_0:K
                                     -> train saliency probe
                                     -> freeze saliency probe

During WAM training:
obs -> FastWAM -> predicted future latents z_hat_1:K
GT latents z_0:K -> frozen probe -> interaction-saliency maps w_1:K
weighted world loss: w_i * loss(z_hat_i, z_i)
```

The probe is not used during deployment. It only supplies loss weights during training.

## Transition Probe

Use a shared two-frame transition probe rather than a full-sequence probe for the first version:

```text
Probe(z_{i-1}, z_i, instruction) -> action / interaction targets between the two frames
```

For the current SVG future4 setting:

```text
z0 -> z1 : actions 0:3
z1 -> z2 : actions 4:7
z2 -> z3 : actions 8:11
z3 -> z4 : actions 12:15
```

For future8 with `action_video_freq_ratio=2`:

```text
z0 -> z1 : actions 0:1
z1 -> z2 : actions 2:3
...
z7 -> z8 : actions 14:15
```

This gives a clear saliency map for each predicted future frame. A full-sequence probe can be added later as an evaluation probe, but it is less suitable for per-frame loss weighting because its attention can blur across time.

## Minimal Probe Module

Minimal attention-probe design:

```text
input: z_prev, z_next, instruction     # [B, C, H, W] plus text/context
flatten two-frame visual tokens        # [B, 2*H*W, C]
linear projection C -> d
add frame and spatial position embeddings
r learned action query tokens          # r = action_video_freq_ratio
cross-attend action queries to visual tokens
MLP head -> [B, r, action_dim]
```

The simplest probe is trained on GT encoded features with action regression:

```text
L_IDM = || IDM(z_{i-1}, z_i) - a_{segment} ||
```

For a stronger interaction-aware version, replace or extend this probe with the latent-action model described in [interaction_aware_latent_action.md](interaction_aware_latent_action.md). After the probe has reasonable prediction accuracy and interpretable saliency, freeze it.

## Saliency Map Construction

Each transition call produces attention:

```text
attn_i: [B, heads, action_queries, 2*H*W]
```

For the weight of future frame `z_i`:

```text
attn_next = attn_i[..., H*W : 2*H*W]
w_i = mean(attn_next over heads and action_queries)
w_i -> reshape [B, H, W]
```

Normalize and clamp to keep the loss stable:

```text
w_i = w_i / mean(w_i)
w_i = clamp(w_i, 0.5, 3.0)
```

Optionally apply light Gaussian blur or EMA smoothing. The final weights have shape:

```text
w: [B, T_future, H, W]
```

and broadcast across feature channels.

## WAM Loss

For feature targets, compute token loss per future frame:

```text
mse_i = mean_C((z_hat_i - z_i)^2)      # [B, H, W]
L_world_as = mean_i mean_HW(w_i * mse_i)
```

For flow-matching training, apply the same weighting to the per-token flow target loss instead of sampled feature MSE:

```text
L_world_as = mean_i mean_HW(w_i * ||v_pred_i - v_target_i||^2)
```

Final training objective:

```text
L_total = L_action + lambda_world * L_world_as
```

The method does not require adding an IDM action loss on predicted futures. The key intervention is interaction-aware world-loss reweighting.

## Required Baselines

The strongest simple baseline is motion-only weighting, because optical flow or feature difference can also identify dynamic regions. It must be included.

| Variant | Weight Source | Purpose |
| --- | --- | --- |
| Uniform | all tokens equal | Original WAM loss |
| Feature difference | `||z_i - z_{i-1}||` | Dynamic-region baseline |
| Optical flow | RGB flow magnitude | Motion-region baseline |
| IDM attention | transition-IDM cross-attention | Minimal saliency method |
| Interaction latent attention | latent-action probe attention | Stronger saliency method |
| IDM gradient | `||d IDM_action / d z_i||` | Stronger saliency check |
| Random weights | shuffled or random map | Control for weighting effects |
| Motion + interaction | weighted combination | Test whether interaction saliency complements motion |

The central distinction is:

```text
motion saliency: where things move
interaction saliency: where changes are useful for task-conditioned control
```

## Novelty Positioning

Do not claim novelty as "using IDM" or "using future features for action"; those are close to prior work such as LDA-1B, DexWorldModel, MWM, EVA, and Reconstruction or Semantics?

Claim the narrower mechanism:

> Use inverse-dynamics-derived token saliency to reweight world-model training, turning uniform future prediction into action-salient future prediction.

Closest overlaps:

- `Reconstruction or Semantics?` and LARY use action recoverability as evaluation; here it becomes a training signal.
- EVA uses inverse-dynamics reward for executable video alignment; here the signal is token-level supervised loss weighting.
- MWM predicts masks to focus on task-relevant structure; here the weighting is representation-agnostic and learned from action recovery.
- REPA-style methods align generative models to semantic teachers; here the teacher is action sensitivity, not visual semantics.

Current novelty estimate: medium. The idea becomes stronger if interaction saliency beats motion-only weighting and improves online control without necessarily improving average feature MSE.

## First Validation Plan

1. Train a transition IDM or interaction probe on GT SVG-DINO features.
2. Verify action or interaction prediction for each transition segment.
3. Visualize attention and gradient saliency maps; check whether they focus on gripper, object, contact, and controllable regions.
4. Train SVG-DINO WAM with uniform loss, motion-weighted loss, and interaction-saliency-weighted loss.
5. Compare action L1/L2, LIBERO success, LIBERO-Plus robustness, feature MSE/cosine, and decoded visualization.

Expected positive result:

```text
Interaction-weighted loss has similar or slightly worse feature MSE,
but lower action error and higher success / robustness than uniform and motion-only baselines.
```

If motion weighting performs as well as interaction weighting, the method should be simplified toward controllable/dynamic-region prediction rather than interaction-salient weighting.
