# Interaction-Aware Latent Action

Date: 2026-05-15

## Idea Summary

Low-level robot actions describe how the robot moves, but they often do not explain why the motion matters: which object is targeted, whether contact happens, which task phase the transition represents, or whether the interaction advances the instruction. This idea learns an instruction-conditioned latent action from visual transitions that captures both low-level control and high-level interaction semantics.

## Core Claim

A useful robot latent action should encode task-conditioned interaction, not only robot self-motion. By training latent action tokens to predict physical actions, target/contact information, and task progress, the representation can bridge semantic visual futures and low-level control. Its attention can also be used by [interaction-saliency world loss](action_salient_world_modeling.md), but the latent action itself is the method focus here.

## Architecture

```text
z_prev, z_next, instruction
        ↓
Interaction Encoder
        ↓
latent action / interaction tokens l_i
        ↓
heads:
  physical action head      -> action segment
  target object head        -> object heatmap or mask
  contact head              -> contact probability / heatmap
  progress head             -> task progress score
```

The latent action is transition-based. For current SVG future4:

```text
l_1 = Enc(z0, z1, instruction) -> actions 0:3
l_2 = Enc(z1, z2, instruction) -> actions 4:7
l_3 = Enc(z2, z3, instruction) -> actions 8:11
l_4 = Enc(z3, z4, instruction) -> actions 12:15
```

## Why Not Plain IDM?

Plain IDM:

```text
IDM(z_{i-1}, z_i) -> action segment
```

can solve the task by attending mostly to robot arm or gripper motion. That produces a motion saliency map, not necessarily an interaction saliency map.

Interaction-aware latent action adds richer supervision:

```text
transition + instruction -> what motion, which object, where contact, what task progress
```

This encourages the latent tokens to encode target-object and contact information instead of only robot self-motion.

## Training Targets

Use the smallest available supervision first.

### 1. Physical Action

Always available from the dataset:

```text
L_action = || Head_action(l_i) - a_segment ||
```

### 2. Target Object

If segmentation masks or object IDs are available, parse the instruction to find the target object and supervise a heatmap:

```text
L_object = BCE(Head_object(l_i), target_object_mask)
```

If masks are unavailable, use GroundingDINO/SAM or LIBERO simulator metadata as pseudo-labels.

### 3. Contact / Interaction

Approximate contact with gripper-object distance, mask overlap, gripper close state plus object motion, or simulator state:

```text
L_contact = BCE(Head_contact(l_i), contact_label_or_heatmap)
```

### 4. Task Progress

Use simulator success predicates, object pose changes, or task-specific heuristics:

```text
L_progress = BCE or regression(Head_progress(l_i), progress_delta)
```

Minimal first version:

```text
L = L_action + lambda_obj * L_object
```

Add contact/progress only after object supervision is working.

## Residual Interaction Variant

To reduce collapse onto robot self-motion, first predict the easy action component from proprioception or robot-only tokens:

```text
a_base = BaseAction(proprio_{i-1:i}, robot_tokens)
a_residual = a_gt - a_base
```

Then train the visual interaction latent to predict only the residual:

```text
Head_action(l_i) -> a_residual
```

This makes the visual latent focus on object/contact-dependent action information beyond robot self-motion.

## Uses

### Use 1: Representation Probe

Evaluate whether different world representations support interaction-aware latent actions:

```text
Enc_VAE, Enc_DINO, Enc_SVG, Enc_SigLIP, Enc_VJEPA
```

Metrics:

- action L1/L2;
- target object heatmap IoU;
- contact prediction accuracy;
- task progress prediction;
- robustness under visual shift.

### Use 2: Saliency Generator

Use cross-attention from latent interaction tokens to visual tokens:

```text
attention(l_i, visual_tokens) -> w_i [H, W]
```

Then pass `w_i` to the interaction-saliency world loss.

### Use 3: Auxiliary WAM Target

Train WAM to predict both future visual latents and latent interaction tokens:

```text
WAM(obs) -> z_hat_future, l_hat_future
```

This is a later extension. The first implementation should use latent action only as a probe/saliency generator.

## Novelty Positioning

Do not claim novelty as simply "learning latent actions"; LARY and other latent-action work already cover that broad space.

Narrower claim:

> Learn instruction-conditioned interaction latent actions that jointly encode low-level action, target object, contact, and task progress, then use them to diagnose or guide robot world-model training.

Closest overlaps:

- LARY evaluates latent action representations; this proposes a task-conditioned interaction latent and uses it inside WAM training.
- Plain IDM probes recover low-level action; this adds target/contact/progress supervision to avoid robot-motion-only shortcuts.
- Action-saliency world loss uses a saliency map; this document defines a richer latent representation that can produce that saliency.

Current novelty estimate: medium. The idea becomes stronger if object/contact/progress supervision changes where attention goes and improves downstream WAM/action metrics beyond plain IDM and motion baselines.

## First Validation Plan

1. Train a plain transition IDM on SVG-DINO features.
2. Train an interaction-aware latent action model with action plus target-object supervision.
3. Compare attention maps:
   - plain IDM;
   - interaction-aware latent action;
   - feature difference / optical flow;
   - random.
4. Quantify overlap with robot masks, target object masks, and contact regions.
5. If interaction-aware attention is meaningfully different, use it as `w_i` in interaction-saliency world loss.

Expected positive result:

```text
plain IDM attends mostly to robot motion;
interaction-aware latent action attends more to target object/contact;
its saliency improves WAM action metrics more than motion-only weighting.
```
