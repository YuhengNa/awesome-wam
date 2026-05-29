# Research Refine: Action-Aligned World Representations

Date: 2026-05-13

This document refines the current WAM project after the novelty check. The project should be framed as a representation study in the spirit of `Reconstruction or Semantics?` and LARY, not as a proposal to replace VAE latents with DINO features.

## Problem Anchor

- **Bottom-line problem:** FastWAM shows world/action co-training can improve control, but it is unclear which predicted world representation actually provides action-relevant supervision, and whether the answer changes across FastWAM inference regimes.
- **Must-solve bottleneck:** Visual fidelity metrics can reward details that are irrelevant or harmful for control. We need to identify representations whose prediction quality aligns with downstream action success and robustness.
- **Non-goals:** Do not propose a new large WAM architecture, new foundation encoder, object-slot model, memory system, or unrelated test-time planning pipeline in the first paper.
- **Constraints:** Reuse the FastWAM codebase, LIBERO data, current 10 fps aligned setting, fixed action horizon, fixed action branch, and feasible 4-GPU H100 training jobs.
- **Success condition:** A matched evaluation shows which representation family improves action success or robustness, and whether world prediction metrics correlate with control quality.

## Refined Research Question

When a WAM learns from future world prediction, **which world representation should it predict, and does the answer depend on how future/action streams are used at inference?**

More concretely:

> Are reconstruction-oriented, structure-oriented, or semantic foundation-model representations most aligned with downstream robot action performance in FastWAM-style world-action models?

## Representation Taxonomy

| Family | Targets | What It Emphasizes | Expected Failure Mode |
| --- | --- | --- | --- |
| Reconstruction pixel space | RGB through Wan VAE | Appearance and visual fidelity | May waste capacity on texture, lighting, and background |
| Structural pixel space | depth, segmentation through Wan VAE | geometry, object extent, contact-relevant structure | May lose semantic identity or fine manipulation cues |
| Semantic feature space | DINOv3, SigLIP2, V-JEPA 2.1 | object/scene semantics and invariant features | May be hard to optimize or discard control-relevant geometry |
| Generative feature space | SVG Res-ViT features plus decoder | semantic feature generation with RGB decode proxy | Decoder quality may not match action relevance |

## Method Thesis

FastWAM provides a useful controlled testbed for action-aligned world representations: by fixing the training protocol and changing only the future world target, we can measure which representation family transfers useful predictive structure into the action system, and whether that transfer is specific to decoupled/action-only inference or generalizes to joint/causal variants.

This is intentionally a **study thesis**, not a new architecture thesis.

## Method Candidate A: Interaction-Saliency World Loss

See [action_salient_world_modeling.md](action_salient_world_modeling.md) for the concrete method idea.

The candidate method keeps the representation study intact but adds a small action-aware training intervention. Instead of applying uniform world loss to every future token, train a transition IDM on GT latent pairs:

```text
IDM(z_{i-1}, z_i) -> action segment
```

Then freeze the IDM and use its cross-attention or gradient saliency to build per-frame action-saliency maps:

```text
w_i: [B, H, W]
```

During WAM training, reweight the per-token world loss:

```text
L_world = mean_i mean_HW(w_i * loss_token(z_hat_i, z_i))
```

This should be compared against uniform loss, optical-flow or feature-difference motion weighting, random weighting, and IDM-gradient saliency. The novelty claim is not "using IDM"; it is using inverse-dynamics- or interaction-derived token saliency to make world prediction interaction-salient rather than uniformly reconstructive.

## Method Candidate B: Interaction-Aware Latent Action

See [interaction_aware_latent_action.md](interaction_aware_latent_action.md) for the concrete method idea.

This candidate changes the saliency source rather than the WAM loss itself. It learns latent action tokens from visual transitions and instructions:

```text
z_{i-1}, z_i, instruction -> latent interaction tokens
```

The tokens are trained with richer targets than low-level action alone:

```text
low-level action + target object + contact + task progress
```

The goal is to avoid the plain-IDM shortcut where attention collapses onto robot motion. If successful, these interaction-aware latent tokens can provide better saliency maps for Candidate A or become an auxiliary WAM prediction target.

## Contribution Focus

- **Dominant contribution:** A controlled empirical study of action-aligned world prediction targets for FastWAM-style world-action models.
- **Supporting contribution:** A unified implementation and evaluation protocol for pixel, structural, and foundation-feature targets in the same WAM codebase.
- **Possible method contribution A:** Interaction-Saliency World Loss, if IDM/interaction-saliency weighting improves action metrics or robustness beyond uniform and motion-only world losses.
- **Possible method contribution B:** Interaction-Aware Latent Action, if richer action/object/contact/progress supervision produces saliency maps and action metrics beyond plain IDM and motion baselines.
- **Explicit non-contributions:** We do not claim DINO, SigLIP, V-JEPA, SVG, depth, or segmentation targets are new. We do not claim a new visual encoder or a new WAM backbone.

## Preliminary Paper Score

**Current score: 7/10, proceed with caution.**

This is a promising paper direction if the experiments reveal a clear and defensible relationship between representation targets and action performance. It is not yet a strong standalone method paper, because the mechanism is intentionally minimal and several close works already study semantic/reconstruction representations.

| Axis | Score | Rationale |
| --- | --- | --- |
| Problem importance | 8/10 | WAMs need to know what future representation is worth predicting and whether this depends on how future/action streams are used at inference. |
| Novelty | 7/10 | The broad representation question is already close to `Reconstruction or Semantics?`, LARY, LDA-1B, and DexWorldModel. The distinct angle is target representation under FastWAM's decoupled, joint, and causal/IDM regimes. |
| Technical elegance | 8/10 | The study is clean because it changes only the target representation while keeping the action path fixed. |
| Experimental risk | 6/10 | If all targets perform similarly, or if action eval is noisy, the story becomes weak. |
| Paper potential | 7/10 | Strong as an empirical/analysis paper if paired with robust action eval, correlation analysis, and shift evaluation. |

**Path to 8+/10:** show a non-obvious result, such as visual metrics failing to predict action success, structural targets outperforming semantic features under shift, or SVG/DINO improving robustness despite worse decoded visual quality.

**Path to 5/10:** only report train/eval losses or feature metrics without convincing online action success and robustness evidence.

## Technical Gap

Related work already argues that semantic latents can be useful for robotic world models and vision-to-action alignment. The remaining gap is narrower:

1. Existing studies do not isolate FastWAM's future world target across its decoupled, joint, and causal/IDM regimes.
2. Reconstruction, structural, and semantic targets are rarely compared under one matched action-learning protocol.
3. It remains unclear whether better world prediction metrics imply better control, and whether that relationship changes when future/action streams are decoupled or coupled.

## Minimal Mechanism

Keep the mechanism deliberately small:

1. Use the same FastWAM action branch, language/state inputs, optimizer, dataset, and action horizon.
2. Swap only the future world prediction target.
3. Evaluate target-specific world metrics and shared action metrics.
4. Add correlation analysis between world metrics and action outcomes.
5. Treat inference regime as a study axis after the decoupled baseline is validated: decoupled/action-only, joint, and causal/IDM.

This avoids contribution sprawl and keeps the study interpretable.

## Hypotheses

- **H1: Representation matters.** World target choice changes action success under matched training and evaluation protocols.
- **H2: Visual fidelity is not sufficient.** RGB/depth/seg reconstruction metrics will not fully predict downstream action success.
- **H3: Structural targets improve robustness.** Depth and segmentation should help more under visual perturbations or LIBERO-Plus scene shifts than under clean in-distribution eval.
- **H4: Inference regime may change the best target.** A target that is good as auxiliary supervision in a decoupled/action-only setting may not be best when future/action streams are coupled at inference.

## Proposed Paper Story

1. FastWAM provides multiple ways to use future prediction: decoupled auxiliary supervision, joint future/action generation, and causal/IDM-style coupling.
2. Therefore, the representation being predicted is a key design choice, not only the inference algorithm.
3. We compare reconstruction, structural, and semantic world targets under a fixed FastWAM protocol.
4. We show which world targets are most action-aligned, when world prediction metrics predict control, and whether the answer depends on inference regime.

## Minimal Validation Blocks

### Block 1: Matched Representation Sweep

- **Claim tested:** World target representation affects downstream action performance.
- **Variants:** no-world-loss, RGB, depth, segmentation, DINOv3, SigLIP2, V-JEPA 2.1, SVG Res-ViT.
- **Metrics:** LIBERO success, action L1/L2, training/eval loss, target-specific world metrics.
- **Success criterion:** At least one representation family consistently improves over no-world-loss and RGB under matched settings.

### Block 2: Metric Alignment Analysis

- **Claim tested:** World prediction quality and action quality are not always equivalent.
- **Analysis:** Correlate world metrics with action error and rollout success across targets/checkpoints.
- **Metrics:** PSNR/SSIM for pixel targets, feature MSE/cosine for feature targets, decoded proxy metrics for SVG, action success/error for all.
- **Success criterion:** Identify which metrics are predictive of control and which are misleading.

### Block 3: Robustness / Shift Evaluation

- **Claim tested:** Structural or semantic targets improve robustness beyond clean LIBERO.
- **Settings:** LIBERO-Plus scenes or controlled visual perturbations.
- **Metrics:** success drop from clean eval, action error shift, qualitative rollout failure types.
- **Success criterion:** At least one non-RGB representation reduces performance degradation under shift.

### Block 4: Inference-Regime Interaction

- **Claim tested:** The preferred world representation depends on how FastWAM couples future and action at inference.
- **Variants:** decoupled/action-only, joint, and causal/IDM for a small subset of targets, e.g. RGB, segmentation, DINO/SigLIP, and SVG.
- **Metrics:** LIBERO success, latency, action error, future target metric where applicable.
- **Success criterion:** Either the ranking is stable across regimes, which strengthens the representation conclusion, or the ranking changes, which becomes a distinct and useful finding.

## What To Avoid

- Do not call the method "DINO-FastWAM" as the main story.
- Do not make claims from feature MSE alone.
- Do not compare DINO/SigLIP/V-JEPA to RGB using PSNR/SSIM.
- Do not mix inference regimes in the main representation sweep; if topology changes, report it as the inference-regime interaction study.
- Do not mix 10 fps and 20 fps timing without explicitly normalizing the temporal span.

## Immediate Next Step

Move from `research-refine` to `experiment-plan`:

1. Freeze a run manifest for all representation variants.
2. Define exact checkpoint selection rules and eval commands.
3. Add the no-world-loss baseline.
4. Prioritize clean LIBERO single-episode sanity eval before full evaluation.
5. Build the first action/representation alignment table in the decoupled setting.
6. After that table is stable, add a small joint/IDM subset to test inference-regime interaction.
