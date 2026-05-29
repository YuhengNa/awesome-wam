# WAM Novelty Check

Date: 2026-05-13

This note manually follows the `novelty-check` workflow for the current FastWAM-based project. The intended direction is not "replace VAE with DINO" as a method claim. The project is closer to `Reconstruction or Semantics?` and LARY: a systematic study of which visual/world representations are most useful for robot action learning.

## Current Project Claim

The current implementation is not a new WAM architecture from scratch. It is a controlled representation-study branch built on FastWAM:

- Base model: FastWAM-style world/action co-training, with evaluation across its inference regimes when compute allows.
- Fixed controls: aligned LIBERO data, same action branch, same MoT topology, same action horizon, and no extra action-conditioned feature dynamics in the first comparison.
- Pixel-space targets: RGB, depth, segmentation, all encoded through the Wan VAE path.
- Feature-space targets: DINOv3, SigLIP2, V-JEPA 2.1, and SVG Res-ViT.
- Main question: which predicted world representation is most aligned with downstream action prediction and rollout robustness?

The target paper should read as an empirical representation study, not as an encoder replacement paper.

## Relationship To Reconstruction Or Semantics And LARY

The closest framing is:

- `Reconstruction or Semantics?`: asks what kind of latent space makes action-conditioned world models useful.
- `LARY`: asks how latent visual/action representations align with physical action.
- Our project: asks what kind of future world prediction target is most useful across FastWAM-style training/inference regimes, including but not limited to action-only inference.

This makes the contribution a controlled benchmark/analysis:

1. Compare pixel-space reconstruction targets, structural pixel targets, and frozen foundation-model feature targets under one WAM training recipe.
2. Measure action success and robustness, not only visual reconstruction quality.
3. Diagnose when representation quality, generation quality, and action quality agree or disagree.

## Closest Prior Work

| Work | What it covers | Overlap with us | Gap we can target |
| --- | --- | --- | --- |
| [Fast-WAM](https://arxiv.org/abs/2603.16666) | Shows video co-training is useful even when future imagination is skipped at test time. | We directly build on this framing. | It does not systematically compare RGB/depth/seg/VFM feature targets. |
| [Reconstruction or Semantics?](https://arxiv.org/abs/2605.06388) | Systematically compares reconstruction and semantic encoders for action-conditioned latent diffusion world models. | Very high overlap in motivation and study style. | We can specialize the question to FastWAM's train/inference regimes, LIBERO-style action eval, and pixel/structural/VFM target comparisons. |
| [LDA-1B](https://arxiv.org/abs/2602.12215) | Uses structured DINO latent prediction for large-scale robot dynamics, policy, and visual forecasting. | Strong overlap with DINO-latent dynamics. | Its novelty is scaling and heterogeneous data ingestion, not controlled target comparison in FastWAM. |
| [DexWorldModel](https://arxiv.org/abs/2604.16484) | Uses DINOv3 features as generative targets for robust embodied world modeling. | Strong overlap with DINOv3 feature prediction. | It adds causal memory, test-time training memory, and async inference; our controlled representation study is different but weaker architecturally. |
| [Mask World Model](https://arxiv.org/abs/2604.19683) | Predicts semantic masks instead of RGB for robust robot policy learning. | Directly overlaps with segmentation-target WAM. | We can only claim mask/seg as one target in a broader representation audit, not as a standalone new method. |
| [X-WAM](https://arxiv.org/abs/2604.26694) | Predicts multi-view RGB-D and 4D structure from video priors. | Overlaps with depth/world-structure prediction. | Its focus is RGB-D/4D synthesis and async denoising, not representation-target comparison. |
| [OA-WAM](https://arxiv.org/abs/2605.06481) | Predicts object-addressable slot states with action decoding. | Overlaps with structured world state representations. | We do not yet model object slots or addressability. |
| [UVA](https://arxiv.org/abs/2503.00200) | Joint video/action latent representation with decoupled action decoding. | Shares the joint video-action training idea. | It is not a study of target representation choices. |
| [RAE](https://arxiv.org/abs/2510.11690), [RAE T2I scaling](https://arxiv.org/abs/2601.16208), [SVG-T2I](https://arxiv.org/abs/2512.11749) | Use frozen representation encoders and decoders for diffusion/generation. | Provides tools and motivation for feature-space generation and SVG decoding. | These are mostly image/T2I generation works, not robot action-alignment studies. |
| [LARY](https://arxiv.org/abs/2604.11689) | Evaluates latent action and visual representations for vision-to-action alignment. | Very close in spirit: representation alignment with physical action. | It is not a FastWAM/WAM target study and does not vary future prediction targets under different WAM inference regimes. |

## Novelty Verdict

The broad idea "semantic feature spaces can be more action-relevant than reconstruction spaces" is **not novel enough** by itself. It is already strongly covered by Reconstruction or Semantics, LARY, LDA-1B, and DexWorldModel.

The segmentation/depth variants are also not standalone novelty. Mask World Model already covers semantic mask prediction for robot policies, and X-WAM covers RGB-D world prediction.

The defensible novelty is a narrower study question:

1. **Controlled FastWAM representation audit.** We isolate the future world target while holding the action branch, temporal horizon, data, and optimizer fixed, then test whether conclusions persist across FastWAM inference regimes.
2. **Pixel, structural, and feature representations in one protocol.** RGB/depth/seg are image-like targets; DINO/SigLIP/V-JEPA/SVG are foundation-model feature targets.
3. **Action alignment rather than visual fidelity.** The key outcome is LIBERO success/action quality and robustness, not just PSNR, SSIM, feature MSE, or decoded visual quality.
4. **FastWAM-specific question.** Does the best world target depend on whether the model uses future prediction only as training supervision or also couples future/action streams at inference?

This is best positioned as a `Reconstruction or Semantics` / LARY-style empirical study for WAMs, not as a new architecture paper.

## Recommended Claim Wording

Use:

> We present a controlled study of representation targets for FastWAM-style world-action learning. Under fixed data, temporal span, action branch, and optimization protocol, we compare RGB reconstruction targets, structural pixel targets, and frozen visual foundation model feature targets, then measure which representations are most aligned with downstream action success and robustness across FastWAM inference regimes.

Avoid:

> We propose using DINO/semantic features instead of VAE latents for robotic world models.

That claim is too broad, too method-like, and already covered by related work.

## Experiments Needed to Support the Claim

- Matched runs for RGB, depth, segmentation, DINOv3, SigLIP2, V-JEPA 2.1, and SVG Res-ViT.
- Same action horizon, same total temporal span, same cameras, same optimizer, same number of epochs/steps, and ideally multiple seeds.
- Action-only online LIBERO eval with train/eval-consistent `action_horizon=16` and `replan_steps=10`.
- Action-only baseline without world loss, to quantify the value of each representation target.
- A representation-alignment analysis inspired by LARY: correlate each target's world metrics with action error/success, and report when visual/generative quality fails to predict control quality.
- Robustness evaluation on LIBERO-Plus or controlled visual perturbations, because the strongest motivation for depth/seg/VFM targets is robustness to appearance shifts.
- Representation-specific world metrics:
  - RGB/depth/seg: target-space reconstruction metrics only within the same modality.
  - DINO/SigLIP/V-JEPA: feature MSE/cosine, not PSNR/SSIM.
  - SVG: feature MSE/cosine plus decoded RGB visualization/metrics as a proxy.
- Compute and latency table, because feature encoders and inference regime choices change training and evaluation cost differently.

## Main Risks

- If online LIBERO success rates are similar across targets, the paper becomes mostly an engineering report.
- If DINO/SigLIP/V-JEPA do not beat RGB/depth/seg, the conclusion is still useful but must be framed as "feature targets are not automatically better in FastWAM-style co-training."
- If temporal or dataset settings differ from official FastWAM, every comparison must explicitly state 10 fps vs 20 fps and the matched 1.6s span.
- Feature normalization may dominate results; unnormalized feature MSE can make optimization look worse than the representation actually is.

## Next Step

Keep the current implementation path, but make the next milestone an evaluation table rather than another architecture change:

1. Finish the running SVG future4 training and sync eval videos.
2. Run matched online LIBERO action eval for RGB/depth/seg/DINO/SigLIP/SVG checkpoints.
3. Add an action-only no-world-loss control.
4. Decide whether the story is "feature targets win," "structural pixel targets are sufficient," "the best target depends on inference regime," or "FastWAM is insensitive to target representation."
