# Related Work PDFs

This folder stores local copies of papers that are directly relevant to the
WAM / FastWAM modality experiments. Use these PDFs as the source of truth for
future reading and summaries.

## Papers

- `2604.19683_mask_world_model.pdf`
  - Title: Mask World Model: Predicting What Matters for Robust Robot Policy Learning
  - Source: https://arxiv.org/pdf/2604.19683
  - Relevance: Predicts future semantic masks instead of RGB and uses mask-centric predictive features for robot policy learning.

- `2604.16484_dexworldmodel_clwm.pdf`
  - Title: DexWorldModel: Causal Latent World Modeling towards Automated Learning of Embodied Tasks
  - Source: https://arxiv.org/pdf/2604.16484
  - Relevance: Uses DINOv3 latent features as generative world-model targets and focuses on efficient causal latent modeling for embodied tasks.

- `2605.06388_reconstruction_or_semantics.pdf`
  - Title: Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models
  - Source: https://arxiv.org/pdf/2605.06388
  - Relevance: Compares reconstruction-aligned latents with semantic latents for action-conditioned robotic world models.

- `2604.11689_lary.pdf`
  - Title: LARY: A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment
  - Source: https://arxiv.org/pdf/2604.11689
  - Relevance: Benchmarks latent action representations on semantic action classification and low-level robot action regression.

- `2602.12215_lda_1b.pdf`
  - Title: LDA-1B: Scaling Latent Dynamics Action Model
  - Source: https://arxiv.org/pdf/2602.12215
  - Relevance: Predicts future DINO features with a multimodal diffusion transformer and jointly models actions and latent visual dynamics.

## Derived Notes

- `SUMMARY.md`: short paper-by-paper summaries.
- `REPRESENTATION_TAXONOMY.md`: representation grouping for this project.
- `FEATURE_DIT_ARCHITECTURES.md`: architecture notes for feature-space world models and a recommended first implementation.
- `LDA_CODE_NOTES.md`: notes from the local LDA-1B code clone, including feature preprocessing and flow-matching details.

## Notes For This Project

- Prefer citing these local PDFs when discussing related work.
- The mask/depth/RGB FastWAM experiments should be interpreted as modality baselines, not full reproductions of MWM.
- For stronger claims about action improvement, add experiments where the action decoder consumes predicted world features or semantic latents.
