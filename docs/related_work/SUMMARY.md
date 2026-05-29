# Related Work Summary

## Mask World Model

File: `2604.19683_mask_world_model.pdf`

This paper argues that RGB future prediction is misaligned with robot control because RGB contains texture, lighting, reflection, and background variation that are weakly related to action selection. It proposes Mask World Model (MWM), which predicts future semantic masks instead of future RGB pixels. The mask target acts as a geometric information bottleneck, preserving object identity, layout, and interaction structure while discarding nuisance appearance.

MWM trains a mask-centric predictive world model and integrates its features with a diffusion policy head. Semantic masks are only used as offline training supervision; deployment uses raw RGB observations. This is directly relevant to our seg/mask FastWAM variant, but MWM goes further because its action policy consumes mask-world predictive features.

## DexWorldModel / CLWM

File: `2604.16484_dexworldmodel_clwm.pdf`

This paper proposes Causal Latent World Model (CLWM) for embodied manipulation. Its main claim is that world-action models should avoid redundant pixel or VAE-latent reconstruction and instead generate DINOv3 semantic features, which better separate interaction semantics from visual noise and improve domain generalization.

It also targets deployment efficiency. Dual-State Test-Time Training memory replaces growing KV cache to keep long-horizon memory at O(1). Speculative Asynchronous Inference overlaps diffusion denoising with robot execution to reduce blocking latency. The paper is relevant to our future direction toward semantic latent WAMs rather than only RGB/depth/seg image prediction.

## Reconstruction or Semantics?

File: `2605.06388_reconstruction_or_semantics.pdf`

This paper systematically studies which latent space is useful for robotic diffusion world models. It compares reconstruction-aligned encoders such as VAE, VA-VAE, and Cosmos with semantic encoders such as V-JEPA 2.1, Web-DINO, and SigLIP 2, while holding the DiT transition model, action conditioning, and data fixed.

The key conclusion is that visual fidelity alone is not enough for selecting a robotic world model. Reconstruction latents often win on pixel-level metrics, but semantic latents perform better on action recoverability, planning, policy-in-world-model rollouts, and latent representation quality. This supports adding semantic latent baselines and action/representation probes to our evaluation.

## LARY

File: `2604.11689_lary.pdf`

LARY introduces a benchmark for evaluating latent action representations learned from visual observations. It evaluates both high-level semantic action understanding, or "what to do", and low-level robot control mapping, or "how to do". The dataset includes over one million videos, 620K image pairs, and 595K motion trajectories across diverse embodiments.

Its main finding is that general visual foundation models, even without explicit action supervision, can outperform specialized embodied latent action models. It also argues that latent visual spaces align better with physical action space than pixel reconstruction spaces. This is relevant to our action evaluation: action loss alone under standard FastWAM may not distinguish RGB/depth/seg unless the action decoder actually uses future world features.

## LDA-1B

File: `2602.12215_lda_1b.pdf`

LDA-1B scales a robot foundation model by training latent dynamics in frozen DINOv3 feature space. It uses a multimodal diffusion transformer to jointly denoise action chunks and future DINO visual features under several objectives: policy learning, forward dynamics, inverse dynamics, and visual forecasting.

The key architectural point for our project is that LDA-1B does not introduce a separate feature-compression adapter. It freezes the DINO encoder, projects action and visual tokens into an MM-DiT, lets the modalities interact through shared attention, and predicts modality-specific outputs with separate heads. This supports our current preference to start with native semantic features plus trainable projection layers.

## Cross-Paper Takeaways

- RGB prediction is a useful baseline but likely spends capacity on task-irrelevant appearance.
- Mask and semantic latent prediction are stronger candidates for action-relevant world modeling.
- For feature-space prediction, the dominant design is frozen semantic encoder plus trainable DiT projections/output heads, not necessarily a learned adapter.
- PSNR/SSIM are insufficient as primary robotics metrics; we need action recovery, planning, policy rollout, mask/depth-specific metrics, or latent probing.
- Our current RGB/depth/seg FastWAM experiments are modality baselines. To test whether world prediction improves action, the action decoder should consume predicted world features or semantic latents.
