# Representation Taxonomy

## 1. Pixel / RGB Representations

These predict or reconstruct future RGB frames directly. They preserve visual detail but also force the world model to spend capacity on texture, lighting, reflections, and background changes.

Examples: RGB video prediction, pixel-space diffusion, RGB-centric world models.

## 2. Reconstruction Latents

These use an autoencoder latent as the world-model space. They are usually easy to decode and score well on PSNR/SSIM, but the latent is optimized for reconstruction rather than control.

Examples: Wan VAE, SD3 VAE, VA-VAE, Cosmos latent, FLUX VAE, MAGVIT2-style generative pixel encoders.

Our current RGB/depth/seg FastWAM variants still belong mostly here because all targets are represented as image-like inputs and encoded by the Wan VAE.

## 3. Structured Visual Targets

These replace RGB targets with more task-relevant visual structure while often still using an image/VAE interface.

Examples:
- Depth maps: geometry and layout.
- Semantic or instance masks: object identity, contact regions, spatial relations.
- Rendered mask palettes: discrete mask IDs converted into RGB-compatible images.

Mask World Model is the clearest example: future semantic masks are rendered into RGB palettes and encoded with a shared VAE, then predictive mask features are used by the action head.

## 4. Semantic Feature Latents

These use frozen visual foundation model features as the prediction space. They emphasize object layout, task semantics, and interaction-relevant changes instead of pixel fidelity.

Examples:
- DINOv2 / DINOv3
- V-JEPA 2 / V-JEPA 2.1
- Web-DINO
- SigLIP 2

Reconstruction or Semantics? compares semantic encoders against VAE-like latents. DexWorldModel predicts DINOv3 features directly. These are strong candidates for the next WAM representation after RGB/depth/seg baselines.

## 5. Latent Action Representations

These represent motion or control itself, usually extracted from visual changes or learned through inverse/forward dynamics.

Examples:
- Embodied LAMs: LAPA, UniVLA, villa-X.
- General LAMs: LAPA-DINOv2, LAPA-DINOv3, LAPA-SigLIP2, LAPA-MAGVIT2.
- Discrete VQ action tokens or continuous latent action embeddings.
- Low-level continuous robot action chunks.

LARY evaluates these representations by semantic action classification and low-level control regression.

## 6. Predictive / Policy-Conditioning Features

These are internal world-model features exposed to an action decoder, not necessarily decoded into images.

Examples:
- MWM predictive feature bank from mask forecasting DiT blocks.
- CLWM predicted DINOv3 future semantic features.
- Future world tokens consumed by an action decoder.

This category is important for our project: predicting better future depth/seg/RGB is not enough unless the action branch can use the predicted world features.

## Practical Grouping For This Project

1. Current baselines: RGB, depth, seg as Wan-VAE image latents.
2. Stronger structured target: mask/depth with modality-specific metrics and future-only eval.
3. Semantic latent WAM: DINO/V-JEPA/SigLIP feature prediction.
4. Action-aware WAM: action decoder attends to predicted world features.
5. Evaluation probes: IDM/action recoverability, latent success classifier, downstream policy rollout.
