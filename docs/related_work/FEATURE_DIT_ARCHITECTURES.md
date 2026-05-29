# Feature-Space DiT Architectures

## Scope

This note summarizes how recent feature-prediction world-model papers build the transition model, with emphasis on whether they use adapters and what DiT architecture is appropriate for our next FastWAM variant.

## LDA-1B

LDA-1B predicts future visual states in frozen DINOv3 feature space rather than VAE pixel latents. It keeps the DINO encoder frozen, uses a VLM for language/current-observation conditioning, and trains a multimodal diffusion transformer.

Architecture:
- Target: DINOv3 feature map, reported as `14 x 14 x 384` for `224 x 224` images with DINOv3-ViT-S.
- History: two past timesteps of DINO observations and actions.
- Backbone: MM-DiT with hidden size `1536`, `16` layers, `32` heads.
- Token mixing: action and visual tokens are concatenated and interact through shared self-attention.
- Modality handling: modality-specific input projections, QKV projections, FFNs, and output heads.
- Conditioning: diffusion timestep, task embedding, and VLM/language tokens via AdaLN and cross-attention.
- Objective: flow matching over future action chunks and future DINO features.
- Adapter: no feature-compression adapter is described; only trainable projection/output layers around frozen feature encoders.

## DexWorldModel / CLWM

DexWorldModel also predicts DINOv3 features directly. It is closer to a causal streaming world model than our current batch FastWAM setting, but its representation choice is relevant.

Architecture:
- Target: frozen DINOv3 feature map `C x H' x W'`, patch size `P=16`.
- Backbone: Mixture of Transformers initialized from Wan2.2-5B.
- Sharing: latent video model and action model share core transformer blocks.
- Modality handling: domain-specific input/output projections and flow timestep embeddings.
- Generation: autoregressive flow matching first predicts future DINO features, then predicts action chunks conditioned on history, language, and predicted future semantics.
- Action horizon: `tau=16`.
- Adapter: no S-VAE-style adapter; feature dimensionality is handled through linear projections.

## Reconstruction or Semantics?

This paper gives the most directly reusable DiT recipe because it controls the transition model while varying the latent representation.

Architecture:
- Target options: VAE-like reconstruction latents, V-JEPA 2.1, Web-DINO, SigLIP 2, with or without compact S-VAE adapter.
- Native feature shape: `N x D`, typically `16 x 16` tokens with `D=1024` or `1152`.
- Backbone presets:
  - DiT-S: hidden `384`, depth `12`, heads `6`.
  - DiT-B: hidden `768`, depth `12`, heads `12`.
  - DiT-L: hidden `1024`, depth `24`, heads `16`.
- Attention: factorized spatial-temporal DiT; spatial attention is non-causal within a frame, temporal attention is causal across frames.
- Context: `H=2` history frames; predicts future frames in a `T=10` clip.
- Objective: optimal-transport flow matching; loss only on future frames.
- Native high-dimensional features: use dimension-aware noise schedule shift and a shallow-wide DDT output head with `2048` readout width.
- Adapter: optional S-VAE compresses `D -> 96`, but native semantic features are explicitly evaluated and considered practical because channel dimension mostly affects input/output projections, not transformer block cost.

## Mask World Model

MWM is not feature-space in the DINO sense, but it is useful for action-head integration. It renders masks into image-like targets, encodes them with a shared VAE, and trains a mask-dynamics DiT.

Architecture:
- Target: future semantic mask latents from a fixed video VAE.
- Backbone: 28-layer DiT, hidden `2048`, `32` heads.
- Tokenization: multi-view spatiotemporal tokens with 3D RoPE.
- Conditioning: text cross-attention and diffusion timestep AdaIN/AdaLN-style conditioning.
- Action integration: action expert transformer attends to predictive feature banks from the mask DiT.
- Key lesson: better world targets only affect action if the action decoder consumes predictive world features.

## Recommendation For Our First Feature Baseline

Use native DINO/V-JEPA features first, without a compression adapter.

Initial design:
- Compute frozen visual features online, matching LDA's public implementation. Offline feature caching is only a later speed optimization.
- Start without extra feature normalization to match LDA; log feature mean/std and keep dataset-level normalization as an ablation flag.
- Use a factorized spatial-temporal DiT:
  - spatial self-attention within each frame/view,
  - causal temporal self-attention for each spatial location,
  - optional cross-view attention every few blocks if using two cameras.
- Use trainable linear input projection `D -> hidden_dim` and output projection `hidden_dim -> D`.
- Add a shallow-wide output head for native `D=768/1024/1152` features before considering any adapter.
- Train with flow matching on future feature frames only; keep history frames clean or lightly noised as context.
- For the first FastWAM-aligned baseline, keep `video_dit_config.action_conditioned=false`. The goal is to change only the prediction target from Wan VAE latents to DINOv3 features.
- Do not change the action decoder or MoT attention mask in the first baseline. Action-conditioned feature prediction and action attention to future features are later ablations.

Practical first config:
- Encoder: DINOv2/DINOv3 ViT with `patch=16`.
- Resolution: `224` gives `14 x 14` feature tokens; `256` gives `16 x 16`.
- Horizon: keep the aligned FastWAM setting first, `9` video frames and `16` action steps.
- DiT size: start with the existing FastWAM video DiT size and Wan-initialized transformer blocks; replace only feature input/output projections.
- Loss: flow-matching velocity MSE on future DINOv3 features, with feature cosine diagnostics.

Do not introduce an adapter in the first implementation. It adds an extra training stage and can change the action-relevant geometry. The papers that most resemble our target direction, LDA-1B and DexWorldModel, both rely on frozen semantic encoders plus trainable projections around the DiT.
