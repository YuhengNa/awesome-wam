# LDA-1B Code Notes

Local clone: `external/LDA-1B`

## Relevant Files

- `lda/model/modules/action_model/MMDiT_ActionHeader.py`: main action and visual feature flow-matching head.
- `lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py`: MMDiT block implementation.
- `lda/model/modules/dino_model/dino.py`: older DINOv2 wrapper.
- `lda/model/modules/dino_model/dino_transforms.py`: ImageNet normalization utilities.
- `lda/config/training/LDA_pretrain.yaml`: pretraining config.

## Visual Feature Processing

For `vision_encoder_type: dinov3`, LDA loads a Hugging Face DINOv3 model and processor:

- `AutoImageProcessor.from_pretrained(...)`
- `DINOv3ViTModel.from_pretrained(...).eval()`

Images are converted by the processor into `pixel_values`, then fed through DINOv3 under `torch.no_grad()`. The code uses `output.last_hidden_state` directly as the visual feature target.

Important observation: the public implementation does not appear to apply an extra dataset-level mean/std normalization to DINO features. It relies on:

- official image preprocessing before DINO,
- DINO's own normalized hidden states,
- flow matching against Gaussian noise,
- MMDiT internal LayerNorm/RMSNorm.

## Flow Matching

The visual target is noised by linear interpolation:

- `noisy_obs = (1 - t) * noise + t * target_feature`
- `obs_velocity = target_feature - noise`

The model predicts `obs_velocity` and uses MSE loss. The same pattern is used for actions:

- `noisy_action = (1 - t) * noise + t * action`
- `action_velocity = action - noise`

Timestep samples come from a beta distribution:

- `Beta(alpha=1.5, beta=1.0)`
- transformed by `noise_s=0.999`

## Tokenization Pattern

For DINO/V-JEPA features, LDA concatenates current observation features and noised future observation features along the channel dimension, then applies:

- `obs_merger: Linear(num_chans * (obs_horizon + 1), input_embedding_dim)`
- `obs_projector: Linear(hidden_size, num_chans)`

This means current and future feature tokens are merged per spatial token before entering MMDiT. It is not a direct Wan-style sequence of separate video tokens.

The DINOv3 path keeps global tokens in `last_hidden_state` and tracks:

- `cls_token = 1`
- `register_tokens = vision_encoder.config.num_register_tokens`
- `glob_len = cls_token + register_tokens`

For our implementation, we should explicitly decide whether to predict only patch tokens or also CLS/register tokens. Predicting patch tokens only is cleaner for world dynamics and evaluation.

## MMDiT Structure

The MMDiT implementation uses:

- separate image/action LayerNorms,
- separate image/action cross-attention to text tokens,
- joint self-attention over image and action streams,
- AdaLN-style conditioning from diffusion timestep plus task embedding,
- final RMSNorm,
- separate `image_proj_out` and `action_proj_out`.

This supports our plan to keep modality-specific input/output projections while sharing the transformer core.

## Takeaways For This Project

Do not copy LDA preprocessing blindly. For our feature-space FastWAM variant:

- Use the official encoder image processor when extracting cached DINO/V-JEPA features.
- Default to online frozen feature extraction, matching LDA's implementation. Offline caching can remain a later speed optimization, but should not be required for the first implementation.
- If online extraction is used, keep the encoder in `eval()` and wrap feature extraction in `torch.no_grad()`.
- Start without extra feature-channel normalization to match LDA, but log feature mean/std and keep a config flag for dataset-level normalization if target scale is unstable.
- Predict patch tokens first; ignore CLS/register tokens unless a later ablation shows they help.
- Keep flow matching target scale controlled: inspect feature mean/std and velocity MSE before long training.
- Use MSE as the primary flow loss and log cosine similarity on predicted vs target velocity/features.
- Use trainable feature input/output projections around a Wan-initialized DiT backbone.
