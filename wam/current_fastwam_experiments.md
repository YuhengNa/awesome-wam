# Current FastWAM Experiment Status

Last updated: 2026-05-15.

## Code and Runtime Layout

- Local development repo: `external/FastWAM`, symlinked to `/data/LFT-W02_data/junjie/VLA_WM/FastWAM`.
- HPC training repo: `/data/user/jhe724/workspace/FastWAM`.
- Standard workflow: edit locally, sync changed configs/scripts/code to HPC, submit jobs with `zzhong778` or `jhe724`.
- Preferred local result mirror: `runs/` under this repository root.

## Data and Temporal Alignment

Two LIBERO data sources differ in FPS:

- Original FastWAM no-noops data: 20 fps.
- Current mask/depth data used by RGB/depth/seg/DINO/SigLIP/SVG aligned runs: 10 fps.

The aligned 10 fps setting uses `num_frames=17` action steps before video subsampling. Therefore:

- `action_horizon=16`, one action every `0.1s`, total action span `1.6s`.
- Original aligned video setting: `action_video_freq_ratio=2`, giving 1 condition frame + 8 future frames at `0.2s` spacing.
- For the aligned 10 fps data, the SVG future4 setting uses `action_video_freq_ratio=4`, giving 1 condition frame + 4 future frames at `0.4s` spacing, still covering `1.6s`.
- For the official no-noops data, the same `num_frames=17` / `action_video_freq_ratio=4` setting is a shorter `0.8s` horizon because the source data is 20 fps.

## Implemented Representation Variants

- Pixel-space VAE variants: aligned RGB, depth, and segmentation targets. Depth and segmentation are treated as RGB-like image targets and share the Wan VAE path.
- Feature-space variants: DINOv3, V-JEPA 2.1, SigLIP2, and SVG Res-ViT.
- DINOv3-S at `224x224`: 2-camera feature grid `[B, 384, T, 14, 28]`.
- SVG Res-ViT at `256x256`: 2-camera feature grid `[B, 392, T, 16, 32]`, with an SVG decoder for RGB visualization.
- SVG-DINO-P at `256x256`: 2-camera feature grid `[B, 384, T, 16, 32]`, with SVG P-stage decoding for RGB visualization.

All feature variants keep FastWAM's action branch and `action_conditioned=false` video dynamics for the first representation-only comparison.

## SVG-DINO-P Full LIBERO Online Eval

Completed on 2026-05-15 with `zzhong778` on HPC, using 4 H100 GPUs for parallel LIBERO task evaluation.

Important setting note: these two runs use the official LIBERO no-noops datasets but a **short-horizon future4 configuration**, not the official FastWAM 32-action / 8-future-frame horizon. With `num_frames=17`, `global_sample_stride=1`, and `action_video_freq_ratio=4`, the model predicts 16 action steps and 4 future frames sampled at indices `[0, 4, 8, 12, 16]`. On the original 20 fps no-noops data this covers about `0.8s`.

Training parameters:

- Data config: `configs/data/libero_svg_official_2cam_future4.yaml`.
- Tasks: `libero_svg_dino_p_2cam256_future4_1e-4` and `libero_svg_dino_p_vaecond_2cam256_future4_1e-4`.
- Cameras: `image` and `wrist_image`, preserved as two views and resized to `256x256`.
- Feature target: SVG P-stage DINO features, `feature_dim=384`, `patch_size=16`, per-view grid `16x16`, concatenated grid `16x32`.
- Model: full Wan2.2 video DiT width (`hidden_dim=3072`, `ffn_dim=14336`, 30 layers, 24 heads), compact ActionDiT branch (`hidden_dim=1024`, `ffn_dim=4096`), `action_conditioned=false`, `mot_checkpoint_mixed_attn=true`.
- Optimizer: batch `8` per GPU, 4 GPUs, gradient accumulation `2`, effective batch `64`; `lr=1e-4`, cosine schedule, `weight_decay=1e-2`, `bf16`, 10 epochs, `save_every=5000`, `eval_every=200`, SVG encoder `microbatch_size=160`.

Eval parameters:

- Checkpoint: `step_030000.pt`.
- Eval mode: action-only online LIBERO eval, `visualize_future_video=false`.
- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; 10 tasks per suite, 50 trials per task.
- Action settings: `action_horizon=16`, `replan_steps=10`, `num_inference_steps=10`.
- Gripper settings: `binarize_gripper=true`, `gripper_action_mode=rlds_01`.
- Normalization: each eval uses its run-local `dataset_stats.json`.
- Text: cached text context enabled.

Results:

| Variant | Overall | Spatial | Object | Goal | LIBERO-10 | Eval Dir |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pure SVG-DINO-P | 87.35% | 485/500 = 97.0% | 465/500 = 93.0% | 405/500 = 81.0% | 392/500 = 78.4% | `runs/libero_eval/libero_svg_dino_p_2cam256_future4_1e-4_step_030000_ah16_rp10_full_zzhong778_20260515_115343` |
| VAE-cond SVG-DINO-P | 88.05% | 475/500 = 95.0% | 485/500 = 97.0% | 414/500 = 82.8% | 387/500 = 77.4% | `runs/libero_eval/libero_svg_dino_p_vaecond_2cam256_future4_1e-4_step_030000_ah16_rp10_full_zzhong778_20260515_115343` |

Interpretation: VAE conditioning gives a small overall gain (`+0.70 pp`) and improves `object`/`goal`, while pure SVG-DINO-P is slightly better on `spatial` and `libero_10`. Both are below the known official FastWAM LIBERO-10 baseline, so the next fair comparison should use the official-aligned `action_horizon=32` / 8-future or 4-future-with-32-action setting.

## Compact Interpolated SVG-DINO-P Run

Purpose: test whether the interpolated compact ActionDiT backbone can initialize the SVG-DINO-P video expert while keeping the architecture as MoT.

Key implementation detail:

- The model still uses `MoT(video_expert, action_expert)`.
- The video expert remains `WanVideoDiT`, but is built at compact ActionDiT scale: `hidden_dim=1024`, `ffn_dim=4096`, 30 layers, 24 heads, `attn_head_dim=128`.
- Shared transformer keys (`text_embedding`, `time_embedding`, `time_projection`, and `blocks`) are initialized from `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`.
- Video-specific `patch_embedding` and `head` stay randomly initialized because their input/output channels are SVG-DINO-P features, not actions.
- The action expert is the normal compact `ActionDiT` initialized from the same checkpoint.

Training setting:

- Task: `libero_svg_dino_p_compact_2cam256_future4_1e-4`.
- Data: official LIBERO no-noops SVG future4 data, `num_frames=17`, `action_horizon=16`, `action_video_freq_ratio=4`.
- Target: SVG P-stage DINO features, `feature_dim=384`, grid `[T, 16, 32]` for two cameras.
- GPUs: 4 H100 with `zzhong778`; batch `8` per GPU, gradient accumulation `2`, effective batch `64`.
- Optimizer: `lr=1e-4`, `weight_decay=1e-2`, `bf16`, 10 epochs, `eval_every=200`, `save_every=5000`, SVG encoder `microbatch_size=160`.

Final status:

- Smoke job `301734` completed successfully for 20 steps.
- Smoke final metrics: `loss=1.1316`, `loss_action=0.3627`, `loss_video=0.7689`, about `0.55 step/s` and `35.11 samples/s`.
- Long run job `301745` was cancelled on 2026-05-16 after `17,220` train steps because the conclusion was already clear.
- Final local mirror: `runs/libero_svg_dino_p_compact_2cam256_future4_1e-4/zz_svgp_compact_future4_10ep_301745_20260515_171741`.
- Final observed train metrics: step `17,220`, `loss_video=0.3793`, about `0.68 step/s`.
- Final observed eval metrics: step `17,200`, `feature_mse=0.3629`, `feature_cosine=0.5119`, `decoded_psnr=13.31`.
- Matched full-Wan SVG-DINO-P eval at step `17,200` had `feature_mse=0.0259`, `feature_cosine=0.9213`, `decoded_psnr=16.14`.
- Conclusion: compact interpolated video expert is trainable but not competitive for SVG-DINO-P feature prediction; it plateaus around `feature_mse=0.36`, roughly an order of magnitude worse than full Wan2.2 video DiT. This supports keeping the full video-pretrained Wan backbone for feature-space WAM, or switching to another genuinely video-pretrained smaller backbone rather than an interpolated compact ActionDiT.

## Current SVG Res-ViT Run

Running job:

- Job: `299097`
- Account: `zzhong778`
- Task: `libero_svg_resvit_2cam256_future4_1e-4`
- Run directory: `/data/user/jhe724/workspace/FastWAM/runs/libero_svg_resvit_2cam256_future4_1e-4/zz_svg_future4_10ep_299097_20260513_200104`
- Local eval mirror: `runs/libero_svg_resvit_2cam256_future4_1e-4/zz_svg_future4_10ep_299097_20260513_200104/eval/`
- GPUs: 4 H100
- Batch: per-GPU `8`, gradient accumulation `2`, effective batch `64`
- Training length: 10 epochs, `52910` optimizer steps
- Current script now defaults to `model.feature_encoder_config.microbatch_size=80`, but job `299097` was launched before this change and is using `microbatch_size=4`.

Observed training behavior:

- Stable speed: about `0.47-0.48 step/s`, `30.3-30.4 samples/s`.
- `microbatch_size=80` benchmark completed without OOM and reached about `0.53 step/s`, `33.6 samples/s`.
- Current job can continue because the speed gap is only about 10-11% and the run is already stable.

## SVG Eval and Visualization

Eval runs every 200 steps. Available local videos:

- `step_000200_rank_000-003.mp4`
- `step_000400_rank_000-003.mp4`
- `step_000600_rank_000-003.mp4`
- `step_000800_rank_000-003.mp4`

Each SVG eval video has 5 rows:

1. Predicted feature PCA.
2. Ground-truth feature PCA.
3. Predicted features decoded to RGB.
4. Ground-truth features decoded to RGB.
5. Ground-truth RGB.

Latest observed eval metrics:


| Step | val_loss | feature_mse | feature_cosine | action_l2 | action_l1 | decoded_psnr | decoded_ssim |
| ---- | -------- | ----------- | -------------- | --------- | --------- | ------------ | ------------ |
| 200  | 1.0995   | 1.0952      | 0.1790         | 0.2168    | 0.2184    | 9.0437       | 0.1067       |
| 400  | 0.7027   | 1.0448      | 0.2770         | 0.1579    | 0.1784    | 9.8981       | 0.1295       |
| 600  | 1.0374   | 0.8687      | 0.3164         | 0.0614    | 0.1664    | 9.8303       | 0.1307       |
| 800  | 0.9655   | 0.7151      | 0.3248         | 0.0377    | 0.1311    | 10.6536      | 0.1396       |


Interpretation: training is normal. `val_loss` is not monotonic, but feature MSE, feature cosine, action error, and decoded RGB metrics are moving in the right direction.

## Important Configs and Scripts

- SVG future4 data: `external/FastWAM/configs/data/libero_svg_2cam_future4.yaml`
- SVG future4 task: `external/FastWAM/configs/task/libero_svg_resvit_2cam256_future4_1e-4.yaml`
- SVG model: `external/FastWAM/configs/model/fastwam_svg_resvit.yaml`
- SVG 10-epoch sbatch: `external/FastWAM/scripts/fastwam_svg_resvit_future4_bs8_acc2_10ep_zzhong778.sbatch`
- Compact SVG-DINO-P task: `external/FastWAM/configs/task/libero_svg_dino_p_compact_2cam256_future4_1e-4.yaml`
- Compact SVG-DINO-P model: `external/FastWAM/configs/model/fastwam_svg_dino_p_compact.yaml`
- Compact SVG-DINO-P smoke sbatch: `external/FastWAM/scripts/fastwam_svg_dino_p_compact_future4_bs8_acc2_smoke_zzhong778.sbatch`
- Compact SVG-DINO-P 10-epoch sbatch: `external/FastWAM/scripts/fastwam_svg_dino_p_compact_future4_bs8_acc2_10ep_zzhong778.sbatch`
- SVG microbatch benchmark scripts:
  - `external/FastWAM/scripts/fastwam_svg_resvit_future4_mb72_bench_zzhong778.sbatch`
  - `external/FastWAM/scripts/fastwam_svg_resvit_future4_mb80_bench_zzhong778.sbatch`

## Next Steps

- Keep job `299097` running and periodically mirror new eval videos back to local `runs/`.
- Compact SVG-DINO-P job `301745` is stopped; use its synced logs/videos only as a negative ablation.
- Use `microbatch_size=80` for the next SVG future4 launch.
- Compare SVG feature prediction against DINOv3/SigLIP/RGB/depth/seg using matched action horizon, dataset, optimizer, and training length.
- For online LIBERO eval, ensure `action_horizon=16`, `replan_steps=10`, and task-specific input resolution stay consistent with training.
