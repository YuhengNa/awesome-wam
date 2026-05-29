# FastWAM Codebase Reading Guide

本文档解释当前项目中 FastWAM 代码的实际组织方式和训练链路，重点服务于 RGB/depth/seg/DINO/SigLIP/V-JEPA/SVG 表征对比实验。

## Project Role

当前主代码库是 `external/FastWAM`，本地为 symlink，HPC 训练使用 `/data/user/jhe724/workspace/FastWAM`。本仓库的 `wam/` 目录负责记录设计、实验计划和状态；真实模型、数据、训练、eval 代码都在 `external/FastWAM`。

这不是从零重构后的最终 WAM 框架，而是一个基于 FastWAM 的研究分支：优先保证不同 world prediction target 在同一训练框架下公平比较。

## Entry Points

训练入口：

- `external/FastWAM/scripts/train.py`
- 通过 Hydra 加载 `external/FastWAM/configs/train.yaml`
- task config 例如 `configs/task/libero_svg_resvit_2cam256_future4_1e-4.yaml` 覆盖 data/model/training 参数
- `fastwam.runtime.run_training()` instantiate model 和 dataset，然后交给 `Wan22Trainer`

HPC sbatch 示例：

- `scripts/fastwam_svg_resvit_future4_bs8_acc2_10ep_zzhong778.sbatch`
- `scripts/fastwam_dino_vits_bs8_ckpt_nowandb_10ep.sbatch`
- `scripts/fastwam_siglip2_base_bs8_acc2_10ep.sbatch`

## Config Layout

核心配置分三层：

- `configs/data/*.yaml`: 数据路径、camera keys、帧采样、图像尺寸、action/state shape。
- `configs/model/*.yaml`: 模型构造函数、Wan/ActionDiT 配置、feature encoder 配置、loss weight。
- `configs/task/*.yaml`: 组合 data/model，并指定 batch size、epoch、eval/save 频率等训练超参。

当前 feature target 统一走：

```yaml
model:
  _target_: fastwam.runtime.create_fastwam_feature
```

它最终构造 `FastWAMDINO`，名字保留为 DINO 但实际支持 DINOv3、SigLIP2、V-JEPA 2.1、SVG Res-ViT。

## Data Flow

数据主入口是 `src/fastwam/datasets/lerobot/robot_video_dataset.py`。

关键行为：

- `BaseLerobotDataset` 负责从 LeRobot 数据集取连续 `num_frames` 条 transition。
- `action_size = num_frames - 1`，因此当前 aligned 配置 `num_frames=17` 对应 16 个 action。
- `action_video_freq_ratio` 控制视频下采样：
  - ratio 2: video indices `[0,2,...,16]`，1 条件帧 + 8 future。
  - ratio 4: video indices `[0,4,8,12,16]`，1 条件帧 + 4 future。
- `preserve_multi_camera_video=true` 时，video 保持 `[V,T,C,H,W]`，供 feature encoder 分 camera 编码。
- `concat_multi_camera=horizontal` 时，camera 会先拼成单张宽图，主要用于 VAE/pixel-space 路径。
- `condition_image_keys` 和 `target_image_keys` 支持 depth/seg 变体：第 0 帧用 RGB，future 帧用 depth 或 seg。

`FastWAMProcessor` 负责 action/state 归一化、camera 数量对齐、image transform、action/state merge。训练中的 action loss 是在归一化 action 空间计算的；反归一化主要用于环境 eval 或额外报告。

## Model Flow

### Pixel/VAE Path

`src/fastwam/models/wan22/fastwam.py` 中的 `FastWAM` 是原始联合 video/action 模型。

流程：

1. Wan VAE 把 video 编码为 latent。
2. 第 0 帧 latent 作为 condition，被强制写回 noisy video latent。
3. video expert 和 action expert 分别做 pre-DiT tokenization。
4. `MoT` 将 video/action 两个 expert 的 tokens 混合训练。
5. video head 预测 flow target，action head 预测 action flow target。
6. 总 loss 为 `lambda_video * loss_video + lambda_action * loss_action`。

### Feature Path

`src/fastwam/models/wan22/fastwam_dino.py` 中的 `FastWAMDINO` 是当前所有 feature target 的统一实现。

变化点：

- 不加载 Wan VAE 作为 target encoder。
- 使用 frozen `feature_encoder` 在线编码 `[B,V,T,3,H,W]`。
- video DiT 的 `in_dim/out_dim` 改为 feature dimension，例如 DINOv3-S 为 384，SVG Res-ViT 为 392。
- action expert、MoT 拓扑、scheduler、action loss 尽量保持和原 FastWAM 对齐。
- video loss 只作用在 future feature frames，frame 0 是 condition，不作为预测目标。
- `action_conditioned=false`，即 first-stage representation comparison 不让 video dynamics 额外依赖 action sequence。

## Feature Encoders

Feature encoder 入口在 `src/fastwam/models/vision/feature_encoders.py`：

- DINOv3: `src/fastwam/models/vision/dinov3.py`
- SigLIP2: `SigLIP2FeatureEncoder`
- V-JEPA 2.1: `VJEPA21FeatureEncoder`
- SVG Res-ViT: `SVGResViTFeatureEncoder`

`build_feature_encoder()` 根据 config 中的 `name` 创建 encoder。

SVG Res-ViT 特别之处：

- 输入 `256x256`。
- 输出每 camera `[392,16,16]`，双 camera 拼成 `[392,16,32]`。
- 带 `decode_features()`，eval 时可以把预测 feature decode 回 RGB。
- 当前训练不使用 decode；decode 只用于 eval/visualization。

## Trainer and Evaluation

`src/fastwam/trainer.py` 中的 `Wan22Trainer` 负责训练循环。

关键点：

- 使用 `Accelerate + DeepSpeed ZeRO-1`。
- optimizer 只训练 `model.dit` 和可选 `proprio_encoder`；VAE/feature encoder/text encoder 都冻结。
- `global_step` 是 optimizer step，也就是梯度累积之后的 step。
- `eval_every` 到点会调用 `evaluate()`。
- feature model eval 会调用 `model.evaluate_feature_prediction()`，报告 feature MSE/cosine 和 action L1/L2。
- 如果 feature encoder 有 `decode_features()`，会额外保存 decoded RGB 指标和视频。

SVG eval 视频行含义：

1. predicted feature PCA
2. ground-truth feature PCA
3. predicted feature decoded RGB
4. ground-truth feature decoded RGB
5. ground-truth RGB

训练输出在 `runs/{task}/{run_id}/`，checkpoint 在 `checkpoints/`，eval 视频在 `eval/`。

## Online LIBERO Eval

环境 eval 入口在 `experiments/libero/eval_libero_single.py`。

当前行为：

- `action_horizon` 默认来自 `cfg.data.train.num_frames - 1`，当前 aligned 版本应为 16。
- `replan_steps=10` 用于与官方 FastWAM action chunk 执行策略对齐。
- `visualize_future_video=false` 时走 action-only eval。
- `visualize_future_video=true` 时：
  - RGB/VAE 模型走 `infer_joint()`。
  - SVG feature 模型走 `infer_features()` 后调用 `decode_features()` 生成 RGB future visualization。
- 在线 eval 输入分辨率会读取 task/data config；SVG 使用 256，DINO/SigLIP 使用各自配置尺寸。

当前 HPC 环境的 FastWAM conda env 不含 LIBERO 包，因此完整环境 eval 更适合在本机已有 LIBERO 环境处跑。

## Current Temporal Contract

一定要区分两个数据源：

- 官方/原始 no-noops FastWAM 数据：20 fps。
- 当前 mask/depth aligned 数据：10 fps。

当前主要实验使用 10 fps 数据：

- action: 16 steps，每步 0.1s，总跨度 1.6s。
- ratio 2 video: 8 future frames，每帧间隔 0.2s。
- SVG future4 ratio 4: 4 future frames，每帧间隔 0.4s。

如果切回 20 fps 数据，同样 `num_frames=17` 只覆盖 0.8s，这会破坏和当前 depth/seg/feature runs 的时间对齐。

## How To Add A New Representation

1. 在 `src/fastwam/models/vision/feature_encoders.py` 中实现 frozen encoder，输入 `[B,V,T,3,H,W]` 或 `[B,3,T,H,W]`，输出 `[B,D,T,H,W]`。
2. 在 `build_feature_encoder()` 注册 `name`。
3. 新增 `configs/model/fastwam_<name>.yaml`，设置 `feature_encoder_config`、`video_dit_config.in_dim/out_dim`。
4. 新增 `configs/data/libero_<name>_2cam.yaml`，保持 dataset、action horizon、camera 处理和对照组一致。
5. 新增 `configs/task/libero_<name>_...yaml`，对齐 batch、epoch、eval/save。
6. 先跑 smoke/benchmark，再跑长训。
7. 对 feature target 不要直接和 RGB/depth/seg 比 PSNR/SSIM；除非像 SVG 一样有 decoder，并且明确指标含义是 decoded RGB proxy。

## Common Pitfalls

- 不要混淆 `num_frames` 和实际 video frames；video frames 是 `range(0, num_frames, action_video_freq_ratio)`。
- 不要把 10 fps mask/depth 数据的时间跨度套到 20 fps 原始数据上。
- feature path 的 action loss 是归一化空间，环境 eval 才需要严格反归一化。
- SVG 当前 job `299097` 使用 `microbatch_size=4`；配置已更新为 80，下一次提交才会生效。
- `microbatch_size` 只影响 frozen encoder 分块，不改变训练目标。
- `eval_every` 会跑 inference 和保存视频，耗时明显高于普通 train step。
- `save_every=5000` 和 `save_at_end=true` 才会保留长训 checkpoint；短 benchmark 通常关闭保存。
