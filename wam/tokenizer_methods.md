# Tokenizer Method Notes

本文档整理当前讨论的三类 semantic / feature tokenizer 方案，并标注它们在本仓库中的实现状态。这里的 `x_t` 默认表示 frozen visual teacher 的 patch feature，例如 SVG-P / DINOv3 / SigLIP feature，而不是 RGB pixel。

## 1. Per-Frame S-VAE / Channel Adapter

**核心定义：** 每一帧独立压缩，不做时序压缩。

```text
RGB_t -> frozen encoder -> x_t [V, N, D]
x_t -> S-VAE encoder -> z_t [V, N, d]
z_t -> S-VAE decoder -> x_hat_t
```

其中 `V` 是相机数，`N` 是每帧 spatial tokens，`D` 是 teacher feature 维度，`d` 是压缩后维度，例如 64/96/128。这个方案最接近 Semantic-WM：adapter 的目标是让高维 semantic feature 变小，后续 world model 再预测 future adapted features。

**训练目标：**

```text
L = feature_mse(x_hat_t, x_t)
  + cosine(x_hat_t, x_t)
  + beta * KL
  + optional pixel decoder loss
```

**优点：** 训练最稳，和 Semantic-WM 对齐；不会引入时间压缩导致的 action alignment 问题；适合作为第一版 semantic tokenizer baseline。

**缺点：** 单纯 channel adapter 的创新性较弱；动态信息不被显式建模；world model 仍需预测整帧 latent，静态背景 token 会占用大量容量。

**当前代码状态：已实现 standalone S-VAE adapter，并完成 SVG-P smoke。**

- `external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py`
  - `SemanticFeatureVAEConfig`
  - `SemanticFeatureVAE`
  - 输入 `[B,V,N,D]`
  - latent `[B,V,N,d]`
  - 重建 `[B,V,N,D]`
- `external/openpi/scripts/train_svae_libero.py`
  - 复用 OpenPI LIBERO dataloader
  - 复用 SVG-P / DINOv3 frozen teacher encode
  - 对 clip feature `[B,V,F,N,D]` 展平帧轴为 `[B*F,V,N,D]` 训练逐帧 adapter

**当前 smoke 结果：**

- 数据：本地 `LIBERO_10 no-noops` LeRobot 数据
- teacher：SVG-P
- 视角：`base_0_rgb`
- future deltas：`1,3,6,9`
- batch size：8
- steps：100
- latent dim：96
- 结果：loss 从 `0.4332` 降到 `0.0926`，recon 从 `0.3759` 降到 `0.0710`，无 NaN，梯度正常。

这只说明训练链路跑通，不代表下游 action performance 结论。

已有相关基础设施：

- `external/openpi/src/openpi/models_pytorch/pi0_pytorch.py`: `encode_future_images_with_teacher()` 支持 raw SigLIP / raw DINOv3 future targets。
- `external/openpi/src/openpi/training/config.py`: `pi05_libero_gen`, `pi05_libero_gen_dino`, `pi05_libero_gen_dino32`, `pi05_libero_gen_dino128` 是当前 pi0.5 generation-loss target 配置；其中 DINO32/128 是固定随机投影，不是可训练 S-VAE。
- `external/openpi/scripts/compute_pi05_gen_target_stats.py`: 统计 raw / projected DINO target normalization。

**建议实现入口：**

- 新增 `external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py`
- 新增 `external/openpi/scripts/train_svae_libero.py`
- 复用 `train_predictive_feature_vae_libero.py` 和 `train_lam_libero.py` 里的 frozen teacher encode、LIBERO future frame loader、PCA/SVG 可视化逻辑。

## 2. PV-VAE-Style Temporal Predictive Feature VAE

**核心定义：** 输入一个 feature clip，把 future frames 按时间分组压缩，训练 decoder 从 observed prefix 重建完整 clip。

```text
x_0, x_1, ..., x_16
  -> encoder
  -> z_0, z_1, ..., z_4        # temporal_compression=4
  -> decoder
  -> x_hat_0, ..., x_hat_16
```

第 0 帧单独作为一个 group，future frames 每 4 帧压成一个 latent group：

```text
group 0: x_0
group 1: x_1 ... x_4
group 2: x_5 ... x_8
group 3: x_9 ... x_12
group 4: x_13 ... x_16
```

训练时随机采样 `observed_groups`，只把 prefix 输入 encoder，对缺失 future groups 使用 learned pad latent。decoder 必须同时 reconstruct observed frames 和 predict dropped future frames。

**训练目标：**

```text
L = weighted_feature_mse
  + cosine_weight * cosine_loss
  + delta_weight * temporal_delta_mse
  + kl_weight * KL
```

当前实现还记录：

- `future_mse`: dropped / future 区域的 feature MSE
- `static_future_mse`: 复制最后 observed frame 的静态 baseline
- `pred_delta_norm`, `target_delta_norm`, `delta_ratio`: 预测动态幅度诊断

**优点：** 真正做 temporal compression；token 数能从 `17 frames` 降到 `5 latent groups`；概念上接近 PV-VAE / Wan-VAE 的 4x temporal compression。

**缺点：** 对 LIBERO 这种静态背景占比极高的机器人视频，predictive objective 很容易学到复制/平均 future；一个 latent group 表示 4 帧，后续和 action chunk 对齐会更复杂。

**当前代码状态：已实现。**

- `external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py`
  - `PredictiveFeatureVAEConfig`
  - `PredictiveFeatureVAE`
  - factorized spatial/temporal attention block
  - temporal attention pooling
  - prefix observed groups + pad latent decode
  - weighted recon / cosine / delta / KL loss
- `external/openpi/scripts/train_predictive_feature_vae_libero.py`
  - OpenPI LIBERO loader
  - SVG-P / DINOv3 frozen teacher encode
  - `future_deltas` 控制采样 stride
  - PCA feature visualization
  - SVG-P decoded-RGB visualization
- `external/openpi/scripts/pvvae_libero_svgp_4gpu_zzhong778.sbatch`
  - HPC 4-GPU launch script

**当前实验经验与路线更新：**

- `stride=1` 对应 LIBERO 20Hz 下 `0.05s` frame interval，17 帧覆盖 `0.8s`。
- `stride=2` 对应 `0.1s` interval，17 帧覆盖 `1.6s`，目前作为更合理的 diagnostic setting。
- 观察到 `static_future_mse` 很低，说明静态 shortcut 很强；这不是代码崩，而是数据运动信号和 predictive objective 的组合问题。
- 根据师兄最新建议，PV-VAE 下一步主线转到 OXE / Bridge v2。LIBERO 只作为历史 smoke/debug，不作为主要结论数据。
- OXE 路径已知：`/data/user/jhe724/workspace/data/OXE`，优先尝试 Bridge v2 子集。
- 当前最先要做的不是改 PV-VAE 模型，而是把 Bridge v2/OXE episode 转成统一 clip dataloader contract：

```python
{
    "images": "[B,V,T,C,H,W]",
    "actions": "[B,T-1,A] or None",
    "instruction": "list[str] or None",
    "dataset_name": "list[str]",
    "episode_id": "list[str]",
}
```

- 参考 `Reconstruction or Semantics?` 的思路，PV-VAE 的评价不能只看 reconstruction，还要看 latent 是否保留 action-relevant / task-relevant 信息。第一阶段先看 `future_mse < static_future_mse`、`delta_ratio` 不塌缩和 SVG/PCA 可视化；后续再加 inverse-dynamics / action probe。

**当前执行入口：**

- `wam/pvvae_oxe_execution_plan.md`: Bridge v2/OXE 执行计划。
- `external/openpi/scripts/inspect_oxe_dataset.py`: OXE/Bridge v2 数据格式体检脚本，先用它确认 image/action/language 字段，再写正式 dataloader。

## 3. Delta Transition Tokenizer / DeltaTok-Style

**核心定义：** 不压缩整帧，也不压缩完整 clip，而是压缩两帧之间的 semantic transition。

```text
x_t, x_{t+k} -> Delta encoder -> z_delta
x_t, z_delta -> Delta decoder -> x_hat_{t+k}
```

这里 `z_delta` 只需要表达从当前 feature 到未来 feature 的变化。静态背景由 decoder 输入的 `x_t` 直接提供，不需要重复塞进 bottleneck。

**训练目标：**

```text
L = feature_mse(x_hat_{t+k}, x_{t+k})
  + optional cosine_loss
  + optional KL or deterministic bottleneck regularization
```

后续 robot world model 可以预测 delta token：

```text
history features + action chunk -> z_delta
x_t + z_delta -> x_hat_{t+k}
```

**优点：** 动态信息天然进入 bottleneck；对机器人视频中“大量静态背景 + 局部交互变化”的结构更合适；比 flow-head auxiliary loss 更干净，因为不依赖一个额外 head 去读出 motion。

**缺点：** `z_delta` 不是完整状态，decode future 必须依赖上一帧 feature；多步 rollout 会有误差累积；single-token delta 可能不足以表示多物体、多区域变化，可能需要少量 delta tokens。

**当前代码状态：已有正式 Delta tokenizer 原型，仍未接 action-conditioned predictor。**

现有最接近的是 DreamDojo-style latent-action feature model：

- `external/openpi/src/openpi/models_pytorch/latent_action.py`
  - `FeatureLatentActionModel`
  - 输入 `current_features, future_features`
  - 编码成一个低维 latent action
  - decoder 条件于 current feature 重建 future feature
- `external/openpi/scripts/train_lam_libero.py`
  - 用 `--lam-stride` 采样 future frame
  - 支持 DINOv3 / SVG-P teacher
  - 支持 feature reconstruction visualization

它和 DeltaTok 的差别：

- 当前 LAM 是 VAE-style global latent action，默认 `latent_dim=32`，不是严格的 DINO feature transition tokenizer。
- 当前 decoder 是把 latent 加到 current feature tokens 上重建 future，不包含 DeltaTok 的 staged tokenizer + world predictor 训练范式。
- 当前没有 action-conditioned predictor 去预测 `z_delta`。

新增正式 Delta tokenizer 原型：

- `external/openpi/src/openpi/models_pytorch/feature_delta_tokenizer.py`
  - `FeatureDeltaTokenizerConfig`
  - `FeatureDeltaTokenizer`
  - deterministic transition tokenizer
  - 支持 `M=1/4/8/...` 个 delta tokens
- `external/openpi/scripts/train_deltatok_libero.py`
  - 已切换到正式 `FeatureDeltaTokenizer`
  - 支持 `--num-delta-tokens`
  - 记录 `copy_mse`, `copy_ratio`, `zero_z_mse`, `shuffle_z_mse`, `delta_ratio`, `z_norm`, `z_raw_norm`, `target_delta_norm`, `z_delta_corr`

当前实现 contract：

```text
encode(x_t, x_{t+k}) -> z_delta [B, M, d]
decode(x_t, z_delta) -> x_hat_{t+k}
```

其中 `M` 可以先从 1 开始，再扩展到 4/8 个 delta tokens。

**当前 Delta smoke 观察：**

- `M=1,d=96,stride=4`、`M=4,d=96,stride=4`、`M=1,d=96,stride=8` 均能正常训练，说明 dataloader、SVG-P teacher、Delta tokenizer、反向传播和 checkpoint 链路已经跑通。
- 但 100-step 结果里 `mse` 仍高于 `copy_mse`，也就是尚未超过直接复制当前 feature 的 static-copy baseline。
- 新增诊断发现 `normal z`、`zero z`、`shuffle z` 的 future feature MSE 非常接近，说明当前 decoder 很可能还没有真正利用 `z_delta` 表达变化。
- 之前的 `z_norm` 来自 LayerNorm 后的 token，数值会天然接近 `sqrt(d)`，不能直接用来判断变化幅度。现在额外记录 `z_raw_norm` 和 `z_delta_corr`，用于检查 raw delta token magnitude 是否和 `||x_{t+k}-x_t||` 相关。

**下一步 Delta 诊断优先级：**

1. 先跑 paper-faithful / low-bottleneck 对照：`M=1,d=384` 与 `M=1,d=96`，并加 `--no-normalize-delta-tokens`，观察 `copy_ratio`、`zero_z_mse`、`shuffle_z_mse`、`z_delta_corr`。
2. 同 teacher、同 stride 下跑 DreamDojo-style LAM baseline，确认现有 `FeatureLatentActionModel` 是否比 Delta prototype 更会利用 bottleneck。
3. 如果仍然被 static-copy baseline 压制，再引入 high-motion / action-salient sampling 或 motion-weighted feature loss，而不是盲目增加 `M` 或训练步数。

尚未完成：

- action-conditioned predictor：`history features + action chunk -> z_delta`
- 与 S-VAE / PV-VAE 的同数据同 teacher 对比
- LARY-style action probe

## 对比与建议

| 方法 | 表示什么 | 时序压缩 | 当前状态 | 主要风险 |
|---|---|---:|---|---|
| Per-frame S-VAE | 单帧 semantic state | 否 | 已实现 standalone adapter，并完成 SVG-P smoke | 创新性弱，仍预测整帧 |
| PV-VAE-style | 一段 clip 的 compressed latent groups | 是，4x | 已实现并在跑 LIBERO/SVG-P | 静态复制 shortcut |
| Delta tokenizer | `x_t -> x_{t+k}` 的变化 | transition-level | 已有正式 Delta tokenizer 原型，正在做 copy/zero/shuffle 诊断 | 依赖上一帧，多步误差；目前尚未超过 static-copy baseline |

当前判断：

1. **最稳 baseline：** per-frame S-VAE / channel adapter。
2. **已实现但风险高：** PV-VAE-style temporal compression，需要 high-motion sampling 或 stride 调整。
3. **最像方法创新主线：** Delta transition tokenizer。它比 per-frame S-VAE 更有动态归纳偏置，比 PV-VAE-style 更不容易被静态背景主导。

建议下一步把 Delta tokenizer 作为主线原型：

```text
SVG-P/DINO feature pair
  -> feature delta tokenizer
  -> reconstruct future feature
  -> compare against current LAM and PV-VAE
```

评价指标先看：

- future feature MSE / cosine
- static-copy baseline gap
- `||z_delta||` 与 feature delta magnitude 的相关性
- SVG decoded-RGB / PCA visualization
- 后续再接 action-conditioned predictor。

## References

- Semantic-WM / Reconstruction or Semantics?: [arXiv 2605.06388](https://arxiv.org/pdf/2605.06388), [project page](https://hskalin.github.io/semantic-wm/)
- DeltaTok / A Frame is Worth One Token: [arXiv 2604.04913](https://arxiv.org/pdf/2604.04913)
- PV-VAE / Predictive Video VAE: [arXiv 2605.02134](https://arxiv.org/pdf/2605.02134)
- Wan / Wan-VAE: [arXiv 2503.20314](https://arxiv.org/pdf/2503.20314)
- DreamDojo-style latent action reference: [arXiv 2602.06949](https://arxiv.org/pdf/2602.06949)
- LTX-Video high-compression Video-VAE: [arXiv 2501.00103](https://arxiv.org/pdf/2501.00103)
- CogVideoX 3D VAE: [arXiv 2408.06072](https://arxiv.org/pdf/2408.06072)
- HunyuanVideo 3D VAE: [arXiv 2412.03603](https://arxiv.org/pdf/2412.03603)
