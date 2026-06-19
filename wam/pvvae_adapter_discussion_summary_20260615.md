# PV-VAE Adapter 训练进展与疑点总结

日期：2026-06-15

## 1. 当前研究主线

当前主线不是训练 OpenPI / VLA，而是为 WAM / FastWAM-style video-action generation 准备一个更适合机器人视频的 semantic latent encoder / adapter。

当前设想可以概括为：

```text
RGB video
  -> SVG-P / DINO semantic feature
  -> PV-VAE-style temporal adapter
  -> compressed semantic latent
  -> 后续接入 WAM / DiT video-action joint training
```

这里需要明确：

- SVG-P / DINO 是 semantic feature encoder。
- PV-VAE 不是最终生成模型，而是 temporal compression adapter。
- WAM / FastWAM 后续才负责在 compressed semantic latent 空间里做 video-action joint modeling。
- 当前训练 PV-VAE 的目标，是先证明这个 adapter 能保留语义结构、压缩时间，并具备一定 temporal reconstruction / completion 能力。

## 2. PV-VAE Adapter 的核心机制

当前使用的 PV-VAE-style adapter 输入是一段 frozen teacher feature clip：

```text
SVG-P feature: [17, 256, 384]
```

含义：

- `17`：时间帧数，`x0...x16`。
- `256`：每帧 spatial patch token 数，即 `16 x 16`。
- `384`：每个 patch token 的 SVG-P semantic feature 维度。

PV-VAE 输出 compressed latent：

```text
PV-VAE latent: [5, 256, 128]
```

含义：

- `5`：temporal latent groups。
- `256`：空间 patch token 数，仍然是 `16 x 16`。
- `128`：PV-VAE latent channel 维度。

时间分组方式：

```text
group0: x0
group1: x1  x2  x3  x4
group2: x5  x6  x7  x8
group3: x9  x10 x11 x12
group4: x13 x14 x15 x16
```

也就是说，PV-VAE 同时做了两种压缩：

```text
temporal compression: 17 frames -> 5 groups
channel compression: 384 dim -> 128 dim
```

单视角 packed latent 形式：

```text
[B, 128, 5, 16, 16]
```

含义：

- `B`：batch size。
- `128`：latent channel。
- `5`：temporal latent groups。
- `16 x 16`：spatial latent grid。

## 3. learnable pad latent 的理解

当训练时只给 PV-VAE 看 observed prefix，比如 `observed_groups=3`，那么 encoder 只会真实编码前三个 group：

```text
[z0, z1, z2]
```

但是 decoder 需要固定长度的 latent 序列，完整长度应为 5：

```text
[z0, z1, z2, ?, ?]
```

因此模型会把两个 learnable pad latent 填进去：

```text
[z0, z1, z2, PAD, PAD]
```

`PAD` 是 PV-VAE 内部的一个可训练参数，不是 SVG-P 输出，也不是某个样本的未来动态编码。

更准确的理解：

- `PAD` 是一个全局共享的缺失 future group 占位符。
- 它通过反向传播训练出来。
- 它告诉 decoder：这里缺一个 future latent group。
- 它本身不包含具体动作或具体未来。
- 真正样本相关的未来信息主要来自 observed latent；后续如果接 WAM，则应由 WAM 生成或去噪出样本相关的 future semantic latent。

因此不能把 pad latent 理解成“加在原始 latent 上的动态 latent”。它更像是 decoder 的 missing future slot。

## 4. observed frame、重构和预测的关系

以 `observed_groups=3` 为例：

```text
observed_frames = 1 + (3 - 1) * 4 = 9
```

所以：

```text
f0...f8   是 observed frames
f9...f16  是 missing future frames
```

PV-VAE 的可视化中：

- `pred_feat_svg_rgb f0...f8`：observed frames 的重构结果。
- `pred_feat_svg_rgb f9...f16`：根据 observed latent + pad latent 补出来的 future feature。

因此：

```text
f0...f8 不是未来预测，但仍然是 PV-VAE decoder 输出；
f9...f16 才是真正的 future completion / prediction 区域。
```

第二行 `gt_feat_svg_rgb` 与第三行 `pred_feat_svg_rgb` 的区别：

```text
gt_feat_svg_rgb:
  RGB -> SVG-P encoder -> SVG-P feature -> SVG-P decoder -> RGB
  这是 teacher feature 的可视化，不经过 PV-VAE。

pred_feat_svg_rgb:
  RGB -> SVG-P encoder -> SVG-P feature
      -> PV-VAE encoder -> PV-VAE latent
      -> PV-VAE decoder -> predicted/reconstructed SVG-P feature
      -> SVG-P decoder -> RGB
  这是 PV-VAE 输出 feature 的可视化。
```

一句话：

```text
第二行是答案，第三行是模型交卷。
f0...f8 看重构能力，f9...f16 看补未来能力。
```

## 5. 已完成的主要实验与观察

### 5.1 OXE / Bridge PV-VAE smoke 与 sanity check

早期 `observed_groups=1` 训练中出现过预测偏静态、可视化黑屏或纯色块等现象。后来排查发现问题来自多方面叠加：

- 数据中存在大量静态 clip 或坏视角。
- SVG-P feature channel scale 不均衡。
- 直接回归未标准化 feature 容易让模型学平均 feature。
- 单帧预测完整 future 是 hard predictive setting，不适合作为第一判断。

因此后续改为更系统的 sanity check：

1. 统计 SVG-P feature mean / std / norm。
2. 加 channel-wise feature normalization。
3. 做 full-observed autoencoding。
4. 做 few-clip overfit。
5. 再做 partial observed prediction。

### 5.2 SVG-P feature stats

OXE / Bridge quality subset 上的 SVG-P feature 统计显示：

```text
channel_std_max_over_min ≈ 7.75
token_norm_mean ≈ 8.41
```

说明不同 feature channel 尺度差异明显，因此使用 `channel_standard` normalization 是合理且必要的。

训练时实际逻辑：

```text
SVG-P feature -> subtract channel mean -> divide channel std -> PV-VAE
```

可视化或 SVG decode 前再 unnormalize 回原始 SVG-P feature 空间。

### 5.3 full-observed / few-clip overfit

full-observed autoencoding 和 few-clip overfit 的结果证明：

- PV-VAE 结构本身能重构 SVG-P feature。
- 模型不是完全没有 capacity。
- 早期纯色块问题不是模型必然失败，而是训练设置、normalization、数据筛选和 observed setting 的组合问题。

few-clip overfit 中可视化明显变好，说明模型能够学习结构化 semantic feature。

### 5.4 OXE main-view mixed baseline 20k

20k 训练后，PV-VAE adapter 已经可以在可视化中看到清晰结构：

- `gt_feat` 和 `pred_feat` 的 PCA feature map 有明显对应。
- `pred_feat_svg_rgb` 能重构 / 补出大体场景结构。
- observed 部分重构较好。
- future 部分前几帧较好，越往后质量越下降。

一个典型可视化标题：

```text
feature_mse=0.01306
pred_d=0.888
gt_d=1.868
ratio=0.476
observed_frames=9
```

这里：

```text
ratio = pred_d / gt_d = 0.888 / 1.868 ≈ 0.476
```

含义是：预测 feature 的动态幅度约为真实动态幅度的 47.6%。这说明模型不是完全静止复制，但动态预测仍偏保守。

## 6. 当前可视化每一行的含义

以 `observed_groups=3` 的可视化为例，一张图通常有以下几行：

1. `target_rgb`
   - 原始真实 RGB。
   - 标注 `obs` 的帧是输入给 PV-VAE 的 observed frames。
   - 标注 `pred` 的帧是模型需要补出的 future frames。

2. `gt_feat_svg_rgb`
   - 真实 SVG-P feature 通过 SVG decoder decode 回 RGB。
   - 不经过 PV-VAE。
   - 表示 teacher feature 自身保留了多少视觉结构。

3. `pred_feat_svg_rgb`
   - PV-VAE 输出 feature 再通过 SVG decoder decode 回 RGB。
   - observed 部分是重构。
   - future 部分是补全 / 预测。

4. `gt_feat`
   - 真实 SVG-P feature 的 PCA / pseudo-color 可视化。

5. `pred_feat`
   - PV-VAE 输出 feature 的 PCA / pseudo-color 可视化。

6. `gt_delta_ref`
   - 真实 feature 相对最后一个 observed frame 的变化热力图。
   - 如果 `observed_frames=9`，参考帧就是 `f8`。

7. `pred_delta_ref`
   - PV-VAE 预测 feature 相对参考帧的变化热力图。

8. `abs_error`
   - `|pred_feature - gt_feature|` 的误差热力图。

其中 `gt_delta_ref` 的含义是：

```text
delta_i = feature_i - feature_last_observed
```

它不是 RGB 像素差，而是 semantic feature 空间中的变化。

## 7. FastWAM 多视角代码检查结论

FastWAM 不是永远双视角，而是每个数据配置固定一个视角布局。

### 7.1 LIBERO 配置

文件：

```text
E:/awesome_wam/FastWAM-main/configs/data/libero_2cam.yaml
```

关键信息：

```yaml
images:
  - key: image
  - key: wrist_image
concat_multi_camera: "horizontal"
num_output_cameras: 2
video_size: [224, 448]
```

说明 LIBERO 是双视角，横向拼接：

```text
224 x 224 + 224 x 224 -> 224 x 448
```

### 7.2 RoboTwin 配置

文件：

```text
E:/awesome_wam/FastWAM-main/configs/data/robotwin.yaml
```

关键信息：

```yaml
images:
  - key: cam_high
  - key: cam_left_wrist
  - key: cam_right_wrist
concat_multi_camera: "robotwin"
num_output_cameras: 3
video_size: [384, 320]
```

说明 RoboTwin 是三视角，并使用固定的 `robotwin` 布局：

```text
top: cam_high
bottom: cam_left_wrist + cam_right_wrist
```

### 7.3 缺视角逻辑

在 `FastWAMProcessor` 里：

- 如果实际相机数少于 `num_output_cameras`，会补零。
- 如果实际相机数多于 `num_output_cameras`，会截断。

在 `RobotVideoDataset` 里：

- `horizontal` 会把多个相机沿 width 拼接。
- `vertical` 会沿 height 拼接。
- `robotwin` 要求正好 3 个相机，否则报错。

因此 FastWAM 的原始思路是 fixed layout，不是任意数量视角动态输入。

对我们后续 multi-view semantic latent 接入的启发：

- 应保持固定 view order。
- 缺失视角不能随便混入坏视频。
- 可以选择过滤、补零、加 mask，或者分 view-layout 训练。
- semantic latent 的多视角拼接应优先沿 spatial dimension 拼，而不是 channel 拼。

## 8. 与 WAM 接入相关的维度对齐问题

当前单视角 PV-VAE latent：

```text
[B, 128, 5, 16, 16]
```

双视角 horizontal packing：

```text
[B, 128, 5, 16, 32]
```

这会使 spatial token 数翻倍，但 channel 不变。

FastWAM / Wan2.2 当前 video DiT 配置：

```yaml
in_dim: 48
out_dim: 48
patch_size: [1, 2, 2]
```

Wan2.2 VAE latent 通常是：

```text
[B, 48, T_latent, H_latent, W_latent]
```

LIBERO 双视角 RGB 是 `224 x 448`，Wan latent 大致是：

```text
[B, 48, 9, 14, 28]
```

而我们的 semantic latent 是：

```text
[B, 128, 5, 16, 32]
```

因此不能直接塞进原 FastWAM DiT，需要解决：

- channel projection：`128 -> 48` 或改 DiT `in_dim`。
- temporal layout：`5 groups` vs Wan latent 的 `9` temporal tokens。
- spatial layout：`16 x 32` vs `14 x 28`。
- 多视角 packing 规则。

这部分是后续接 WAM 的核心工程问题。

## 9. 数据集与视角审计进展

已审计数据集：

```text
OXE:
  /data/user/jhe724/workspace/data/OXE

DROID:
  /data/user/jhe724/workspace/data/droid_success

Behavior 1K:
  /data/user/jhe724/workspace/data/2025-challenge-demos
```

第一版 main-view source：

```json
[
  {
    "name": "oxe_bridge_image0",
    "video_keys": ["observation.images.image_0"],
    "fps": 5,
    "role": "external_primary"
  },
  {
    "name": "oxe_fractal_image",
    "video_keys": ["observation.images.image"],
    "fps": 3,
    "role": "external_primary"
  },
  {
    "name": "behavior_rgb_head",
    "video_keys": ["observation.images.rgb.head"],
    "fps": 30,
    "role": "head_primary"
  },
  {
    "name": "droid_left_external",
    "video_keys": ["observation.images.left_external"],
    "fps": 15,
    "role": "external_primary"
  }
]
```

当前判断：

- OXE Bridge `image_0` 是较可靠主视角。
- OXE Bridge `image_1/2/3` 有大量占位/黑屏/坏视角，不适合作为 main-view baseline。
- Behavior `rgb.head` 可作为主视角。
- DROID 当前下载版本主要可用的是 `left_external`。

## 10. 当前关键疑点

### 10.1 DROID shard 对齐问题

服务器智能体发现 DROID 的某些 parquet 文件包含多个 episode，而对应视频文件不是一一 episode 视频。

出现过类似问题：

```text
parquet rows 很多，但视频 frames 少很多
```

服务器端临时修复是：如果 video frames 少于 parquet rows，就按 video frames 限制采样长度。

这个修复可以避免 out-of-range，但是否科学还需要验证，因为可能存在：

- parquet 内多个 episode 与视频 shard 的复杂对齐。
- 不能简单把整段 shard 当作单个 episode。
- 需要确认 `episode_index`、`frame_index`、video frame index 是否一致。

因此 multi-dataset 正式训练前，DROID 对齐仍需 sanity check。

### 10.2 PV-VAE 的预测能力是否重要

当前结论：

PV-VAE 本身不是最终 WAM predictor，但它的 partial prediction / completion 能力仍然重要。

原因：

- 它说明 adapter latent 不是只会静态压缩。
- 它说明 temporal group latent 中有一定动态结构。
- advisor 也认为带时序语义的 VAE 可能比普通 VAE 更适合视频生成。
- 未来接 WAM 时，DiT 在这个 latent 空间建模动作和视频变化，latent 空间如果太静态或太平均，会影响后续生成。

但需要避免误解：

```text
PV-VAE 的 future completion 能力不是最终目标；
它是 tokenizer / adapter 质量的诊断指标。
```

### 10.3 当前 future 预测偏保守

已有可视化显示：

- 前几帧 future 预测较好。
- 越往后质量下降。
- `pred_delta_norm / gt_delta_norm` 低于 1，说明动态幅度不足。

这支持后续做：

- fixed `observed_groups=4` ablation。
- 更大数据集 main-view training。
- delta_weight / future_loss_weight 对照。
- held-out eval matrix。

## 11. 下一步建议

短期不要立刻跳到完整 WAM rollout，而是先把 adapter 主线补齐：

1. 确认 DROID shard 读取逻辑是否正确。
2. 跑 multi-dataset main-view PV-VAE baseline。
3. 做 fixed `observed_groups=4` ablation。
4. 做 held-out eval matrix：

```text
checkpoint x observed_groups=3/4/5
```

5. 分析：

- feature MSE
- pred_delta_norm / gt_delta_norm
- horizon-wise error
- SVG decoded RGB visualization
- PCA feature visualization

6. 再进入 multi-view adapter 与 FastWAM-style latent layout 对接。

## 12. 当前阶段的一句话总结

目前已经证明 PV-VAE adapter 在 OXE main-view 上不是纯失败：它能重构 SVG-P semantic feature，也能补出部分 future semantic structure；但动态幅度仍偏保守，多数据集训练和固定 partial prediction ablation 还没完成。下一阶段的关键是把 main-view semantic adapter 从 OXE 扩展到 OXE + DROID + Behavior，并用 held-out eval 判断它是否稳定泛化，然后再讨论 multi-view packing 和 WAM 接入。
