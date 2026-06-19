# PV-VAE Adapter 维度、可视化与预测机制说明

## 1. 关键表示的维度含义

### 1.1 SVG-P Feature

当前单视角、单个 clip 的 SVG-P semantic feature 形状通常是：

```text
[17, 256, 384]
```

每个维度含义：

| 维度 | 含义 |
|---:|---|
| `17` | 时间帧数，对应 `x0...x16` |
| `256` | 每帧的 spatial patch token 数，通常是 `16 x 16` |
| `384` | 每个 patch token 的 SVG-P semantic feature 维度 |

也就是说，SVG-P 不是把一帧图像 encode 成一个向量，而是 encode 成一个二维语义 token 网格：

```text
每帧 RGB -> [16, 16, 384]
```

17 帧堆起来就是：

```text
[17, 16*16, 384] = [17, 256, 384]
```

### 1.2 PV-VAE Latent

PV-VAE adapter 把 SVG-P feature clip 压缩成：

```text
[5, 256, 128]
```

每个维度含义：

| 维度 | 含义 |
|---:|---|
| `5` | temporal latent groups |
| `256` | spatial patch token 数，仍然是 `16 x 16` |
| `128` | PV-VAE 压缩后的 latent channel 维度 |

时间压缩方式：

```text
group0: x0
group1: x1  x2  x3  x4
group2: x5  x6  x7  x8
group3: x9  x10 x11 x12
group4: x13 x14 x15 x16
```

所以 PV-VAE 做了两种压缩：

```text
时间压缩: 17 frames -> 5 latent groups
通道压缩: 384 dim -> 128 dim
```

### 1.3 PV-VAE Packed Latent

为了接近 video DiT / WAM 的输入格式，单视角 PV-VAE latent 可以 reshape 成：

```text
[B, 128, 5, 16, 16]
```

每个维度含义：

| 维度 | 含义 |
|---:|---|
| `B` | batch size |
| `128` | latent channel |
| `5` | temporal latent groups |
| `16` | latent spatial height |
| `16` | latent spatial width |

它和 `[5, 256, 128]` 本质是同一份信息，只是布局不同：

```text
[5, 256, 128]
= [G, H*W, C]
= [5, 16*16, 128]

reshape / permute 后：

[C, G, H, W]
= [128, 5, 16, 16]
```

如果做双视角 horizontal packing，不建议拼 channel，而是拼 spatial width：

```text
单视角: [B, 128, 5, 16, 16]
双视角: [B, 128, 5, 16, 32]
```

这样 channel 不变，但 spatial token 数增加。

### 1.4 FastWAM / Wan2.2 VAE Latent

FastWAM 官方代码中，Wan2.2 video DiT 配置里使用：

```yaml
in_dim: 48
out_dim: 48
patch_size: [1, 2, 2]
```

因此 Wan2.2 VAE latent 通常可以理解为：

```text
[B, 48, T_latent, H_latent, W_latent]
```

每个维度含义：

| 维度 | 含义 |
|---:|---|
| `B` | batch size |
| `48` | Wan2.2 VAE latent channel |
| `T_latent` | 时间压缩后的 latent frame 数 |
| `H_latent` | 空间高度压缩后的 latent height |
| `W_latent` | 空间宽度压缩后的 latent width |

以 FastWAM LIBERO 双视角配置为例：

```text
RGB video: [B, 3, 33, 224, 448]
```

其中 `224 x 448` 来自两个 `224 x 224` 视角横向拼接。

经过 Wan2.2 VAE 后大致是：

```text
Wan latent: [B, 48, 9, 14, 28]
```

含义：

- `48`：Wan latent channel。
- `9`：33 帧经过约 4x temporal compression 后的时间 latent 数。
- `14 x 28`：`224 x 448` 经过约 16x spatial compression 后的空间 latent grid。

因此我们的 PV-VAE semantic latent 与 FastWAM / Wan2.2 latent 不能直接无缝替换：

```text
PV-VAE semantic latent: [B, 128, 5, 16, 32]
Wan2.2 VAE latent:     [B, 48,  9, 14, 28]
```

后续接 WAM 时至少需要处理：

- channel 对齐：`128 -> 48` 或修改 DiT `in_dim`。
- temporal 对齐：`5 groups` 与 `9 latent frames`。
- spatial 对齐：`16 x 32` 与 `14 x 28`。
- 多视角 packing 规则。

## 2. 可视化图每一行代表什么

当前 PV-VAE 可视化图通常按行展示如下内容。

### 第 1 行：`target_rgb`

真实 RGB 视频帧。

```text
target_rgb f0, target_rgb f1, ..., target_rgb f16
```

其中：

- 标注 `obs` 的帧是输入给 PV-VAE 的 observed frames。
- 标注 `pred` 的帧是模型需要补全的 future frames。

### 第 2 行：`gt_feat_svg_rgb`

真实 SVG-P feature decode 回 RGB 后的结果。

流程：

```text
真实 RGB
  -> SVG-P encoder
  -> SVG-P feature
  -> SVG-P decoder
  -> gt_feat_svg_rgb
```

这一行没有经过 PV-VAE。

它表示 teacher feature 本身保留了多少视觉结构，可以看作 PV-VAE 要拟合的 feature target 的可视化。

### 第 3 行：`pred_feat_svg_rgb`

PV-VAE 输出 feature decode 回 RGB 后的结果。

流程：

```text
真实 RGB
  -> SVG-P encoder
  -> SVG-P feature
  -> PV-VAE encoder
  -> PV-VAE latent
  -> PV-VAE decoder
  -> predicted / reconstructed SVG-P feature
  -> SVG-P decoder
  -> pred_feat_svg_rgb
```

这一行是模型输出的可视化。

如果 `observed_groups=3`，那么：

```text
f0...f8   是 observed frames 的重构结果
f9...f16  是 missing future frames 的补全 / 预测结果
```

注意：

```text
f0...f8 不是未来预测，但也是 PV-VAE decoder 输出。
f9...f16 才是真正的 future completion / prediction 区域。
```

### 第 4 行：`gt_feat`

真实 SVG-P feature 的 PCA / pseudo-color 可视化。

它不是 RGB 图，而是把高维 semantic feature 投影到可视化颜色空间。

作用：

- 看真实 semantic feature 的空间结构。
- 判断 SVG-P feature 是否清楚表达物体、机器人、背景等区域。

### 第 5 行：`pred_feat`

PV-VAE 输出 feature 的 PCA / pseudo-color 可视化。

作用：

- 和第 4 行 `gt_feat` 对比。
- 判断 PV-VAE 输出的 semantic feature 是否保留了结构。
- 比 `pred_feat_svg_rgb` 更直接反映 feature 空间是否对齐。

### 第 6 行：`gt_delta_ref`

真实 feature 相对最后一个 observed frame 的变化热力图。

如果 `observed_groups=3`，则：

```text
observed_frames = 1 + (3 - 1) * 4 = 9
最后一个 observed frame = f8
```

那么对每一帧 `fi`：

```text
gt_delta_i = gt_feature_i - gt_feature_f8
```

`gt_delta_ref` 可视化的是这个变化幅度。

它不是 RGB 像素差，而是 semantic feature 空间中的变化。

### 第 7 行：`pred_delta_ref`

PV-VAE 输出 feature 相对参考帧的变化热力图。

可以用来判断模型是否预测出了足够的动态幅度。

如果 `pred_delta_ref` 明显比 `gt_delta_ref` 弱，通常说明模型预测偏保守，倾向于输出变化较小的 future feature。

### 第 8 行：`abs_error`

预测 feature 与真实 feature 的误差热力图。

计算：

```text
abs_error_i = |pred_feature_i - gt_feature_i|
```

作用：

- 看错误集中在哪些空间区域。
- 看错误是否随时间 horizon 增大。
- 判断模型是整体偏色 / 平滑，还是具体动态区域预测失败。

## 3. PV-VAE 是如何预测 future feature 的

PV-VAE 的预测不是凭空生成 RGB，而是在 semantic feature 空间里做 masked / prefix reconstruction。

### 3.1 以 `observed_groups=3` 为例

完整 clip 有 5 个 latent groups：

```text
[group0, group1, group2, group3, group4]
```

其中：

```text
group0: x0
group1: x1  x2  x3  x4
group2: x5  x6  x7  x8
group3: x9  x10 x11 x12
group4: x13 x14 x15 x16
```

如果训练时设置 `observed_groups=3`，encoder 只会看到：

```text
x0...x8
```

并编码成：

```text
[z0, z1, z2]
```

其中：

- `z0` 来自 `x0`
- `z1` 来自 `x1...x4`
- `z2` 来自 `x5...x8`

### 3.2 learnable pad latent 是什么

decoder 需要输入完整长度的 latent 序列，但现在只有前三个 observed latent：

```text
[z0, z1, z2, ?, ?]
```

于是模型用 learnable pad latent 补齐：

```text
[z0, z1, z2, PAD, PAD]
```

`PAD` 是模型内部的可训练参数：

```text
pad_latent = nn.Parameter(...)
```

它不是某个样本的真实 future feature，也不是 SVG-P 编码出来的东西。

它的作用是告诉 decoder：

```text
这里缺少一个 future latent group，请结合前面的 observed latent 和时间位置信息补出来。
```

### 3.3 pad latent 是怎么训练出来的

训练过程：

```text
1. 初始化 pad_latent
2. 输入 observed latent + pad latent
3. decoder 输出完整 clip feature
4. 与真实 SVG-P feature 计算 loss
5. loss 反向传播
6. optimizer 更新 pad_latent 和 decoder/encoder 参数
```

损失函数主要包括：

```text
feature reconstruction MSE
cosine loss
temporal delta loss
KL loss / regularization
```

所以 pad latent 会通过反向传播学成一个“对 decoder 有用的缺失 future 占位符”。

### 3.4 pad latent 不是什么

需要避免一个误解：

```text
pad latent 不是样本相关的 dynamic latent。
```

它不是：

```text
未来帧 - 当前帧 的真实动态编码
```

也不是：

```text
加在原始 latent 上的 motion vector
```

更准确地说：

```text
pad latent 是所有样本共享的 missing future slot。
```

真正的 future prediction 来自三部分共同作用：

```text
1. observed latent: 当前视频 prefix 的真实语义和趋势
2. pad latent: 缺失 future group 的占位符
3. group/time positional embedding: 告诉 decoder 当前 slot 是哪个未来时间段
```

### 3.5 PV-VAE 预测能力的意义

PV-VAE 不是最终 WAM 生成器，但它的 partial prediction 能力很重要。

原因是：

- 它说明 adapter 不只是压缩静态语义。
- 它说明 latent space 中保留了一定时序结构。
- 它让 decoder 学会从 compressed semantic latent 还原 temporal feature。
- 后续 WAM / DiT 在这个 latent 空间建模 video-action dynamics 时，会更容易利用这种结构化语义 latent。

因此当前阶段评估 PV-VAE，不是为了证明它能独立完成未来视频生成，而是为了判断：

```text
这个 semantic temporal adapter 是否适合作为后续 WAM 的 latent representation。
```
