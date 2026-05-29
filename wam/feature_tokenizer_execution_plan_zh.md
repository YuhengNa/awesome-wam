# Feature Tokenizer 执行计划

本文档将当前关于 tokenizer 的讨论转化为可执行计划。
建议与 `tokenizer_methods.md`、`current_fastwam_experiments.md` 和
`research_refine.md` 一起阅读。

## 目标

为机器人世界模型构建并评估视觉特征 tokenizer。当前近期目标不是完整替换
FastWAM，而是构建一个可靠的 tokenizer 层，能够在压缩教师视觉特征的同时，
保留对动作预测和未来状态建模有用的信息。

当前研究问题是：

> 世界模型的潜空间应优先针对重建、语义、时间预测，
> 还是与动作相关的状态转移进行优化？

## 当前状态

目前有三条 tokenizer 方向在推进：

1. 逐帧 S-VAE / 通道适配器
   - 协议：`[B,V,T,N,D] -> [B,V,T,N,d]`。
   - 对每一帧独立压缩。
   - 主要用途：语义特征压缩的基线。
   - 当前缺口：在本轮之前没有独立的模型/训练脚本。

2. PV-VAE 风格的时序预测特征 VAE
   - 协议：`[B,V,1+T,N,D] -> [B,V,1+T/4,N,d]`。
   - 同时压缩通道和时间分组。
   - 已在 `predictive_feature_vae.py` 和
     `train_predictive_feature_vae_libero.py` 中实现。
   - 主要风险：模型可能学习到静态复制式重建，而不是有用的
     预测动力学。

3. DeltaTok / 转移 tokenizer
   - 协议目标：`x_t, x_{t+k} -> z_delta [B,V,M,d]`。
   - 当前实现是确定性的全局转移 token：
     `x_t, x_{t+k} -> z [B,d]`。
   - 已在 `delta_tokenizer.py` 和 `train_deltatok_libero.py` 中实现。
   - 主要用途：学习与动作显著相关的状态变化，而不是完整场景重建。

## 数据格式

训练脚本应消费来自 OpenPI LIBERO 数据路径的 batch，并将图像观测转换为教师特征张量。

图像张量：

```text
current_images: [B,V,C,H,W]
future_images:  [B,V,T,C,H,W]
image_clip:     [B,V,1+T,C,H,W]
```

教师特征张量：

```text
features:       [B,V,F,N,D]
B: batch size
V: camera views
F: frame count, usually 1 + number of future deltas
N: spatial tokens per view, for example 16 * 16
D: teacher feature width, for example 384 or 1024
d: compressed latent/token width
```

第一个可用的数据目标是 LIBERO，因为现有 OpenPI 脚本已经提供了
dataloader、相机处理和教师特征提取。更大的数据集（如 DROID、OXE、
RoboCasa 和 Behavior1K）应在 tokenizer 协议和评估脚本稳定后再加入。

## 评估

对 tokenizer 的评估应避免只依赖重建损失。

核心指标：

```text
recon_mse: 特征重建误差
cosine_loss: 语义方向保持
future_mse: 未来帧重建或预测误差
static_future_mse: 通过复制当前特征得到的误差
delta_ratio: 模型未来误差 / 静态复制误差
kl_loss: 启用 VAE 时的正则项
```

动作相关指标：

```text
LARY 风格动作探针：
  冻结 tokenizer 的 latent/features
  训练一个小型动作头来预测机器人动作
  在不同 tokenizer 方案之间比较动作损失/成功率代理指标

FastWAM 风格下游检查：
  将 tokenizer 输出接入 world/action 分支
  对比动作解码、时间预测和 rollout 质量
```

经验法则：

```text
good tokenizer != lowest reconstruction loss
good tokenizer = compact latent + semantic preservation + action-predictive signal
```

## 近期实现步骤

### 本轮已落地

1. 新增逐帧 S-VAE 模型：
   - 文件：`external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py`
   - 单帧输入：`[B,V,N,D]`
   - latent 输出：`[B,V,N,d]`
   - 重建输出：`[B,V,N,D]`

2. 新增 LIBERO 训练入口：
   - 文件：`external/openpi/scripts/train_svae_libero.py`
   - 读取 clip 特征：`[B,V,F,N,D]`
   - 训练时展平帧轴：`[B*F,V,N,D]`
   - 复用已有 PV-VAE/LAM 脚本的数据加载、SVG-P/DINO 特征提取、可视化与 checkpoint 逻辑。

3. 调整 `.gitignore`：
   - 仍默认忽略完整 `external/` 树。
   - 但白名单允许本项目实际维护的 OpenPI 实验脚本和 tokenizer 模型文件进入 Git。

1. 增加独立的逐帧 S-VAE 模型。
   - 输入：`[B,V,N,D]`。
   - 输出：重建特征 `[B,V,N,D]` 与 latent token `[B,V,N,d]`。
   - 在 clip 上训练时，将帧轴展平：`[B,V,F,N,D] -> [B*F,V,N,D]`。

2. 增加用于 S-VAE 的 LIBERO 训练脚本。
   - 复用现有 OpenPI 的 LIBERO dataloader 工具。
   - 复用 SVG-P 与 DINO 特征提取工具。
   - 记录 reconstruction、cosine、KL、latent norm、target norm。

3. 运行小规模 smoke 实验。
   - Teacher：先用 `svg_p`，因为当前脚本已经支持 SVG-P 可视化。
   - Views：`base_0_rgb`。
   - Future deltas：`1,3,6,9`。
   - Steps：100 到 500 用于健全性检查。

4. 与已有 PV-VAE 和 DeltaTok 结果对比。
   - S-VAE 回答：逐帧方式能否压缩语义？
   - PV-VAE 回答：时间分组是否有帮助？
   - DeltaTok 回答：仅转移信息是否与动作更相关？

## 建议的 Smoke 命令

请在具备 OpenPI 依赖和数据访问权限的机器上运行：

```bash
python external/openpi/scripts/train_svae_libero.py \
  --output-dir runs/svae/smoke_svgp \
  --teacher svg_p \
  --views base_0_rgb \
  --future-deltas 1,3,6,9 \
  --batch-size 8 \
  --max-steps 100 \
  --latent-dim 96 \
  --model-dim 384
```

使用 `--teacher svg_p` 时，还需要补上现有 LAM/PV-VAE 脚本使用的
SVG-P 本地路径：

```bash
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

如果这些路径还没有整理好，可以先用
`--teacher dinov3_vits16 --decode-svg-rgb false` 做第一次 smoke。

然后对比：

```bash
python external/openpi/scripts/train_predictive_feature_vae_libero.py ...
python external/openpi/scripts/train_deltatok_libero.py ...
```

## 决策标准

只有在满足以下至少一项时，才继续推进某个 tokenizer 方向：

1. 在时序指标上优于静态复制基线。
2. 在高压缩率下仍保留教师语义结构。
3. 提升冻结 latent 的动作探针表现。
4. 能无别扭 shape 转换地接入 FastWAM/OpenPI。

如果某方法只产生漂亮的重建结果，但动作探针表现较弱，
应将其视为辅助基线，而不是主研究路线。
