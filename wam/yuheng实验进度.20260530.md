# Awesome-WAM / Feature Tokenizer 当前研究计划与进度总结
# （重点看第三部分当前总体进度）

## 1. 总体研究目标

当前项目围绕机器人 world model 中的视觉 latent / tokenizer 展开。

核心问题是：

```text
什么样的视觉 latent space 对机器人世界模型和动作预测最有用？
是重构像素/特征更重要，还是语义、更紧凑的压缩、更动作相关的状态转移更重要？
```

我们目前不是直接做完整新架构，而是在 FastWAM / OpenPI 体系下，研究不同视觉表示和 tokenizer 对 world model / action learning 的影响。

整体 pipeline 可以理解为：

```text
RGB observation
 -> frozen visual teacher encoder
    例如 SVG-P / DINOv3 / SigLIP
 -> feature tokenizer / adapter / VAE
 -> compressed latent tokens
 -> world model / action model 使用这些 latent
 -> 评测未来预测、动作预测、下游 LIBERO 成功率
```

## 2. 当前 tokenizer 三条路线

根据 `tokenizer_methods.md`，当前 tokenizer 研究有三条路线。

### 2.1 Per-frame S-VAE / Channel Adapter

核心定义：

```text
RGB_t -> frozen encoder -> x_t [V, N, D]
x_t -> S-VAE encoder -> z_t [V, N, d]
z_t -> S-VAE decoder -> x_hat_t
```

其中：

```text
V = camera views
N = spatial tokens
D = teacher feature dim，例如 384
d = 压缩后的 latent dim，例如 64/96/128
```

特点：

```text
每一帧独立压缩
不做时序压缩
不显式建模 transition
主要作为最稳的 semantic feature compression baseline
```

训练目标：

```text
loss = feature_mse(x_hat_t, x_t)
     + cosine_loss(x_hat_t, x_t)
     + beta * KL
```

研究意义：

```text
验证：只做逐帧语义特征压缩，能否保留 SVG-P / DINO 等 teacher feature 的语义信息。
```

它对应师兄说的：

```text
训练一个 adapter
接入高维特征
把大规模数据整理成同样格式后训练 tokenizer / adapter
```

### 2.2 PV-VAE-style Temporal Predictive Feature VAE

核心定义：

```text
x_0, x_1, ..., x_16
 -> encoder
 -> z_0, z_1, ..., z_4
 -> decoder
 -> x_hat_0, ..., x_hat_16
```

特点：

```text
做 temporal compression
例如 17 frames -> 5 latent groups
第 0 帧单独成组
future frames 每 4 帧压成一个 latent group
```

当前状态：

```text
已实现
已有 train_predictive_feature_vae_libero.py
已有 LIBERO / SVG-P 相关实验经验
```

主要风险：

```text
LIBERO 中静态背景占比很高
predictive objective 可能学到 static-copy shortcut
即复制当前帧就能得到较低 future MSE
```

作用：

```text
作为时序压缩 baseline
用于比较 temporal compression 是否真的带来有效动态建模
```

### 2.3 Delta Transition Tokenizer / DeltaTok-style

核心定义：

```text
x_t, x_{t+k} -> Delta encoder -> z_delta
x_t, z_delta -> Delta decoder -> x_hat_{t+k}
```

特点：

```text
不压缩完整状态
不压缩完整视频 clip
只压缩从当前 feature 到未来 feature 的 semantic transition
```

优势：

```text
静态背景由 x_t 直接提供
z_delta 更容易被迫表示动态变化
更适合机器人视频中“静态背景 + 局部交互变化”的结构
```

当前状态：

```text
已有 LAM / Delta 原型
但正式 DeltaTok 还未完整实现
```

研究判断：

```text
S-VAE 是最稳 baseline
PV-VAE 已实现但有 static shortcut 风险
DeltaTok 是最像方法创新主线的方向
```

## 3. 当前总体进度

### 3.1 已完成：S-VAE baseline 代码实现

本地已经新增：

```text
external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
external/openpi/scripts/train_svae_libero.py
```

实现的 contract：

```text
input:  [B, V, N, D]
latent: [B, V, N, d]
output: [B, V, N, D]
```

训练时对 video clip 的处理：

```text
features [B, V, F, N, D]
 -> flatten frame axis
 -> [B*F, V, N, D]
 -> S-VAE training
```

也就是说，虽然 dataloader 取了 current + future frames，但 S-VAE 本身仍然是逐帧训练，不做时间建模。

### 3.2 已完成：服务器环境打通

为了跑 OpenPI / LIBERO / SVG-P smoke，解决了以下问题：

```text
1. openpi import path
2. etils / flax / tyro / lerobot / augmax / beartype / jaxtyping 等依赖
3. Python 3.10 兼容 datetime.UTC 问题
4. datasets 与 lerobot 版本兼容问题
5. 本地 LIBERO LeRobot 数据路径问题
6. SVG-P autoencoder root/config/checkpoint/DINO weights 路径问题
```

最终使用环境：

```text
conda env: fastwam
Python: 3.10.20
PyTorch: 2.7.1+cu128
CUDA available: True
```

### 3.3 已完成：本地 LIBERO no-noops 数据 schema 适配

原始 OpenPI 默认期望：

```text
actions
image
wrist_image
state
```

本地 LeRobot 数据实际字段：

```text
action
observation.images.image
observation.images.wrist_image
observation.state
```

因此临时 patch 了 config，使其适配本地数据：

```text
actions -> action
image -> observation.images.image
wrist_image -> observation.images.wrist_image
state -> observation.state
```

注意：这是临时 schema patch。后续应该整理成正式参数或单独 config，而不是长期手改 `config.py`。

## 4. 当前已跑通的实验：S-VAE + SVG-P Smoke

### 4.1 实验命令

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/train_svae_libero.py \
  --output-dir runs/svae/smoke_svgp \
  --teacher svg_p \
  --views base_0_rgb \
  --future-deltas 1,3,6,9 \
  --batch-size 8 \
  --max-steps 100 \
  --latent-dim 96 \
  --model-dim 384 \
  --svg-autoencoder-root /data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/external/SVG-T2I/autoencoder \
  --svg-config /data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/external/SVG-T2I/svg_t2i/pre-trained/autoencoder/svg_autoencoder_P_stage1_256.yaml \
  --svg-checkpoint /data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/svg_t2i/pre-trained/autoencoder/svg_autoencoder-P-stage1.ckpt \
  --svg-dinov3-weights /data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/svg_t2i/pre-trained/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

### 4.2 实验配置

```text
dataset: LIBERO_10 no-noops local LeRobot data
teacher: SVG-P
view: base_0_rgb
future_deltas: 1,3,6,9
batch_size: 8
max_steps: 100
latent_dim: 96
model_dim: 384
S-VAE params: about 15.0M
```

### 4.3 日志结果

```text
step=20  loss=0.433202 recon=0.375942 cos=0.572579 kl=2.565940
step=40  loss=0.163203 recon=0.129262 cos=0.339363 kl=5.348406
step=60  loss=0.126011 recon=0.098272 cos=0.277317 kl=6.779077
step=80  loss=0.103454 recon=0.079952 cos=0.234949 kl=7.290751
step=100 loss=0.092608 recon=0.070972 cos=0.216286 kl=7.525980
```

### 4.4 结果含义

这不是正式结论，只是 smoke test。

它说明：

```text
1. LIBERO 本地数据可以被读取
2. SVG-P teacher 可以成功加载并提特征
3. S-VAE tokenizer shape 正确
4. loss 可以正常计算
5. backward / optimizer 正常
6. 没有 NaN
7. 100 steps 内 loss 明显下降
```

具体 loss 含义：

```text
loss = S-VAE adapter/tokenizer 自己的训练 loss
recon = x_hat_t 和 SVG-P teacher feature x_t 的 MSE
cos = x_hat_t 和 x_t 的 cosine direction loss
kl = VAE latent regularization
```

这里的 loss 不是：

```text
不是 FastWAM action loss
不是 LIBERO success rate
不是 world model rollout loss
```

它只代表：

```text
S-VAE 可以学习压缩并重建 SVG-P feature
```

## 5. 当前结论

目前可以确认：

```text
Per-frame S-VAE / Channel Adapter baseline 已经具备可训练闭环。
```

更具体：

```text
文档状态：尚未实现 standalone S-VAE
当前状态：已实现 + SVG-P smoke 跑通
```

所以 S-VAE 这条 baseline 已经从“计划”进入“可正式扩展训练”的阶段。

但是：

```text
100 steps smoke 不足以证明方法有效
不能说明下游机器人控制会提升
不能说明优于 PV-VAE 或 DeltaTok
```

它只证明工程链路打通。

## 6. 与师兄“大规模数据训练 adapter”要求的关系

师兄说：

```text
之前只是在小规模数据集上实验
现在要准备更大规模的数据
整理成同样格式
训练 adapter
```

这对应当前路线里的：

```text
Per-frame S-VAE / Channel Adapter
```

我们刚刚完成的是：

```text
LIBERO_10 no-noops 上的最小 smoke
```

还没有完成：

```text
大规模数据整理
多数据集统一格式
长时间训练 adapter
```

下一阶段应该把以下数据整理为统一格式：

```text
LIBERO_10 no-noops
LIBERO_90 no-noops
LIBERO spatial/object/goal
后续 DROID / OXE / RoboCasa / Behavior1K 等
```

目标 schema：

```text
observation.images.image
observation.images.wrist_image
observation.state
action
timestamp
frame_index
episode_index
task_index
```

训练目标：

```text
用 SVG-P teacher 提取 feature
训练 S-VAE adapter/tokenizer
比较 latent_dim = 64 / 96 / 128
```

## 7. 下一步计划

### Step 1：把临时 schema patch 正式化

现在为了跑通 smoke，手改了 OpenPI `config.py`。后续应改成更干净的方式，例如：

```text
--libero-schema local_no_noops
```

或：

```text
--action-key action
--image-key observation.images.image
--wrist-image-key observation.images.wrist_image
--state-key observation.state
```

目的：

```text
避免每次换数据都手动改 config.py
支持不同 LIBERO / LeRobot 数据格式
```

### Step 2：S-VAE 正式小规模训练

在 smoke 成功后，可以跑更长一些：

```text
steps: 3k / 10k
dataset: LIBERO_10 no-noops
teacher: SVG-P
latent_dim: 64 / 96 / 128
```

观察：

```text
feature MSE
cosine loss
SVG decoded visualization
checkpoint
训练稳定性
```

### Step 3：整理大规模 adapter 数据

对应师兄要求：

```text
把更多数据整理成同样格式
统一 dataloader
用同一个 S-VAE / adapter 训练
```

需要整理的数据候选：

```text
LIBERO_90 no-noops
LIBERO spatial/object/goal
RoboCasa
DROID
OXE
Behavior1K
```

### Step 4：回到 tokenizer_methods.md 的主线，推进 DeltaTok

S-VAE 是 baseline，不是最主要创新。

文档判断中最像主线的是：

```text
Delta Transition Tokenizer
```

下一步需要实现/整理：

```text
feature_delta_tokenizer.py
train_feature_delta_tokenizer_libero.py
```

目标 contract：

```text
encode(x_t, x_{t+k}) -> z_delta [B, M, d]
decode(x_t, z_delta) -> x_hat_{t+k}
```

并与：

```text
S-VAE
PV-VAE
DeltaTok
```

统一比较。

### Step 5：统一评测

后续三条 tokenizer 应该统一评测：

```text
1. feature reconstruction MSE
2. cosine similarity
3. static-copy baseline gap
4. delta_ratio / transition magnitude
5. SVG decoded visualization
6. LARY-style action probe
7. 最终 FastWAM / OpenPI downstream action eval
```

## 8. 当前一句话状态

当前项目状态可以概括为：

```text
S-VAE / Channel Adapter baseline 已实现并在 LIBERO + SVG-P 上完成 100-step smoke，证明数据、teacher、tokenizer、训练闭环已经打通。下一步应将数据 schema 适配正式化，然后进行更长的 adapter 训练和大规模数据整理；同时按 tokenizer_methods.md 回到主线，推进 Delta Transition Tokenizer，并最终做 S-VAE / PV-VAE / DeltaTok 的统一比较。
```