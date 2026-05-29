# WAM Project Plan

## 总体架构

### 核心抽象层

项目围绕 WAM 的四个设计维度，设计四个可插拔的核心模块：

```
输入 (images, instruction, actions)
    │
    ├── WorldEncoder          ← 维度一：表征空间
    │     ├── VAEEncoder         (VAE latent, 可解码回像素)
    │     ├── DINOEncoder        (DINO latent, 语义结构特征)
    │     └── MultiModalEncoder  (多模态结构: 深度/光流/分割...)
    │
    ├── WorldBranch           ← 维度二 + 三：生成时机 + 信息流拓扑
    │     ├── TrainOnlyBranch    (仅训练时生成, 推理时跳过)
    │     ├── JointBranch        (联合去噪, 双向注意力)
    │     ├── CausalBranch       (因果: 先生成未来, 再条件化动作)
    │     ├── DecoupledBranch    (解耦: 动作看不到未来, 共享骨干)
    │     └── EncoderOnlyBranch  (无视频生成, 仅特征提取, 兼容 WM4A)
    │
    ├── DataStrategy          ← 维度四：数据利用策略
    │     ├── UniformStrategy    (所有数据同等对待)
    │     └── QualityAwareStrategy (按质量分级分配学习目标)
    │
    ├── Backbone              ← 视频 DiT 骨干
    │     ├── Wan22             (Alibaba Wan2.2, 5B)
    │     ├── CosmoPredict2    (NVIDIA Cosmos, 2B)
    │     └── ...              (可扩展)
    │
    └── ActionDecoder         ← 动作解码头 (与 StarVLA 兼容)
          ├── OFT              (MLP 回归)
          ├── GR00T            (Flow Matching)
          └── PI               (Layer-wise Cross-Attention DiT)
```

### 与 StarVLA 的代码关系

```
StarVLA (fork 源)
    └── starVLA/model/framework/WM4A/    ← 保留为兼容参考

本项目 (fork 后重构)
    ├── core/                             ← 全新核心抽象
    │     ├── world_encoder/
    │     ├── world_branch/
    │     ├── attention_topology/
    │     ├── data_strategy/
    │     └── backbone/
    ├── action_decoder/                   ← 保留 StarVLA 接口兼容
    ├── data/                             ← 保留 StarVLA 数据加载格式
    ├── eval/                             ← 保留 + 扩展评测
    ├── configs/                          ← 全新配置体系
    └── docs/
          ├── story.md
          └── plan.md
```

---

## 实施阶段

### Phase 0: 基础搭建

**目标：** Fork StarVLA，搭建项目骨架，建立两条对照基线（外部参考 + 内部起点）。

**具体任务：**
- [ ] Fork StarVLA starVLA_dev 分支，重命名项目，建立独立 repo 结构
- [ ] 建立基本的配置体系（支持通过 config 切换模块）
- [ ] **外部参考基线（冻结骨干）：** 复现 WM4A 的 CosmoPredict2OFT / WanOFT，用于验证数据管线和评测环境。退出标准：LIBERO 各 suite 成功率与 WM4A 原报告相差 ≤ 2 pp
- [ ] **内部起点基线（可训练骨干）：** 最简 EncoderOnly 配置（无视频生成 + 可训练骨干），作为 Phase 1+ 所有消融的对照原点

**产出：** 两条基线同时跑通。外部参考证明基础设施无缺陷；内部起点为后续所有实验提供公平对照。

### Phase 1: 世界模型表征空间

**目标：** 做一个类似 `Reconstruction or Semantics?` / LARY 的表征空间研究：在尽量固定数据、动作头、时间跨度和优化协议的前提下，比较不同 world prediction target 与机器人动作表现之间的关系，并进一步检查该关系是否随 FastWAM 推理模式变化。

**当前 FastWAM-DINOv3 baseline 计划：** 见 [dinov3_fastwam_plan.md](dinov3_fastwam_plan.md)。第一阶段仅将 world prediction target 从 Wan VAE latent 替换为 frozen DINOv3 patch features，保持 FastWAM 原始 `action_conditioned=false` 与 action branch/attention topology 不变。

**当前 FastWAM 实验状态：** 见 [current_fastwam_experiments.md](current_fastwam_experiments.md)。该文档记录正在运行的 RGB/depth/seg/DINO/SigLIP/V-JEPA/SVG 表征实验、10 fps 数据时间对齐、HPC job、eval 视频路径和最新 loss。

**当前 FastWAM 代码导读：** 见 [fastwam_codebase_reading_guide.md](fastwam_codebase_reading_guide.md)。该文档解释 config、dataset、feature encoder、FastWAM/FastWAMDINO、trainer 和 LIBERO eval 的主链路。

**当前 novelty check：** 见 [novelty_check.md](novelty_check.md)。结论是我们不应把工作表述为“用 DINO 替代 VAE”，而应定位为 FastWAM train/inference regimes 下的表征目标系统研究，风格接近 `Reconstruction or Semantics?` 和 LARY。

**当前 research refine：** 见 [research_refine.md](research_refine.md)。该文档冻结问题锚点、表征 taxonomy、核心假设和最小验证块，下一步应进入 claim-driven `experiment-plan`。

**当前 method idea A：** 见 [action_salient_world_modeling.md](action_salient_world_modeling.md)。该文档提出 `Interaction-Saliency World Loss`：先训练 transition IDM 或 interaction probe 估计 token-level saliency，再用 saliency map 对 WAM 的 world loss 做加权，避免 uniform future prediction 过度优化交互无关区域。

**当前 method idea B：** 见 [interaction_aware_latent_action.md](interaction_aware_latent_action.md)。该文档提出 `Interaction-Aware Latent Action`：从视觉 transition + instruction 中学习 latent action tokens，同时预测低层 action、目标物体、接触和任务进展，用于诊断或生成 interaction saliency。

**当前 method idea C：** 见 [pi05_native_denoising_gen_expert.md](pi05_native_denoising_gen_expert.md)。该文档提出 `Pi0.5-Native Denoising Generation Expert`：在 pi0.5/OpenPI 风格 VLA 中加入第三个随机初始化的 generation expert，用训练期 future representation denoising 约束 backbone，但推理时保持 action-only。

**当前 tokenizer 方案对比：** 见 [tokenizer_methods.md](tokenizer_methods.md)。该文档整理 per-frame S-VAE/channel adapter、PV-VAE-style temporal predictive tokenizer、Delta transition tokenizer 三条路线，并标注当前 OpenPI/FastWAM 代码中的实现状态。

**当前 Pi0.5 method benchmark：** LIBERO。第一版实现以 OpenPI `pi05_libero` 为 action-only baseline，使用官方 `physical-intelligence/libero` 数据、两路相机 `image`/`wrist_image`、`action_horizon=10`，新增 `pi05_libero_gen` 训练配置。该配置额外读取 4 个 future RGB target frames，并用 frozen pi0.5/PaliGemma SigLIP teacher 生成训练期 future visual-token denoising 目标。

**具体任务：**
- [ ] 实现 DINOEncoder（基于 DINOv2，参考 LDA-1B）
- [ ] 实现 VAEEncoder（复用 Wan2.2 / Cosmos 的 VAE）
- [ ] 统一 WorldEncoder 接口，确保下游模块对表征空间无感知
- [ ] 在 LIBERO 上系统比较 RGB/depth/seg/DINO/SigLIP/V-JEPA/SVG 表征目标的动作预测性能
- [ ] 验证 Interaction-Saliency World Loss：训练 transition IDM/interaction probe，比较 uniform、motion-weighted、interaction-saliency-weighted world loss
- [ ] 验证 Interaction-Aware Latent Action：比较 plain IDM 与 interaction-aware latent action 的 attention map、object/contact overlap 和 action recoverability
- [ ] 验证 Pi0.5-native denoising gen expert：在 OpenPI/pi0.5 上加入训练期 future visual token loss，比较 action-only baseline 与 random-init 三塔 MoT 的控制表现和鲁棒性
- [ ] （可选）实现 MultiModalEncoder，支持深度/光流等结构模态

**关键实验：**
- reconstruction/structural/semantic feature targets 与 action success/action error 的相关性
- uniform world loss、动态区域加权 loss、IDM action-saliency 加权 loss 的对照
- 验证 `Reconstruction or Semantics?` / LARY 风格结论是否在 FastWAM 的 decoupled、joint、causal/IDM 推理模式下成立或发生变化

### Phase 1a: Depth / Seg Future Target 快速变体

**目标：** 在不改动 FastWAM 模型主体的前提下，先验证结构模态监督是否能提升动作预测。第一版采用"RGB 条件帧 + 未来结构模态帧"的最小设计。

**设计：**
- 原始 FastWAM baseline 使用 20 fps 数据、33 个原始 step、32 个 action、9 个视频帧，覆盖 1.6s。
- 当前 aligned LIBERO mask/depth 数据为 10 fps，因此使用 `num_frames=17` 与 16 个 action step，同样覆盖 1.6s。
- 第 0 帧仍使用 RGB，作为动作分支的视觉条件。
- RGB/depth/seg/DINO/SigLIP 的 aligned 设置使用 1 个 condition frame + 8 个 future frames，future frame 间隔 0.2s。
- SVG Res-ViT 当前长训采用 future4 设置：1 个 condition frame + 4 个 future frames，future frame 间隔 0.4s，总跨度仍为 1.6s。
- depth/seg 暂时复用 Wan2.2 VAE，把它们视作 3 通道图像建模。

**数据要求：**
- LeRobot 数据中需要提供 RGB、depth、seg 三类 image keys。
- LIBERO 2-camera 默认 key 约定：
  - RGB: `image`, `wrist_image`
  - Depth: `depth`, `wrist_depth`
  - Seg: `seg`, `wrist_seg`
- depth 需归一化并编码为 `uint8 [3,H,W]`；seg 需用固定 palette 编码为 `uint8 [3,H,W]`。

**实现任务：**
- [x] 在 FastWAM 数据层增加 `condition_image_keys` 与 `target_image_keys`，支持第 0 帧来自 RGB、future frames 来自 depth/seg。
- [x] 为 segmentation 增加 nearest resize 路径，避免 palette 被插值污染。
- [x] 新增 `libero_depth_2cam` 与 `libero_seg_2cam` 数据配置。
- [x] 新增 `libero_depth_2cam224_1e-4` 与 `libero_seg_2cam224_1e-4` 训练任务配置。
- [x] 新增静态配置检查脚本，确认 video frames、future loss frames、action horizon 正确。
- [x] 在带真实 depth/seg 数据的机器上运行样本可视化，检查 batch shape、depth 归一化、seg palette 和 VAE reconstruction。
- [x] 启动 aligned RGB、Depth target、Seg target smoke/long-run 对照训练。
- [x] 启动 DINOv3/SigLIP2/SVG Res-ViT feature target 对照训练；SVG 支持 PCA 与 decoded-RGB eval 可视化。

**验证命令：**

```bash
cd external/FastWAM
python scripts/verify_modality_config.py --task libero_depth_2cam224_1e-4
python scripts/verify_modality_config.py --task libero_seg_2cam224_1e-4
python scripts/verify_modality_config.py --task libero_seg_2cam224_1e-4 --load-sample
```

**退出标准：**
- depth/seg 配置能正常 compose、加载样本，并在当前 aligned 设置下输出 2-camera video 与 `action=[16,D]`。
- 训练 loss 正常下降，无 VAE shape、dtype、palette resize 错误。
- 在 LIBERO 上比较成功率、action L1/L2、future target 重建指标，判断结构模态是否值得进入正式 `MultiModalEncoder` 设计。

### Phase 2: 世界模型分支与注意力拓扑

**目标：** 实现 WorldBranch 模块，支持不同的生成时机和信息流拓扑。

> **说明：** 视频 DiT 骨干从 Phase 0 的内部起点基线起就是可训练的，这是本项目与 WM4A 最本质的区别。本阶段的核心是在可训练骨干之上引入视频生成分支与注意力拓扑。

**具体任务：**
- [ ] 实现视频分支的 Flow Matching 训练目标（骨干与视频分支联合优化）
- [ ] 实现可配置的注意力掩码系统
  - [ ] Decoupled: 动作 tokens 看不到未来视频 tokens
  - [ ] Joint: 动作与视频 tokens 双向可见
  - [ ] Causal: 视频先去噪，动作条件化于视频
- [ ] 实现训练/推理生成时机的配置化
  - [ ] TrainOnly 模式：训练时有视频损失，推理时跳过视频分支
  - [ ] Full 模式：训练和推理都生成
  - [ ] EncoderOnly 模式：无视频生成分支；骨干冻结与否由 `backbone.freeze` 独立控制（冻结版即 WM4A 兼容基线）

**关键实验：**
- 复现 Fast-WAM 的核心消融：训练时视频联合训练 vs 不联合训练的性能差异
- 三种注意力拓扑（联合/因果/解耦）的公平对比
- 推理时生成 vs 不生成的速度-性能权衡

### Phase 3: 数据质量分级训练

**目标：** 实现 DataStrategy 模块，支持异质数据的质量感知训练。

**具体任务：**
- [ ] 实现数据质量标注接口（支持高质量/低质量/无动作三级）
- [ ] 实现 QualityAwareStrategy
  - [ ] 高质量数据：策略 + 动力学 + 视觉预测（全目标）
  - [ ] 低质量数据：仅动力学 + 视觉预测
  - [ ] 无动作视频：仅视觉预测
- [ ] 实现 task embedding 路由机制（参考 LDA-1B 的 4 个 task embedding）
- [ ] 接入多源数据集（参考 EI-30k 的数据格式）

**关键实验：**
- 加入低质量数据后性能变化（对比 LDA-1B 的 +10% 结论）
- 质量分级策略在不同 WAM 配置下的普适性

### Phase 4: 系统性消融与 Benchmark 论文

**目标：** 利用框架产出系统性消融结论，发表 benchmark 论文。

> **算力注记：** 完整矩阵(表征 × 拓扑 × 动作头 × 生成时机 × 数据策略)在 5B 可训练骨干下组合数超百，无法直接跑完。因此本阶段按 **4a 主效应 → 4b 二阶交互** 的两段式推进，避免 Phase 4 爆炸。

**Phase 4a: 主效应扫描**
- [ ] 以 Phase 0 的**内部起点基线**为原点，每个设计维度独立做 1D 扫描（共约 15 个 config）
- [ ] 在 1-2 个主 benchmark（LIBERO + RoboTwin 择一）上评测
- [ ] 产出每个维度的重要性排序，识别"主导维度"

**Phase 4b: 二阶交互扫描**
- [ ] 仅在 4a 识别出的主导维度间做 2×2 或 2×3 交叉
- [ ] 扩展到全部 benchmark：LIBERO / RoboTwin 2.0 / RoboCasa
- [ ] （可选）在最优组合上做真实世界实验验证

**公平对比方法论（贯穿 4a/4b）：**
- [ ] 明确 hyperparameter 预算：每个 config 给定固定 HP 搜索次数（例如 3 次 seed + 2 次 LR），汇报 best-of-N 而非单次结果
- [ ] 所有 config 共享相同的数据配比、训练步数、评测脚本
- [ ] 敏感度检查：对 seed / LR / 训练步数的鲁棒性分析

**期望产出：**
- "通过系统搜索 WAM 设计空间，我们发现 {最优组合}，在无预训练的情况下达到 SOTA"
- WAM 各设计维度的重要性排序和二阶交互效应分析
- 为社区提供 WAM 设计的实践指南

---

## 评测体系

### 动作性能评测（与 StarVLA 兼容）

| Benchmark | 任务类型 | 指标 |
|-----------|---------|------|
| LIBERO | 桌面操作 (4 suite × 10 tasks) | 成功率 |
| RoboTwin 2.0 | 双臂操作 (50+ tasks) | 成功率 |
| RoboCasa | 家庭场景操作 | 成功率 |

### WAM 专用评测（本项目新增）

| 指标 | 评测内容 | 说明 |
|------|---------|------|
| 推理延迟 | 端到端 inference latency | 区分有/无视频生成 |
| 动力学预测误差 | 未来状态预测的 MSE/LPIPS | 评估世界模型质量 |
| 表征质量 | 线性探测 / 下游任务迁移 | 评估学到的物理先验 |
| 数据效率 | 不同数据量下的性能曲线 | 评估数据利用效率 |
| FVD | 生成视频与真实视频的分布距离 | 仅对有视频生成的配置 |

---

## 配置体系设计

所有设计维度通过统一的 config 系统配置。

> **合法性校验：** 四个设计维度并非完全正交，存在若干互斥/依赖约束，config 加载时必须校验：
> - `world_branch: EncoderOnlyBranch` 时，`attention_topology.action_sees_future` 无意义（无未来视频 token）
> - `world_encoder: DINOEncoder` 时，`world_branch` 不能生成可解码回像素的视频（DINO 特征不可解码），FVD 评测不可用
> - `data_strategy: QualityAwareStrategy` 的"无动作视频→仅视觉预测"依赖 `world_branch` 输出视觉重建目标，与 `EncoderOnlyBranch` 互斥
> - `world_branch.train_video_loss: false` 且 `inference_generate: false` 等价于 EncoderOnly，应收敛为同一配置

示例：

```yaml
# 示例: 复现 Fast-WAM 的 Decoupled + TrainOnly 配置
backbone:
  name: Wan22
  pretrained: path/to/wan2.2-5b
  freeze: false                    # 与 WM4A 的关键区别

world_encoder:
  name: VAEEncoder
  source: backbone                 # 复用骨干的 VAE

world_branch:
  name: DecoupledBranch
  train_video_loss: true           # 训练时有视频目标
  inference_generate: false        # 推理时跳过视频生成
  video_loss_weight: 1.0

attention_topology:
  action_sees_future: false        # 解耦: 动作看不到未来
  future_sees_action: false

data_strategy:
  name: UniformStrategy

action_decoder:
  name: GR00T
  action_window_size: 7
```

```yaml
# 示例: 复现 LDA-1B 风格的 DINO + 质量分级配置
backbone:
  name: Wan22
  pretrained: path/to/wan2.2-5b
  freeze: false

world_encoder:
  name: DINOEncoder
  model: dinov2-large

world_branch:
  name: JointBranch
  train_video_loss: true
  inference_generate: true

attention_topology:
  action_sees_future: true
  future_sees_action: true

data_strategy:
  name: QualityAwareStrategy
  high_quality_objectives: [policy, forward_dynamics, inverse_dynamics, visual_forecast]
  low_quality_objectives: [forward_dynamics, visual_forecast]
  actionless_objectives: [visual_forecast]

action_decoder:
  name: PI
  action_window_size: 10
```

---

## 里程碑与优先级

| 阶段 | 核心产出 | 优先级 |
|------|---------|--------|
| Phase 0 | 可运行 baseline，验证基础设施 | P0 |
| Phase 1 | world representation target study | P0 |
| Phase 2 | 注意力拓扑 + 生成时机的消融 | P0 |
| Phase 3 | 数据质量分级训练 | P1 |
| Phase 4 | 系统消融论文 | P1 |
