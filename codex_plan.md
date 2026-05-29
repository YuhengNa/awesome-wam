# 请你深入解析这个awesomne wam的文件夹目录有什么，要干什么，以及这个.md是当前在做实验的文档，请你给我讲解

> 请你深入解析这个awesomne wam的文件夹目录有什么，要干什么，以及这个.md是当前在做实验的文档，请你给我讲解
>
> User context
>
> Attachments:
> - tokenizer_methods.md: `./wam/tokenizer_methods.md`

<details><summary>3 previous messages</summary>

> 我先把仓库目录和你点名的 `tokenizer_methods.md` 读一遍，再结合 `plan.md/story.md` 解释它现在像是在做什么实验、和整体 WAM 规划怎么接上。
>
> 目录比最初的 AGENTS 说明丰富很多：除了 `wam/` 规划文档，已经有 `runs/` 实验产物、`scripts/` 同步/训练脚本、`docs/related_work` 文献整理和多个 `external/` 参考代码库。我继续读关键 markdown，重点抓实验主线。
>
> `tokenizer_methods.md` 不是孤立笔记，它正好接在当前两条线后面：一条是 FastWAM 表征目标对比，另一条是 OpenPI/pi0.5 上的 generation expert/tokenizer 实验。现在我再看一下 `runs/` 里 tokenizer 相关产物，确认哪些已经真的跑过。
</details>

我看完了。这个 `E:\awesome_wam` 不是一个单纯代码项目，更像是一个 **WAM 机器人世界模型研究工作区**：里面同时放了研究规划、参考代码库、HPC 训练脚本、实验结果镜像和方法草稿。

**整体在干什么**
核心目标是研究：机器人 VLA / WAM 里，训练时预测“未来世界状态”到底应该预测什么表示，才能真正提升动作控制。

也就是说，不只是让模型看当前图像然后出动作，而是在训练时加一个 world prediction loss，让模型学到更好的动态/交互表征。当前主线是 LIBERO 机器人任务。

**目录结构**
- [wam](/E:/awesome_wam/wam)：项目大脑。这里是计划、故事线、实验状态和方法草稿。
- [external](/E:/awesome_wam/external)：参考/开发代码库。最重要的是 `FastWAM` 和 `openpi`。
- [scripts](/E:/awesome_wam/scripts)：HPC 同步、训练、监控、特征统计、可视化脚本。
- [runs](/E:/awesome_wam/runs)：本地镜像的实验结果、日志、eval 视频、可视化图。
- [docs/related_work](/E:/awesome_wam/docs/related_work)：相关论文和整理笔记。
- [papers](/E:/awesome_wam/papers)：论文 PDF，比如 PV-VAE。
- `libero_plus_obs*.zip`：LIBERO 观测样本/相机视角样本。
- `wam.zip`：早期 `wam/` 文档归档。

**当前主代码在哪里**
真实训练代码主要不在根目录，而在：

[external/FastWAM](/E:/awesome_wam/external/FastWAM)

`wam/fastwam_codebase_reading_guide.md` 说得很清楚：当前不是最终重构后的 WAM 框架，而是基于 FastWAM 的研究分支，目的是公平比较不同 world prediction target。

目前已经在比较这些未来预测目标：

- RGB / depth / segmentation，走 Wan VAE 或类图像路径；
- DINOv3 / SigLIP2 / V-JEPA / SVG Res-ViT / SVG-DINO-P，走 frozen feature encoder；
- 训练时预测未来 feature，推理时通常 action-only；
- benchmark 主要是 LIBERO 成功率、action error、feature MSE/cosine、decoded RGB 可视化。

**`tokenizer_methods.md` 在讲什么**
这个文件是当前实验的 tokenizer 方法笔记：

[tokenizer_methods.md](/E:/awesome_wam/wam/tokenizer_methods.md)

它讨论的不是普通 RGB tokenizer，而是 **semantic / feature tokenizer**。文档开头定义了：

`x_t` = frozen visual teacher 的 patch feature，比如 SVG-P、DINOv3、SigLIP feature，而不是像素。

所以它关心的是：未来世界状态已经被 frozen teacher 编成高维 patch feature 后，怎么进一步压缩、预测、对齐动作。

里面有三条路线。

**1. Per-Frame S-VAE / Channel Adapter**
这是最稳的 baseline。

每一帧单独压缩：

```text
RGB_t -> frozen encoder -> x_t [V, N, D]
x_t -> S-VAE encoder -> z_t [V, N, d]
z_t -> S-VAE decoder -> x_hat_t
```

意思是：DINO/SVG/SigLIP 这类 teacher feature 太大，先用一个 VAE 或 adapter 把通道维从 `D` 压到 `d`，例如 64/96/128。

优点是稳、简单、和 Semantic-WM 类似。缺点是它不建模时间，只是“单帧语义压缩”，创新性弱。

当前状态：还没实现 standalone S-VAE，但 OpenPI/pi0.5 里已经有 raw DINO/SigLIP future target 和 DINO32/128 随机投影基础设施。

**2. PV-VAE-Style Temporal Predictive Feature VAE**
这是已经实现并跑过的 temporal tokenizer。

它把一段 feature clip 压成更少的 latent groups：

```text
x_0, x_1, ..., x_16
  -> encoder
  -> z_0, z_1, ..., z_4
  -> decoder
  -> x_hat_0, ..., x_hat_16
```

例如 17 帧被压成 5 组：第 0 帧单独一组，后面每 4 帧一组。

优点是它真的做了时间压缩，接近 PV-VAE / Wan-VAE 那种 temporal compression 思路。缺点也很关键：LIBERO 场景里大量背景静止，模型容易学会“复制当前帧”这个 shortcut，而不是学真实交互动态。

我也看到本地确实有 PV-VAE 实验产物：

[runs/pvvae](/E:/awesome_wam/runs/pvvae)

包括 smoke、3k、30k、stride2 等实验目录。

**3. Delta Transition Tokenizer / DeltaTok-Style**
这是文档里最像“方法创新主线”的方案。

它不压缩整帧，也不压缩整个 clip，而是只压缩变化：

```text
x_t, x_{t+k} -> Delta encoder -> z_delta
x_t, z_delta -> Delta decoder -> x_hat_{t+k}
```

直觉很好：机器人视频里大部分东西不动，真正重要的是 gripper、目标物体、接触和状态变化。所以不要让 bottleneck 浪费容量记背景，而是让 `z_delta` 专门表示“从现在到未来发生了什么变化”。

当前状态：已有 LAM / latent-action 原型，不是完整 DeltaTok。文档建议下一步正式实现：

```text
encode(x_t, x_{t+k}) -> z_delta [B, M, d]
decode(x_t, z_delta) -> x_hat_{t+k}
```

本地也已经有 DeltaTok 风格实验结果：

[runs/deltatok](/E:/awesome_wam/runs/deltatok)

其中 `deltatok_libero_svg_p_stride4_bs64_3000_20260522_171438` 跑到了 3000 step。日志显示最后大概：

```text
step=3000 loss=0.003512 mse=0.007110 copy_mse=0.008126 delta_ratio=0.870
```

这说明它至少已经跑通了一个 SVG-P feature transition tokenizer 原型，并且有 copy baseline 对比。

**我对当前状态的理解**
这个工作区现在大概处在 Phase 1：不是急着搭最终大框架，而是在回答一个核心研究问题：

> 对机器人 WAM 来说，未来预测目标应该是 RGB、结构模态、语义 feature、SVG feature，还是进一步压缩后的 transition token？

`tokenizer_methods.md` 正是在把这个问题推进到下一层：既然 DINO/SVG/SigLIP 这类 feature target 已经能跑，那么下一步该不该做 tokenizer？如果做，是单帧压缩、时间压缩，还是变化压缩？

当前文档自己的判断也很明确：

- 最稳 baseline：Per-frame S-VAE。
- 已实现但风险高：PV-VAE temporal compression。
- 最值得作为主线推进：Delta transition tokenizer。

我会把这套项目一句话概括成：

**awesome_wam 是一个围绕机器人世界模型表征目标与 tokenizer 设计的实验工作区，当前重点是用 FastWAM/OpenPI 在 LIBERO 上比较“预测什么未来表示”以及“怎样把未来变化压成对动作有用的 token”。**

> 所以这个文档是在做什么？

这个 `tokenizer_methods.md` 本质上是在做一件事：

**梳理“未来视觉特征该怎么压缩成 token，供机器人世界模型学习”的实验路线。**

更具体地说，它不是普通项目说明文档，而是当前研究的 **方法选择/实验设计笔记**。它在比较三种 tokenizer 方案：

1. **Per-frame S-VAE**
   每帧单独压缩 DINO/SVG/SigLIP 这类 frozen feature。
   稳，但创新弱。

2. **PV-VAE temporal tokenizer**
   把一段未来 feature clip 压成少量 temporal latent groups。
   已实现、已跑实验，但容易学成“复制静态背景”。

3. **Delta transition tokenizer**
   只压缩 `当前帧 -> 未来帧` 的变化。
   最符合机器人场景，因为真正重要的是 gripper、目标物体、接触、状态变化，而不是背景。文档认为这是最值得继续推进的主线。

所以这个文档在回答的问题是：

> 我们已经能用 DINO/SVG/SigLIP 等视觉特征作为 world prediction target 了，下一步要不要把这些高维 feature 再 tokenizer 化？如果要，哪种 tokenizer 最适合机器人 WAM？

它的结论倾向是：

**先用 S-VAE 做稳 baseline，PV-VAE 作为已实现但有风险的对照，主线应该推 Delta tokenizer。**

换句话说，这个文档是在为下一阶段实验定方向：
从“预测未来整帧特征”升级到“预测对动作有用的未来变化 token”。

> tokenizer和未来视觉特征，encoder，VAE， DINO/SVG/SigLIP 等等有什么关系

它们的关系可以按一条流水线理解：

```text
RGB 图像/视频
  -> 视觉 encoder / teacher
  -> 未来视觉特征 x_t
  -> tokenizer / VAE / adapter
  -> 更小的 token z_t 或 z_delta
  -> world model / action model 学习预测
```

**1. DINO / SVG / SigLIP 是“特征提取器”**
它们通常是 frozen visual encoder，也就是不训练或少训练的视觉老师。

输入是图像：

```text
RGB image
```

输出是 patch feature：

```text
x_t = [N, D]
```

比如每张图被切成很多 patch，每个 patch 有一个高维向量。这个 `x_t` 就叫未来视觉特征。

区别大概是：

- **DINO**：偏语义和空间结构，适合 object/geometry。
- **SigLIP**：偏视觉语言对齐，和文本语义关系更强。
- **SVG**：这里多指 SVG Res-ViT / SVG-DINO-P 一类特征，优点是特征还能被 decoder 近似还原成 RGB，方便可视化。
- **V-JEPA**：偏视频/动态表征。

**2. Encoder 是更泛的概念**
Encoder 就是“把输入编码成表示”的模块。

DINO 是 encoder，SigLIP 是 encoder，SVG feature encoder 也是 encoder。

VAE 里也有 encoder：

```text
VAE encoder: image/feature -> latent
VAE decoder: latent -> image/feature
```

所以 encoder 是大类，DINO/SigLIP/SVG/VAE encoder 是不同实例。

**3. 未来视觉特征是什么**
如果当前时刻是 `t`，未来帧是 `t+1, t+2, ...`。

把未来 RGB 帧送进 frozen encoder：

```text
future RGB frames -> DINO/SVG/SigLIP encoder -> future feature x_{t+k}
```

这个 `x_{t+k}` 就是未来视觉特征。

在 WAM 里，模型训练时会被要求预测它：

```text
当前观测 + 指令 + 动作上下文 -> 预测未来视觉特征
```

这样模型被迫学习“世界将如何变化”。

**4. Tokenizer 是进一步压缩/离散化/结构化这些特征的东西**
DINO/SVG/SigLIP 输出的 feature 往往很大：

```text
x_t: [patch 数量 N, 特征维度 D]
```

直接预测很重，而且里面很多是静态背景。

Tokenizer 的作用是把它变成更小、更有用的 token：

```text
x_t -> tokenizer -> z_t
```

或者对变化建模：

```text
x_t, x_{t+k} -> tokenizer -> z_delta
```

这里的 `z_t` / `z_delta` 就是 tokenizer 输出的 token。

**5. VAE 和 tokenizer 的关系**
VAE 可以看成一种 tokenizer。

比如：

```text
DINO feature x_t
  -> VAE encoder
  -> latent token z_t
  -> VAE decoder
  -> reconstructed feature x_hat_t
```

如果这个 VAE 是逐帧压缩，就是文档里的 **Per-frame S-VAE**。

如果这个 VAE 压缩一段视频特征，就是 **PV-VAE temporal tokenizer**。

如果它压缩两帧之间的变化，就是 **Delta tokenizer**。

**最核心的一句话**
DINO/SVG/SigLIP 负责把 RGB 变成高质量视觉特征；
tokenizer/VAE 负责把这些高维未来特征进一步压成更小、更适合预测的 token；
WAM 再学习预测这些 token，从而获得对未来和动作更有用的世界模型能力。

> 所以这个.md文档里面的实验，是在干什么？是在研究encode的方法，还是tokenizer的方法

更准确地说：这个 `.md` 里的实验主要是在研究 **tokenizer 方法**，但它建立在已经选好的 **encoder / teacher feature** 之上。

也就是：

```text
RGB 图像
  -> frozen encoder: DINO / SVG / SigLIP
  -> 高维视觉特征 x_t
  -> tokenizer: S-VAE / PV-VAE / Delta tokenizer
  -> 压缩后的 token z
```

这个文档不是主要在比较：

```text
DINO vs SVG vs SigLIP 哪个 encoder 更好
```

而是在比较：

```text
拿到 DINO/SVG/SigLIP 这些 feature 之后，
应该怎么把它们压缩成更适合 world model/action model 学习的 token？
```

所以它的研究对象是 **feature tokenizer / semantic tokenizer**。

具体三条路线是：

- **Per-frame S-VAE**：研究“每一帧 feature 怎么压缩”。
- **PV-VAE-style**：研究“一段未来 feature clip 怎么做时间压缩”。
- **Delta tokenizer**：研究“当前到未来的变化怎么压缩”。

其中 DINO/SVG/SigLIP 在文档里更像是 **输入特征来源**，不是主要被研究的变量。

不过要注意：这个大项目里还有别的文档是在研究 encoder / representation target，比如 FastWAM 里比较 RGB、depth、seg、DINO、SigLIP、SVG。
但你问的这个 [tokenizer_methods.md](/E:/awesome_wam/wam/tokenizer_methods.md)，重点是 **tokenizer 方法选择**。

> 现在是怎么进行评测的，实验室怎么跑的

<details><summary>1 previous message</summary>

> 我把评测链路再对齐一下：一边是训练时的 validation/eval 视频，一边是 LIBERO 在线 rollout 成功率；实验室/HPC 跑法主要靠 sbatch 提交、同步结果回这个 `runs/` 镜像。
</details>

现在评测分两层：**离线指标** + **在线 LIBERO rollout 成功率**。

**1. 训练时离线评测**
训练过程中每隔 `eval_every` 会跑一次 validation，主要看：

- `loss_action`：动作预测损失。
- `loss_video` / `feature_mse`：未来视觉特征预测误差。
- `feature_cosine`：预测 feature 和真实 feature 的方向相似度。
- `action_l1 / action_l2`：动作误差。
- 如果是 SVG 特征，还会 decode 回 RGB，看 `decoded_psnr / decoded_ssim`，并保存可视化视频。

SVG eval 视频一般有 5 行：

```text
1. 预测 feature 的 PCA
2. GT feature 的 PCA
3. 预测 feature decode 回 RGB
4. GT feature decode 回 RGB
5. GT RGB
```

这部分主要回答：

> 模型有没有学会预测未来视觉 feature？预测出来的 feature 有没有结构？动作误差有没有下降？

相关说明在 [fastwam_codebase_reading_guide.md](/E:/awesome_wam/wam/fastwam_codebase_reading_guide.md) 和 [current_fastwam_experiments.md](/E:/awesome_wam/wam/current_fastwam_experiments.md)。

**2. 真正的控制评测：LIBERO 在线 rollout**
最终更重要的是在线环境成功率。当前主要跑 LIBERO 四个 suite：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

每个 suite 10 个任务，每个任务 50 次 trial，所以一个 suite 是 500 次，总共 2000 次 rollout。

当前 SVG-DINO-P 已经有完整结果：

```text
Pure SVG-DINO-P:      87.35%
VAE-cond SVG-DINO-P:  88.05%
```

评测设置大概是：

```text
checkpoint: step_030000.pt
eval mode: action-only online eval
visualize_future_video=false
action_horizon=16
replan_steps=10
num_inference_steps=10
```

也就是说，虽然训练时有未来 feature/world loss，但评测时主要还是让模型直接出动作，不强制它“想象未来视频”。

**3. tokenizer 实验怎么评测**
你问的 `tokenizer_methods.md` 这条线，目前更多是 tokenizer 本身的诊断评测，还没完全进入最终控制成功率评测。

它看的是：

- `feature_mse`：重建/预测未来 feature 的误差。
- `cosine_loss`：feature 方向是否接近。
- `future_mse`：未来区域预测误差。
- `static_future_mse` 或 `copy_mse`：直接复制当前帧的 baseline。
- `delta_ratio`：模型预测出来的动态幅度是否接近真实变化。
- PCA / SVG decoded RGB 可视化。

比如 DeltaTok 实验里会比较：

```text
模型预测 future feature 的 mse
vs
直接 copy 当前 feature 的 copy_mse
```

如果模型比 copy baseline 好，说明它不是只学静态背景，而是真的学到了一点 transition。

**4. 实验室/HPC 是怎么跑的**
现在流程大概是：

```text
本地改代码/配置
  -> sync 到 HPC
  -> sbatch 提交训练
  -> 训练结果保存在远端 runs/
  -> scp/rsync 同步 eval 视频、json、log 回本地 runs/
```

FastWAM 主代码在：

[external/FastWAM](/E:/awesome_wam/external/FastWAM)

同步脚本是：

[sync_fastwam_to_hpc.sh](/E:/awesome_wam/scripts/sync_fastwam_to_hpc.sh)

它会把 `src/`、`configs/`、`experiments/`、`scripts/` 等同步到 HPC：

```text
/data/user/jhe724/workspace/FastWAM
```

常见训练设置：

```text
4 H100
batch 8 per GPU
gradient accumulation 2
effective batch 64
lr 1e-4
bf16
10 epochs
save_every=5000
eval_every=200
```

结果同步脚本例如：

[sync_current_svg_eval.sh](/E:/awesome_wam/scripts/sync_current_svg_eval.sh)

会把远端 `eval/*.mp4`、`*.json`、slurm 日志同步回本地 [runs](/E:/awesome_wam/runs)。

**一句话总结**
现在评测不是只看重建好不好，而是分三步：先看训练 loss 和 feature 预测指标，再看 SVG/PCA 可视化确认模型没学歪，最后用 LIBERO 在线 rollout 成功率判断这个 world/tokenizer 方法是否真的帮动作控制。当前 tokenizer 文档里的实验还偏“中间表示是否学得好”，最终目标是接到 LIBERO success 上。

> 请你仔细深入解析阅读以下几个arxiv链接的文献，以及我上传的pdf文献。之后我们继续深入做这个awesome wam的工作，需要熟悉这些工作的原理，实验方法等。https://arxiv.org/pdf/2605.06388
> https://github.com/chandar-lab/semantic-wm   https://arxiv.org/pdf/2605.02134      https://arxiv.org/pdf/2604.04913
>
> User context
>
> Attachments:
> - 2604.11689v1.pdf: `C:/Users/Lenovo/Downloads/2604.11689v1.pdf`
> - 重构还是语义_什么使潜在空间对机器人世界模型有用.pdf: `C:/Users/Lenovo/xwechat_files/wxid_qa71qlxk71kq22_a9f8/msg/file/2026-05/重构还是语义_什么使潜在空间对机器人世界模型有用.pdf`

<details><summary>11 previous messages</summary>

> Stream error
>
> Reconnecting... 2/5
> request timed out
>
> Stream error
>
> Reconnecting... 3/5
> request timed out
>
> Stream error
>
> Reconnecting... 4/5
> request timed out
>
> Stream error
>
> Reconnecting... 5/5
> request timed out
>
> 我会把这几篇当成后续 `awesome_wam` 的方法底座来读：先确认论文标题/代码仓库，再用本地 PDF 和已有 txt 提取细节，最后把它们和当前 WAM/tokenizer 实验怎么接起来讲清楚。
>
> <details><summary>Ran 4 commands, searched web 1 time</summary>
>
> <details><summary>Searched web 1 time</summary>
>
> - Searched web for `2605.06388 Reconstruction or Semantics What Makes Latent Spaces Useful for Robot World Models`
>
> </details>
>
> - Ran `Get-ChildItem -LiteralPath "C:\Users\Lenovo\Downloads\2604.11689v1.pdf"`
>
> - Ran `Get-ChildItem -LiteralPath "C:\Users\Lenovo\xwechat_files\wxid_qa71qlxk71kq22_a9f8\msg\file\2026-05\重构还是语义_什么使潜在空间对机器人世界模型有用.pdf"`
>
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt -TotalCount 120`
>
> - Ran `Get-Content docs\related_work\2604.11689_lary.pdf -ErrorAction SilentlyContinue`
>
> </details>
>
> 本地已经有 `2605.06388` 的正文 txt，微信里的中文 PDF 也能访问；另一个 `2604.11689v1.pdf` 是 LARYBench。刚才直接读 PDF 会输出二进制，我接下来用 PDF 文本提取方式读摘要、方法和实验表，避免被 PDF 内部对象干扰。
>
> <details><summary>Ran 4 commands, searched web 4 times</summary>
>
> - Ran `Get-Command pdftotext -ErrorAction SilentlyContinue`
>
> - Ran `@'
> import importlib.util
> mods=['pypdf','PyPDF2','pdfplumber','fitz']
> for m in mods:
>     print(m, bool(importlib.util.find_spec(m)))
> '@ | python -`
>
> - Ran `Get-ChildItem docs\related_work -Filter "*.txt" | Select-Object Name,Length`
>
> - Ran `Get-ChildItem papers\pvvae -Force`
>
> <details><summary>Searched web 4 times</summary>
>
> - Searched web for `arXiv 2605.02134 PV-VAE Predictive Video VAE abstract`
> - Searched web for `https://arxiv.org/abs/2605.02134`
> - Searched web for `'BridgeData V2' in https://github.com/chandar-lab/semantic-wm`
> - Searched web for `https://github.com/chandar-lab/semantic-wm/blob/main/README.md`
>
> </details>
>
> </details>
>
> 四篇的定位已经很清楚了：Semantic-WM 研究“重构 latent 还是语义 latent 更适合机器人 world model”，PV-VAE 研究“VAE/tokenizer 怎么带预测性”，DeltaTok 研究“把帧间变化压成 delta token”，LARY 研究“latent action/视觉特征如何对齐动作”。我继续把关键实验协议和结论抽出来，方便后面直接指导 awesome_wam。
>
> <details><summary>Ran 32 commands, searched web 14 times</summary>
>
> - Ran `rg -n "Bridge|PCK|controll|policy|V-JEPA|SigLIP|Web-DINO|S-VAE|Table|Conclusion|planning|trajectory" docs\related_work\2605.06388_reconstruction_or_semantics.txt`
>
> - Ran `Get-Content docs\related_work\SUMMARY.md`
>
> - Ran `Get-Content docs\related_work\REPRESENTATION_TAXONOMY.md`
>
> - Ran `Get-Content docs\related_work\FEATURE_DIT_ARCHITECTURES.md`
>
> <details><summary>Searched web 7 times</summary>
>
> - Searched web for `ar5iv 2605.02134 Predictive Video VAE`
> - Searched web for `https://arxiv.org/abs/2604.04913`
> - Searched web for `'randomly discard' in https://arxiv.org/pdf/2605.02134`
> - Searched web for `'DeltaTok' in https://arxiv.org/pdf/2604.04913`
> - Searched web for `https://github.com/chandar-lab/semantic-wm`
> - Searched web for `https://arxiv.org/abs/2604.11689`
> - Searched web for `https://meituan-longcat.github.io/LARYBench`
>
> </details>
>
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 240 -First 120`
>
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 451 -First 130`
>
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 660 -First 90`
>
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 913 -First 35`
>
> <details><summary>Searched web 7 times</summary>
>
> - Searched web for `https://arxiv.org/pdf/2605.06388`
> - Searched web for `'Abstract' in https://arxiv.org/pdf/2605.02134`
> - Searched web for `https://arxiv.org/pdf/2605.02134`
> - Searched web for `'Abstract' in https://arxiv.org/pdf/2604.04913`
> - Searched web for `https://raw.githubusercontent.com/chandar-lab/semantic-wm/main/README.md`
> - Searched web for `arxiv 2604.11689 LARY latent action representation yielding benchmark`
> - Searched web for `https://arxiv.org/abs/2604.11689`
>
> </details>
>
> - Ran `rg -n "LARY|Latent Action Representation|semantic action|low-level|LAPA-DINO|DINOv3|SigLIP" -S . -g "*.md"`
>
> - Ran `Get-ChildItem external\DeltaTok -Force | Select-Object -First 40`
>
> - Ran `Get-ChildItem external\LDA-1B -Force | Select-Object -First 20`
>
> - Ran `rg -n "delta|latent|token|DINO|predict|encoder|decoder|diffusion|Kinetics|Something" external\DeltaTok\README.md external\DeltaTok\configs external\DeltaTok\models -S`
>
> - Ran `rg -n "PV-VAE|Predictive Video|temporal_compression|observed_groups|future_mse|static_future_mse|predictive" -S . -g "*.md" -g "*.py"`
>
> - Ran `Get-Content external\DeltaTok\README.md -TotalCount 120`
>
> - Ran `Get-Content external\DeltaTok\models\deltatok.py`
>
> - Ran `Get-Content external\DeltaTok\models\deltaworld.py`
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py -TotalCount 260`
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\latent_action.py -TotalCount 260`
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py -TotalCount 220`
>
> - Ran `Get-Content external\openpi\scripts\train_predictive_feature_vae_libero.py -TotalCount 80`
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\latent_action.py -TotalCount 220`
>
> - Ran `Get-Item external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py | Format-List Length,FullName`
>
> - Ran `Get-Item external\openpi\src\openpi\models_pytorch\latent_action.py | Format-List Length,FullName`
>
> - Ran `rg -n "class PredictiveFeatureVAE|compute_loss|future_mse|static_future_mse|delta_ratio|class FeatureLatentActionModel|latent_dim" external\openpi\src\openpi\models_pytorch external\openpi\scripts`
>
> - Ran `Get-Content external\openpi\scripts\train_deltatok_libero.py -TotalCount 300`
>
> - Ran `Get-Content external\openpi\scripts\train_lam_libero.py | Select-Object -Skip 430 -First 90`
>
> - Ran `Get-Content external\openpi\scripts\pvvae_libero_svgp_4gpu_zzhong778.sbatch`
>
> - Ran `Get-Item external\openpi\src\openpi\models_pytorch\delta_tokenizer.py | Format-List Length,FullName`
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\delta_tokenizer.py -TotalCount 260`
>
> - Ran `Get-ChildItem external -Filter semantic* -Force`
>
> - Ran `Get-Content docs\related_work\README.md`
>
> - Ran `Get-Content external\DeltaTok\README.md | Select-Object -Skip 112 -First 100`
>
> </details>
>
> 可以。这个阅读任务的核心结论我先给你压成一版“后续做 awesome_wam 能直接用”的研究笔记。
>
> 我确认了这些材料：
>
> - [2605.06388 Reconstruction or Semantics?](https://arxiv.org/pdf/2605.06388)
> - [semantic-wm GitHub](https://github.com/chandar-lab/semantic-wm)
> - [2605.02134 PV-VAE](https://arxiv.org/pdf/2605.02134)
> - [2604.04913 DeltaTok](https://arxiv.org/pdf/2604.04913)
> - 本地 PDF：`C:/Users/Lenovo/Downloads/2604.11689v1.pdf`，是 **LARY**
> - 本地中文 PDF：`重构还是语义_什么使潜在空间对机器人世界模型有用.pdf`，对应 **Reconstruction or Semantics?** 的中文材料
>
> **总体脉络**
> 这几篇其实围绕同一个问题：
>
> > 机器人 world model 里，“未来状态”到底应该用什么 latent / feature / token 表示，才对动作控制有用？
>
> 它们不是单纯研究视频生成质量，而是在逼近一个更机器人化的问题：
>
> ```text
> 视觉未来预测好看
>   不等于
> 动作控制有用
> ```
>
> 所以 awesome_wam 的方向应该从“预测未来画面”转成：
>
> ```text
> 预测 action-relevant future representation
> ```
>
> 也就是预测对动作、接触、目标物体、任务进展有用的未来表示。
>
> **1. Reconstruction or Semantics?**
> 这篇是 awesome_wam 最直接的理论参照。
>
> 它做的是一个 controlled study：固定数据、DiT transition model、动作条件、训练协议，只改变 latent space。
>
> 比较两类 encoder：
>
> ```text
> 重构型 latent:
>   SD3 VAE / VA-VAE / Cosmos
>
> 语义型 latent:
>   V-JEPA 2.1 / Web-DINO / SigLIP 2
> ```
>
> 语义 encoder 的高维 feature 还有两种用法：
>
> ```text
> native high-dimensional feature
> 或
> S-VAE adapter 压缩到 d=96
> ```
>
> 实验数据是 Bridge V2，约 60K 真实机器人 demonstration。模型用 action-conditioned latent diffusion world model，训练目标是 flow matching，条件历史帧 `H=2`，预测 8 个 future frames。
>
> 它的评测非常关键，分三类：
>
> ```text
> 1. visual fidelity:
>    PSNR / SSIM / LPIPS / FVD 等
>
> 2. action / planning:
>    CEM action recovery
>    OpenVLA 在 world model 里 rollout 的成功率
>    OOD distractor / instruction robustness
>
> 3. latent quality:
>    IDM action recoverability
>    task success classifier
>    PCK point tracking
> ```
>
> 最重要结论：
>
> ```text
> 视觉重建好，不代表机器人控制好。
> ```
>
> VAE/Cosmos 这类 reconstruction latent 往往像素指标好，但 semantic latent 在 action recovery、task success、planning、policy-in-world rollout 上更强。
>
> 对 awesome_wam 的启发是：
>
> - 不能只报 feature MSE / PSNR / decoded RGB。
> - 必须报 LIBERO success / action L1/L2 / action recoverability / robustness。
> - DINO/SigLIP/V-JEPA/SVG 是否好，要看它们是否保留 action-relevant dynamics。
> - S-VAE adapter 有用，但可能压掉细粒度控制几何；native semantic feature 不一定太贵，因为 DiT token 数固定时，通道维主要影响输入/输出 projection。
>
> **2. PV-VAE**
> PV-VAE 对我们最有用的点不是“换一个视觉 encoder”，而是 **怎么训练一个 predictive tokenizer**。
>
> 它的思想可以概括成：
>
> ```text
> 不要只做 reconstruction tokenizer
> 要做 predictive tokenizer
> ```
>
> 普通 VAE 是：
>
> ```text
> clip -> latent -> reconstruct same clip
> ```
>
> PV-VAE-style 是：
>
> ```text
> observed prefix -> compressed latent groups -> reconstruct/predict full clip
> ```
>
> 在 awesome_wam 当前文档里，对应的是：
>
> ```text
> x_0, x_1, ..., x_16
>   -> encoder
>   -> z_0, z_1, ..., z_4
>   -> decoder
>   -> x_hat_0, ..., x_hat_16
> ```
>
> 也就是 17 帧压成 5 组 latent，做 temporal compression。
>
> 它对我们有两个启发：
>
> - tokenizer 不能只问“能不能重建 feature”，还要问“是否带预测性”。
> - 但在 LIBERO 这种静态背景很多的机器人视频里，PV-VAE 很容易学到 copy/average shortcut。
>
> 所以我们现在的 `static_future_mse` / `copy_mse` 诊断是对的。只要模型比复制当前帧没好多少，就说明它没有学到真正的 interaction dynamics。
>
> **3. DeltaTok**
> DeltaTok 是 tokenizer_methods.md 里最值得继续推的路线。
>
> 它的核心非常漂亮：
>
> ```text
> 不要压缩整帧
> 压缩帧间变化
> ```
>
> 标准形式是：
>
> ```text
> x_t, x_{t+k} -> encoder -> z_delta
> x_t, z_delta -> decoder -> x_hat_{t+k}
> ```
>
> 官方 DeltaTok 是在 frozen DINOv3 feature space 里做的。它把 frame-to-frame change 压成一个 delta token，然后 DeltaWorld 再学习预测这些 delta tokens。
>
> 它的代码结构也和我们当前想法很接近：
>
> ```text
> encode(x, y):
>   z token + previous feature x + next feature y
>   -> transformer encoder
>   -> one delta token
>
> decode(z, x):
>   delta token + previous feature x
>   -> transformer decoder
>   -> reconstructed next feature y_hat
> ```
>
> 评测不是看 RGB 像素，而是把预测 DINOv3 features 接 downstream heads：
>
> ```text
> segmentation mIoU
> depth RMSE
> RGB visualization
> copy-last baseline
> ```
>
> 对 awesome_wam 的启发：
>
> - Delta tokenizer 比 PV-VAE 更适合机器人，因为机器人视频大部分是静态背景，真正重要的是局部变化。
> - `z_delta` 应该表示 gripper、目标物体、接触、状态改变。
> - 我们不应该只做 one-token，机器人场景可能需要 `M=1/4/8` 个 delta tokens。
> - 后续最好加 action-conditioned predictor：
>
> ```text
> history feature + action chunk -> z_delta
> x_t + z_delta -> x_hat_future
> ```
>
> 这会比“直接预测整帧 future feature”更符合 interaction dynamics。
>
> **4. LARY**
> LARY 是另一个重要参照，它研究的是：
>
> ```text
> latent representation 是否和 physical action 对齐
> ```
>
> 它不是 WAM 论文，而是 latent action / vision-to-action alignment benchmark。
>
> 它评测两层能力：
>
> ```text
> what to do:
>   高层语义动作理解
>
> how to do:
>   低层机器人控制回归
> ```
>
> 本地 summary 里记录的规模是：
>
> ```text
> >1M videos
> 620K image pairs
> 595K motion trajectories
> ```
>
> 关键结论是：一些通用视觉基础模型，即使没有专门 action supervision，也能比专门 embodied latent action model 更好地对齐物理动作。
>
> 对 awesome_wam 的启发：
>
> - 我们不能只问 future feature 能不能预测。
> - 要问这个 feature/token 能不能 recover action。
> - 可以训练一个 IDM / latent action probe：
>
> ```text
> z_t, z_{t+k}, instruction -> action chunk
> ```
>
> 如果某个 representation 的 action recoverability 高，它更可能对 robot control 有用。
>
> 这也直接支持我们当前的两个想法：
>
> ```text
> Interaction-Saliency World Loss
> Interaction-Aware Latent Action
> ```
>
> 也就是用 action recoverability 或 latent action attention 来判断哪些 token/区域真的重要。
>
> **放回 awesome_wam**
> 现在 awesome_wam 应该被定位成：
>
> ```text
> 在 FastWAM / OpenPI / LIBERO 框架下，
> 系统研究未来世界表示和 tokenizer 设计，
> 以及它们是否真的提升动作控制。
> ```
>
> 不是简单说：
>
> ```text
> DINO 比 VAE 好
> ```
>
> 这个已经被很多工作覆盖了。我们的更好角度是：
>
> ```text
> 在 FastWAM 的 train-only / joint / causal / action-only inference regimes 下，
> 不同 future target 和 tokenizer 是否改变 action success？
> 哪些 world metrics 真的预测 control performance？
> ```
>
> 我建议后面按这个顺序推进：
>
> 1. **Representation baseline**
>    固定 LIBERO 数据、horizon、optimizer、动作头，比较 RGB/depth/seg/DINO/SigLIP/V-JEPA/SVG。
>
> 2. **Action-alignment probe**
>    给每个 representation 加 IDM/action recovery：
>
>    ```text
>    feature_t, feature_future -> action chunk
>    ```
>
> 3. **Tokenizer comparison**
>    做三组：
>
>    ```text
>    S-VAE per-frame
>    PV-VAE temporal compression
>    Delta transition tokenizer
>    ```
>
> 4. **强制和 copy baseline 比**
>    每个 tokenizer 都要报：
>
>    ```text
>    future_mse
>    copy_mse
>    delta_ratio
>    action recoverability
>    visualization
>    ```
>
> 5. **最终看 LIBERO success**
>    只有 feature/tokenizer 指标好还不够，最终必须跑 rollout success。
>
> 一句话总结：
>
> **Semantic-WM 告诉我们“语义 latent 通常比重构 latent 更 action-relevant”；PV-VAE 告诉我们“tokenizer 要有预测性”；DeltaTok 告诉我们“压变化比压整帧更高效”；LARY 告诉我们“最终要看 latent 是否对齐 physical action”。awesome_wam 下一步正好应该把这四件事合在 LIBERO/FastWAM 里验证。**
</details>

可以。这个阅读任务的核心结论我先给你压成一版“后续做 awesome_wam 能直接用”的研究笔记。

我确认了这些材料：

- [2605.06388 Reconstruction or Semantics?](https://arxiv.org/pdf/2605.06388)
- [semantic-wm GitHub](https://github.com/chandar-lab/semantic-wm)
- [2605.02134 PV-VAE](https://arxiv.org/pdf/2605.02134)
- [2604.04913 DeltaTok](https://arxiv.org/pdf/2604.04913)
- 本地 PDF：`C:/Users/Lenovo/Downloads/2604.11689v1.pdf`，是 **LARY**
- 本地中文 PDF：`重构还是语义_什么使潜在空间对机器人世界模型有用.pdf`，对应 **Reconstruction or Semantics?** 的中文材料

**总体脉络**
这几篇其实围绕同一个问题：

> 机器人 world model 里，“未来状态”到底应该用什么 latent / feature / token 表示，才对动作控制有用？

它们不是单纯研究视频生成质量，而是在逼近一个更机器人化的问题：

```text
视觉未来预测好看
  不等于
动作控制有用
```

所以 awesome_wam 的方向应该从“预测未来画面”转成：

```text
预测 action-relevant future representation
```

也就是预测对动作、接触、目标物体、任务进展有用的未来表示。

**1. Reconstruction or Semantics?**
这篇是 awesome_wam 最直接的理论参照。

它做的是一个 controlled study：固定数据、DiT transition model、动作条件、训练协议，只改变 latent space。

比较两类 encoder：

```text
重构型 latent:
  SD3 VAE / VA-VAE / Cosmos

语义型 latent:
  V-JEPA 2.1 / Web-DINO / SigLIP 2
```

语义 encoder 的高维 feature 还有两种用法：

```text
native high-dimensional feature
或
S-VAE adapter 压缩到 d=96
```

实验数据是 Bridge V2，约 60K 真实机器人 demonstration。模型用 action-conditioned latent diffusion world model，训练目标是 flow matching，条件历史帧 `H=2`，预测 8 个 future frames。

它的评测非常关键，分三类：

```text
1. visual fidelity:
   PSNR / SSIM / LPIPS / FVD 等

2. action / planning:
   CEM action recovery
   OpenVLA 在 world model 里 rollout 的成功率
   OOD distractor / instruction robustness

3. latent quality:
   IDM action recoverability
   task success classifier
   PCK point tracking
```

最重要结论：

```text
视觉重建好，不代表机器人控制好。
```

VAE/Cosmos 这类 reconstruction latent 往往像素指标好，但 semantic latent 在 action recovery、task success、planning、policy-in-world rollout 上更强。

对 awesome_wam 的启发是：

- 不能只报 feature MSE / PSNR / decoded RGB。
- 必须报 LIBERO success / action L1/L2 / action recoverability / robustness。
- DINO/SigLIP/V-JEPA/SVG 是否好，要看它们是否保留 action-relevant dynamics。
- S-VAE adapter 有用，但可能压掉细粒度控制几何；native semantic feature 不一定太贵，因为 DiT token 数固定时，通道维主要影响输入/输出 projection。

**2. PV-VAE**
PV-VAE 对我们最有用的点不是“换一个视觉 encoder”，而是 **怎么训练一个 predictive tokenizer**。

它的思想可以概括成：

```text
不要只做 reconstruction tokenizer
要做 predictive tokenizer
```

普通 VAE 是：

```text
clip -> latent -> reconstruct same clip
```

PV-VAE-style 是：

```text
observed prefix -> compressed latent groups -> reconstruct/predict full clip
```

在 awesome_wam 当前文档里，对应的是：

```text
x_0, x_1, ..., x_16
  -> encoder
  -> z_0, z_1, ..., z_4
  -> decoder
  -> x_hat_0, ..., x_hat_16
```

也就是 17 帧压成 5 组 latent，做 temporal compression。

它对我们有两个启发：

- tokenizer 不能只问“能不能重建 feature”，还要问“是否带预测性”。
- 但在 LIBERO 这种静态背景很多的机器人视频里，PV-VAE 很容易学到 copy/average shortcut。

所以我们现在的 `static_future_mse` / `copy_mse` 诊断是对的。只要模型比复制当前帧没好多少，就说明它没有学到真正的 interaction dynamics。

**3. DeltaTok**
DeltaTok 是 tokenizer_methods.md 里最值得继续推的路线。

它的核心非常漂亮：

```text
不要压缩整帧
压缩帧间变化
```

标准形式是：

```text
x_t, x_{t+k} -> encoder -> z_delta
x_t, z_delta -> decoder -> x_hat_{t+k}
```

官方 DeltaTok 是在 frozen DINOv3 feature space 里做的。它把 frame-to-frame change 压成一个 delta token，然后 DeltaWorld 再学习预测这些 delta tokens。

它的代码结构也和我们当前想法很接近：

```text
encode(x, y):
  z token + previous feature x + next feature y
  -> transformer encoder
  -> one delta token

decode(z, x):
  delta token + previous feature x
  -> transformer decoder
  -> reconstructed next feature y_hat
```

评测不是看 RGB 像素，而是把预测 DINOv3 features 接 downstream heads：

```text
segmentation mIoU
depth RMSE
RGB visualization
copy-last baseline
```

对 awesome_wam 的启发：

- Delta tokenizer 比 PV-VAE 更适合机器人，因为机器人视频大部分是静态背景，真正重要的是局部变化。
- `z_delta` 应该表示 gripper、目标物体、接触、状态改变。
- 我们不应该只做 one-token，机器人场景可能需要 `M=1/4/8` 个 delta tokens。
- 后续最好加 action-conditioned predictor：

```text
history feature + action chunk -> z_delta
x_t + z_delta -> x_hat_future
```

这会比“直接预测整帧 future feature”更符合 interaction dynamics。

**4. LARY**
LARY 是另一个重要参照，它研究的是：

```text
latent representation 是否和 physical action 对齐
```

它不是 WAM 论文，而是 latent action / vision-to-action alignment benchmark。

它评测两层能力：

```text
what to do:
  高层语义动作理解

how to do:
  低层机器人控制回归
```

本地 summary 里记录的规模是：

```text
>1M videos
620K image pairs
595K motion trajectories
```

关键结论是：一些通用视觉基础模型，即使没有专门 action supervision，也能比专门 embodied latent action model 更好地对齐物理动作。

对 awesome_wam 的启发：

- 我们不能只问 future feature 能不能预测。
- 要问这个 feature/token 能不能 recover action。
- 可以训练一个 IDM / latent action probe：

```text
z_t, z_{t+k}, instruction -> action chunk
```

如果某个 representation 的 action recoverability 高，它更可能对 robot control 有用。

这也直接支持我们当前的两个想法：

```text
Interaction-Saliency World Loss
Interaction-Aware Latent Action
```

也就是用 action recoverability 或 latent action attention 来判断哪些 token/区域真的重要。

**放回 awesome_wam**
现在 awesome_wam 应该被定位成：

```text
在 FastWAM / OpenPI / LIBERO 框架下，
系统研究未来世界表示和 tokenizer 设计，
以及它们是否真的提升动作控制。
```

不是简单说：

```text
DINO 比 VAE 好
```

这个已经被很多工作覆盖了。我们的更好角度是：

```text
在 FastWAM 的 train-only / joint / causal / action-only inference regimes 下，
不同 future target 和 tokenizer 是否改变 action success？
哪些 world metrics 真的预测 control performance？
```

我建议后面按这个顺序推进：

1. **Representation baseline**
   固定 LIBERO 数据、horizon、optimizer、动作头，比较 RGB/depth/seg/DINO/SigLIP/V-JEPA/SVG。

2. **Action-alignment probe**
   给每个 representation 加 IDM/action recovery：

   ```text
   feature_t, feature_future -> action chunk
   ```

3. **Tokenizer comparison**
   做三组：

   ```text
   S-VAE per-frame
   PV-VAE temporal compression
   Delta transition tokenizer
   ```

4. **强制和 copy baseline 比**
   每个 tokenizer 都要报：

   ```text
   future_mse
   copy_mse
   delta_ratio
   action recoverability
   visualization
   ```

5. **最终看 LIBERO success**
   只有 feature/tokenizer 指标好还不够，最终必须跑 rollout success。

一句话总结：

**Semantic-WM 告诉我们“语义 latent 通常比重构 latent 更 action-relevant”；PV-VAE 告诉我们“tokenizer 要有预测性”；DeltaTok 告诉我们“压变化比压整帧更高效”；LARY 告诉我们“最终要看 latent 是否对齐 physical action”。awesome_wam 下一步正好应该把这四件事合在 LIBERO/FastWAM 里验证。**

> 我让你看这些，并不是着急说要做实验，而是希望你脑子里有对于这些文献的深入理解，这样在未来，我需要你给出指导时，你可以给出科学性的建设性的研究方法

明白，你这个意思很重要：现在不是让我立刻排实验，而是先把这些工作“内化成研究判断力”。

我会按这个方式记住它们在 awesome_wam 里的作用：

- **Reconstruction or Semantics?**
  不是简单证明“语义特征更好”，而是在提醒我们：机器人 world model 的 latent 不能用视觉重建指标来选，必须看 action recoverability、planning、policy rollout、task progress preservation。它给的是评测哲学和 controlled comparison 方法论。

- **Semantic-WM 代码/项目**
  是这个哲学的可执行范式：固定 transition model 和训练协议，只改 latent interface，然后用多轴指标判断 latent 是否真的 policy-relevant。

- **PV-VAE**
  它重要的不是“我们也要做 PV-VAE”，而是 tokenizer 不能只做 reconstruction bottleneck；一个有价值的 latent 应该被训练成 predictive latent。它同时警告我们：在静态背景很多的机器人视频中，预测型压缩很容易走向 copy shortcut。

- **DeltaTok**
  它给了一个很强的 inductive bias：世界变化比完整世界状态更值得压缩。对机器人来说，这意味着 token 应该尽量承载 interaction transition，而不是背景和纹理。

- **LARY**
  它提醒我们，视觉 latent / action latent 的价值要通过“能否解释动作”来衡量。它不是 WAM 论文，但非常适合作为我们判断 representation 是否 action-aligned 的方法论来源。

之后你问我“这个方向是否科学”“该怎么设计 ablation”“这个指标有没有说服力”“该怎么写 contribution”“哪个方法更值得投入”时，我会以这几条原则作为底层判断：

```text
1. 不把视觉好看等同于机器人有用。
2. 不把 feature MSE 下降等同于控制提升。
3. 每个 representation/tokenizer 都要问它保留了什么 action-relevant information。
4. tokenizer 的核心不是压缩率，而是压缩后是否更利于预测、动作恢复、任务进展判断。
5. 对机器人视频，copy/static shortcut 是必须主动排查的问题。
6. 任何方法 claim 都要和已有 Semantic-WM、PV-VAE、DeltaTok、LARY 区分清楚。
```

所以后面我们继续做 awesome_wam 时，我不会急着“堆实验”，而会先帮你判断：这个实验到底在验证哪个科学问题，是否有对照，指标是否能支撑 claim，以及它在相关工作里站不站得住。

> 请你重点再读一下，Reconstruction or Semantics?论文中的experiment和findings部分的内容

<details><summary>3 previous messages</summary>

> 好，我这次只聚焦 `Reconstruction or Semantics?` 的 **Experiment setup** 和 **Findings**，不发散到别的论文。我会把它拆成：实验控制变量、模型/数据/评测轴、主要发现、对 awesome_wam 的方法论含义。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 232 -First 250`
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 428 -First 360`
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 785 -First 190`
> - Ran `Get-Content docs\related_work\2605.06388_reconstruction_or_semantics.txt | Select-Object -Skip 2178 -First 190`
>
> </details>
>
> 我重点重读了 `Reconstruction or Semantics?` 的实验和 findings。它最核心的地方不是“semantic latent 赢了”，而是它把 **latent space choice** 设计成一个严格受控变量来研究。
>
> **实验设计**
> 论文的实验问题是：
>
> ```text
> 在机器人 action-conditioned latent diffusion world model 中，
> 到底什么 latent space 更有用：
> 重构型 latent，还是语义型 latent？
> ```
>
> 它固定了这些因素：
>
> ```text
> dataset
> history length
> action conditioning
> DiT transition architecture
> optimizer
> training schedule
> ```
>
> 只改变：
>
> ```text
> encoder f_phi
> optional adapter alpha_psi
> decoder path
> ```
>
> 这点很关键。它不是随便比较一堆模型，而是在控制变量下问：**latent interface 本身对机器人 world model 有什么影响。**
>
> 数据集是 **Bridge V2**，约 60K WidowX 250 机器人 demonstrations，包含 RGB、7-DoF action 和语言指令。额外用 SOAR 的成功/失败轨迹训练 success classifier。
>
> 比较的 latent 分两类：
>
> ```text
> Reconstruction-aligned:
>   SD3 VAE, VA-VAE, Cosmos
>
> Semantics-aligned:
>   V-JEPA 2.1, Web-DINO, SigLIP 2
> ```
>
> 语义 latent 又分两种：
>
> ```text
> native high-dimensional feature
> S-VAE adapter 压缩到 d=96
> ```
>
> transition model 是 spatial-temporal DiT，用 flow matching 训练。每两帧采样一次，条件历史 `H=2`，预测 8 个未来帧。语义 encoder 不直接用语言训练 DiT，主要看视觉 latent space 本身是否保留 action/task structure。
>
> **评测轴**
> 它的评测设计比普通视频生成论文强很多，分三大类：
>
> ```text
> 1. Planning / downstream policy
>    CEM action controllability
>    OpenVLA-7B inside world model rollout success
>    OOD distractor / OOD instruction robustness
>
> 2. Pixel fidelity / scene geometry
>    FID, SSIM, LPIPS, FVD, temporal LPIPS
>    point tracking consistency
>    WorldArena perceptual/geometric scores
>
> 3. Latent representation quality
>    IDM action recoverability
>    success classifier accuracy
> ```
>
> 这里最值得我们学习的是第 3 类。它不只看生成画面，而是问：
>
> ```text
> 生成出来的 latent 还能不能恢复动作？
> 还能不能区分任务是否成功？
> ```
>
> 这正是 awesome_wam 后续应该补的评测。
>
> **主要发现 1：semantic latent 更利于 policy 和 planning**
> 在 DiT-S 下，reconstruction latent 的 policy-in-world success 很低：
>
> ```text
> VAE:     consensus SR 0.169
> VA-VAE:  0.175
> Cosmos:  0.244
> ```
>
> semantic latent 明显更好：
>
> ```text
> V-JEPA 2.1:    0.344
> V-JEPA 2.1-96: 0.362
> SigLIP 2:      0.325
> SigLIP 2-96:   0.331
> Web-DINO-96:   0.300
> ```
>
> CEM action recovery 也更支持 semantic latent。比如 k=1 CEM error 越低越好：
>
> ```text
> VAE:        0.111
> Cosmos:     0.112
> V-JEPA 2.1: 0.084
> SigLIP 2:   0.082
> ```
>
> 这说明 semantic latent 的 dynamics 更 action-sensitive，更容易从 latent transition 反推出动作。
>
> **主要发现 2：semantic latent 保留更多 action 和 task 信息**
> IDM Pearson r 是动作可恢复性指标。语义 encoder 的 real latent 和 world-model latent 都明显更好。
>
> 例子：
>
> ```text
> VAE:
>   Enc r k=1: 0.507
>   WM  r k=1: 0.476
>
> V-JEPA 2.1:
>   Enc r k=1: 0.829
>   WM  r k=1: 0.781
>
> Web-DINO:
>   Enc r k=1: 0.820
>   WM  r k=1: 0.729
>
> SigLIP 2:
>   Enc r k=1: 0.772
>   WM  r k=1: 0.697
> ```
>
> success classifier 也类似。SigLIP 2 在 generated WM latent 上的 success accuracy 最高：
>
> ```text
> SigLIP 2 WM Acc: 0.823
> V-JEPA 2.1 WM Acc: 0.789
> Web-DINO WM Acc: 0.788
> VAE WM Acc: 0.716
> ```
>
> 这说明语义 latent 不只是“看起来懂物体”，而是真的保留了任务进展信息。
>
> **主要发现 3：视觉 fidelity 不能解释控制表现**
> 论文明确指出：视觉重建指标和控制表现不是一回事。
>
> reconstruction latent 在局部外观、像素细节上有优势，尤其模型变大后 VAE/Cosmos 的 visual metrics 会追上甚至超过 semantic latent。但它们仍然在 action recovery、success classifier、CEM planning 上落后。
>
> 这对 awesome_wam 很重要：
>
> ```text
> feature MSE 好
> decoded RGB 好
> PSNR/SSIM 好
> 都不能单独证明 world model 对机器人有用
> ```
>
> 必须加 action-facing metrics。
>
> **主要发现 4：模型变大能缩小 policy gap，但不能消除 action-centric gap**
> DiT 从 S 到 L 变大后，reconstruction latent 的视觉质量和 VLA rollout success 有提升。作者解释为：更大模型让 VAE/Cosmos 生成更好看，OpenVLA 作为视觉策略从更清晰图像中受益。
>
> 但 action-centric 指标仍然落后，例如 CEM、IDM、success classifier。这说明：
>
> ```text
> 大模型可以补视觉质量，
> 但不一定补 latent 本身缺少 action geometry 的问题。
> ```
>
> **主要发现 5：adapter 是双刃剑**
> S-VAE adapter 把 semantic feature 压到 `d=96`，通常让 diffusion 更容易训练，decoded rollout 也更好。
>
> 但 adapter 也可能损伤细粒度控制几何。论文说 adapter 往往提升高层任务完成和 denoising ease，但在这些指标上可能不如 native semantic feature：
>
> ```text
> CEM action error
> OOD robustness
> PCK coverage
> ```
>
> 对我们来说，这意味着：
>
> ```text
> S-VAE / tokenizer 不应该只以 reconstruction loss 为目标。
> 要检查压缩后 action recoverability 有没有下降。
> ```
>
> **对 awesome_wam 的直接启发**
> 我会把这篇论文的方法论记成四条：
>
> ```text
> 1. 研究 representation 时必须固定训练协议，只变 target/latent。
> 2. 不能用视觉指标替代机器人指标。
> 3. 每个 latent/tokenizer 都要测 action recoverability。
> 4. 压缩 adapter/tokenizer 可能方便预测，但可能破坏控制几何。
> ```
>
> 所以未来我们做 awesome_wam，如果比较 RGB/depth/seg/DINO/SigLIP/SVG 或 S-VAE/PV-VAE/DeltaTok，我会建议至少同时报告：
>
> ```text
> world metric:
>   feature MSE / cosine / decoded proxy
>
> action metric:
>   action L1/L2
>   LIBERO success
>
> latent probe:
>   IDM action recovery
>   task-progress/success classifier
>
> shortcut diagnostic:
>   copy baseline
>   delta_ratio
> ```
>
> 这篇论文真正教我们的不是“选 DINO 就完了”，而是：**一个机器人 world latent 是否有价值，要看它是否保留 action geometry 和 task progress，而不是看它能不能漂亮地重构画面。**
>
> 来源：[arXiv 2605.06388](https://arxiv.org/pdf/2605.06388)，[semantic-wm GitHub](https://github.com/chandar-lab/semantic-wm)
</details>

我重点重读了 `Reconstruction or Semantics?` 的实验和 findings。它最核心的地方不是“semantic latent 赢了”，而是它把 **latent space choice** 设计成一个严格受控变量来研究。

**实验设计**
论文的实验问题是：

```text
在机器人 action-conditioned latent diffusion world model 中，
到底什么 latent space 更有用：
重构型 latent，还是语义型 latent？
```

它固定了这些因素：

```text
dataset
history length
action conditioning
DiT transition architecture
optimizer
training schedule
```

只改变：

```text
encoder f_phi
optional adapter alpha_psi
decoder path
```

这点很关键。它不是随便比较一堆模型，而是在控制变量下问：**latent interface 本身对机器人 world model 有什么影响。**

数据集是 **Bridge V2**，约 60K WidowX 250 机器人 demonstrations，包含 RGB、7-DoF action 和语言指令。额外用 SOAR 的成功/失败轨迹训练 success classifier。

比较的 latent 分两类：

```text
Reconstruction-aligned:
  SD3 VAE, VA-VAE, Cosmos

Semantics-aligned:
  V-JEPA 2.1, Web-DINO, SigLIP 2
```

语义 latent 又分两种：

```text
native high-dimensional feature
S-VAE adapter 压缩到 d=96
```

transition model 是 spatial-temporal DiT，用 flow matching 训练。每两帧采样一次，条件历史 `H=2`，预测 8 个未来帧。语义 encoder 不直接用语言训练 DiT，主要看视觉 latent space 本身是否保留 action/task structure。

**评测轴**
它的评测设计比普通视频生成论文强很多，分三大类：

```text
1. Planning / downstream policy
   CEM action controllability
   OpenVLA-7B inside world model rollout success
   OOD distractor / OOD instruction robustness

2. Pixel fidelity / scene geometry
   FID, SSIM, LPIPS, FVD, temporal LPIPS
   point tracking consistency
   WorldArena perceptual/geometric scores

3. Latent representation quality
   IDM action recoverability
   success classifier accuracy
```

这里最值得我们学习的是第 3 类。它不只看生成画面，而是问：

```text
生成出来的 latent 还能不能恢复动作？
还能不能区分任务是否成功？
```

这正是 awesome_wam 后续应该补的评测。

**主要发现 1：semantic latent 更利于 policy 和 planning**
在 DiT-S 下，reconstruction latent 的 policy-in-world success 很低：

```text
VAE:     consensus SR 0.169
VA-VAE:  0.175
Cosmos:  0.244
```

semantic latent 明显更好：

```text
V-JEPA 2.1:    0.344
V-JEPA 2.1-96: 0.362
SigLIP 2:      0.325
SigLIP 2-96:   0.331
Web-DINO-96:   0.300
```

CEM action recovery 也更支持 semantic latent。比如 k=1 CEM error 越低越好：

```text
VAE:        0.111
Cosmos:     0.112
V-JEPA 2.1: 0.084
SigLIP 2:   0.082
```

这说明 semantic latent 的 dynamics 更 action-sensitive，更容易从 latent transition 反推出动作。

**主要发现 2：semantic latent 保留更多 action 和 task 信息**
IDM Pearson r 是动作可恢复性指标。语义 encoder 的 real latent 和 world-model latent 都明显更好。

例子：

```text
VAE:
  Enc r k=1: 0.507
  WM  r k=1: 0.476

V-JEPA 2.1:
  Enc r k=1: 0.829
  WM  r k=1: 0.781

Web-DINO:
  Enc r k=1: 0.820
  WM  r k=1: 0.729

SigLIP 2:
  Enc r k=1: 0.772
  WM  r k=1: 0.697
```

success classifier 也类似。SigLIP 2 在 generated WM latent 上的 success accuracy 最高：

```text
SigLIP 2 WM Acc: 0.823
V-JEPA 2.1 WM Acc: 0.789
Web-DINO WM Acc: 0.788
VAE WM Acc: 0.716
```

这说明语义 latent 不只是“看起来懂物体”，而是真的保留了任务进展信息。

**主要发现 3：视觉 fidelity 不能解释控制表现**
论文明确指出：视觉重建指标和控制表现不是一回事。

reconstruction latent 在局部外观、像素细节上有优势，尤其模型变大后 VAE/Cosmos 的 visual metrics 会追上甚至超过 semantic latent。但它们仍然在 action recovery、success classifier、CEM planning 上落后。

这对 awesome_wam 很重要：

```text
feature MSE 好
decoded RGB 好
PSNR/SSIM 好
都不能单独证明 world model 对机器人有用
```

必须加 action-facing metrics。

**主要发现 4：模型变大能缩小 policy gap，但不能消除 action-centric gap**
DiT 从 S 到 L 变大后，reconstruction latent 的视觉质量和 VLA rollout success 有提升。作者解释为：更大模型让 VAE/Cosmos 生成更好看，OpenVLA 作为视觉策略从更清晰图像中受益。

但 action-centric 指标仍然落后，例如 CEM、IDM、success classifier。这说明：

```text
大模型可以补视觉质量，
但不一定补 latent 本身缺少 action geometry 的问题。
```

**主要发现 5：adapter 是双刃剑**
S-VAE adapter 把 semantic feature 压到 `d=96`，通常让 diffusion 更容易训练，decoded rollout 也更好。

但 adapter 也可能损伤细粒度控制几何。论文说 adapter 往往提升高层任务完成和 denoising ease，但在这些指标上可能不如 native semantic feature：

```text
CEM action error
OOD robustness
PCK coverage
```

对我们来说，这意味着：

```text
S-VAE / tokenizer 不应该只以 reconstruction loss 为目标。
要检查压缩后 action recoverability 有没有下降。
```

**对 awesome_wam 的直接启发**
我会把这篇论文的方法论记成四条：

```text
1. 研究 representation 时必须固定训练协议，只变 target/latent。
2. 不能用视觉指标替代机器人指标。
3. 每个 latent/tokenizer 都要测 action recoverability。
4. 压缩 adapter/tokenizer 可能方便预测，但可能破坏控制几何。
```

所以未来我们做 awesome_wam，如果比较 RGB/depth/seg/DINO/SigLIP/SVG 或 S-VAE/PV-VAE/DeltaTok，我会建议至少同时报告：

```text
world metric:
  feature MSE / cosine / decoded proxy

action metric:
  action L1/L2
  LIBERO success

latent probe:
  IDM action recovery
  task-progress/success classifier

shortcut diagnostic:
  copy baseline
  delta_ratio
```

这篇论文真正教我们的不是“选 DINO 就完了”，而是：**一个机器人 world latent 是否有价值，要看它是否保留 action geometry 和 task progress，而不是看它能不能漂亮地重构画面。**

来源：[arXiv 2605.06388](https://arxiv.org/pdf/2605.06388)，[semantic-wm GitHub](https://github.com/chandar-lab/semantic-wm)

> 这是师兄跟我的一些聊天记录，告诉我了大概要做什么，请你告诉我下一步该如何展开
>
> User context
>
> Attachments:
> - Feature tokenizer.pdf: `C:/Users/Lenovo/Downloads/Feature tokenizer.pdf`
> Images:
> - `data:image/png;base64,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`
> - `data:image/png;base64,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OnjwYNn3kls6Pem8TqnUiC5cuOC8ZuZ1NdyPm0gktX//Pt9jMefFfh3d/x/JZDJ65JFvluwCTqcnPZ8nWee33GdPZT7P60EXMLA1+VYAd+7cuaETP9baBZxKjRRVE9KuapkJldV0i3Z3P6lMJlP0hWlcuHBBra2tnsrXww93KZfLaWJiXENDw3ruue85x9va2uqMQZTrS8gsl9VlbTNfZqnUiPr6Tmj//n1F5yRW6LocH39RY2NnnW3taktHx4GiLjr3+KVSKp0TP08//V3PaxaJHFI83q3BwdNOV9/o6BnFClVM8wVj2mKxI4pEDimRSHr26+4ay+VynkBuqnnmfLi7J+3qSXf3k8rlcspkMmpra/O8Rm72OXTf7IrVyEiq6HWJFbrwVKK71gSLo0efcoLR4OBpzc7OOo9jtq02/HV1RRSNRp3u1kqv2+uvT2l6+lJRF7AkRSKHnC/6eLxb+fycVPgczMzMeAJNNdZyPu3KV1dhOIS7Inffffc7XduJRNJTPc3lckWf90jkkM6de0mxwvuyo+NA0XvD7eDBg5qZmbGbfZn36ptvvmEvcrS1tamtrc036JXy9NPf1fT0paLwNzMz4zzX/fv3lf0sT09f0sGDBz1tBw8edN6bpT6vcoW/Sp9NANubbwCsqfE2r7f6dyMymYzni9Wv+6kazz33PalMKDNfwqabR4WQduHCBanwV737f9TT05ecL0nzP0z3X87nzr1UcmJCIpFULpdz7g8OnlZT022eL0237u4nNT7+olQijExPX9L09KU1n6dK58SPu8ozMTGuubm8du++07OO2xNPxDU9fckzViqbPa/Ozk7Pei+88LzzbxPGSxkcPO37HNPpSU9F5+jRp3y/0PzOod+tkoMHD2puLl+0fz8mtEYihzQwcMr5MraDTDXc56oa16792m7S6OgZ5322e/edTkVQhff67Oysa+3y1no+p6aympvL67777pcKn7lcLqdwuMn5A6y9fa9zTCMjKU9IvnDhgpqaws59FT6P5j1m3pdm/7ZUakThcFNVYS0e73a62ssF7pmZGafiW274hPuPlytX3iuqFtrH1dv7bNH73O3116c85y0e71Y43OSE7HKf12o/mwC2N98AuLDwqa5fv+7cX10t6iVeE/d4Pfvm7v5zi0ajnvXK/TVshMNNnm2MkZGUZmdnPdU09/6mpy9p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`
> - `data:image/png;base64,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`

<details><summary>7 previous messages</summary>

> 我先把师兄给的 PDF 文件确认一下，看看它是不是同一篇 `Reconstruction or Semantics?` 中文版，还是另一个专门讲 feature tokenizer / PV-VAE 的材料；然后我会把聊天里的要求翻译成一套可执行的下一步路线。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `Get-ChildItem -LiteralPath "C:\Users\Lenovo\Downloads\Feature tokenizer.pdf" | Format-List FullName,Length,LastWriteTime`
>
> </details>
>
> PDF 文件不大，看起来像师兄专门整理给你的“Feature tokenizer”材料。我会用本地文档运行时抽取文本，不需要联网。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `& "C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" - <<'PY'
> import importlib.util
> mods=['pypdf','PyPDF2','pdfplumber','fitz','pymupdf']
> for m in mods:
>     print(m, bool(importlib.util.find_spec(m)))
> PY`
> - Ran `& "C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" -e "for (const m of ['pdf-parse','pdfjs-dist']) { try { console.log(m, require.resolve(m)); } catch(e) { console.log(m, false); } }"`
> - Ran `@'
> import importlib.util
> mods=['pypdf','PyPDF2','pdfplumber','fitz','pymupdf']
> for m in mods:
>     print(m, bool(importlib.util.find_spec(m)))
> '@ | & "C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -`
>
> </details>
>
> 运行时里有 `pypdf`，可以直接抽 PDF 文本。我接下来只抽标题、摘要/方法附近，不会把整篇都塞进上下文。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `@'
> from pypdf import PdfReader
> from pathlib import Path
> path=Path(r"C:\Users\Lenovo\Downloads\Feature tokenizer.pdf")
> reader=PdfReader(str(path))
> print('pages', len(reader.pages))
> for i,p in enumerate(reader.pages[:8]):
>     text=p.extract_text() or ''
>     print(f'\n--- PAGE {i+1} ---')
>     print(text[:3500])
> '@ | & "C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -`
>
> </details>
>
> 明白。结合聊天记录和 `Feature tokenizer.pdf`，师兄现在给你的方向不是“马上做 FastWAM 主实验”，而是先做一个 **feature tokenizer 基础设施**。
>
> 也就是：
>
> ```text
> 大量视频/机器人数据
>   -> frozen visual encoder: DINO/SVG/SigLIP
>   -> 高维 feature
>   -> 训练 tokenizer / adapter / VAE
>   -> 得到高压缩 latent
>   -> 之后再接 WAM / policy
> ```
>
> **师兄的意思**
> 他主要在说几件事：
>
> 1. **现在要训 tokenizer**
>    不是先训完整 world model，而是先把视觉 feature 压缩器训出来。
>
> 2. **已有小数据实验不够**
>    你们可能已经在 LIBERO 或小规模数据上跑过 PV-VAE / DeltaTok smoke，但下一步需要更大、更杂的数据。
>
> 3. **需要整理统一数据格式**
>    Droid、Behavior-1K、OXE、Ego4D 等都要转成同一种 clip 格式，这样一个 dataloader 就能读。
>
> 4. **目标是高压缩 VAE / adapter**
>    因为 Wan2.2 的 3D VAE latent 还是太多，师兄希望参考 PV-VAE 或 Semantic-WM 的 S-VAE，做一个更高压缩的 feature tokenizer。
>
> 5. **tokenizer 质量可以用 LARY 思路验证**
>    不只看 reconstruction loss，还要看压缩后的 latent 是否还能恢复 action、语义动作、任务进展。
>
> **下一步怎么展开**
> 我建议分成 4 个阶段，不要一上来就做复杂模型。
>
> **第一步：先定 tokenizer 的输入输出 contract**
> 先写清楚我们到底压什么。
>
> 建议先固定一种 teacher：
>
> ```text
> DINOv3 或 SVG-DINO-P
> ```
>
> 然后统一 feature shape：
>
> ```text
> RGB clip
>   -> frozen encoder
>   -> x: [B, V, T, N, D]
> ```
>
> 例如：
>
> ```text
> V = camera 数
> T = clip 帧数
> N = spatial token 数，例如 16x16
> D = feature dim，例如 384 / 1024
> ```
>
> 然后 tokenizer 输出：
>
> ```text
> Per-frame S-VAE:
>   [B,V,T,N,D] -> [B,V,T,N,d]
>
> PV-VAE:
>   [B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
>
> DeltaTok:
>   x_t, x_{t+k} -> z_delta [B,V,M,d]
> ```
>
> 这一步非常重要，否则后面 dataloader、loss、eval 都会乱。
>
> **第二步：整理统一数据 dataloader**
> 这是师兄现在最明确让你做的事。
>
> 把这些数据源统一成同一种格式：
>
> ```text
> DROID
> /data/user/jhe724/workspace/data/droid_success
>
> Behavior-1K
> /data/user/jhe724/workspace/data/2025-challenge-demos
>
> OXE
> /data/user/jhe724/workspace/data/OXE
>
> Ego4D
> /data/user/jhe724/workspace/data/Ego4D
> ```
>
> 统一 dataloader 最少返回：
>
> ```python
> {
>   "images": [T, V, C, H, W],
>   "actions": [T-1, A] or None,
>   "instruction": str or None,
>   "dataset_name": str,
>   "episode_id": str,
>   "timestamps": [...]
> }
> ```
>
> 注意分两类数据：
>
> ```text
> 有 action 的机器人数据:
>   DROID / OXE / LIBERO / RoboCasa / Behavior-1K
>   可以做 reconstruction + prediction + action probe
>
> 无 robot action 的自然视频:
>   Ego4D
>   可以做 feature reconstruction / temporal prediction
>   但不能直接做 robot action recoverability
> ```
>
> 所以 dataloader 要允许 `actions=None`。
>
> **第三步：先实现最稳的 tokenizer baseline**
> 我建议先做 **Per-frame S-VAE / Channel Adapter**，因为它最稳，最适合做基础设施验证。
>
> 目标：
>
> ```text
> x_t -> encoder adapter -> z_t -> decoder adapter -> x_hat_t
> ```
>
> loss：
>
> ```text
> feature_mse(x_hat, x)
> + cosine_loss(x_hat, x)
> + beta * KL
> ```
>
> 先不要急着做 temporal compression。先证明：
>
> ```text
> 大规模数据能读
> frozen teacher 能稳定抽 feature
> adapter 能稳定压缩和重建
> 可视化/metric pipeline 能跑
> ```
>
> 这个 baseline 对应师兄说的：
>
> ```text
> 训练一个 adapter
> 这样可以接入高纬度的特征
> ```
>
> **第四步：再做 PV-VAE 和 DeltaTok**
> 等数据和 S-VAE baseline 稳了，再做两个更有研究味的版本。
>
> PV-VAE：
>
> ```text
> 17 frames -> 5 latent groups
> ```
>
> 重点看：
>
> ```text
> future_mse
> static_future_mse
> delta_ratio
> ```
>
> 防止它只学 copy 当前帧。
>
> DeltaTok：
>
> ```text
> x_t, x_{t+k} -> z_delta
> x_t, z_delta -> x_hat_{t+k}
> ```
>
> 这个最有潜力，因为它更符合机器人视频：
>
> ```text
> 背景多数不动
> 真正重要的是 gripper/object/contact 的变化
> ```
>
> 可以先 `M=1` 个 delta token，再扩展到 `M=4/8`。
>
> **怎么验证 tokenizer 好不好**
> 不要只看 reconstruction loss。建议至少有四类指标：
>
> ```text
> 1. Feature reconstruction:
>    MSE / cosine
>
> 2. Copy baseline gap:
>    pred_mse vs copy_mse
>
> 3. Dynamics:
>    delta_ratio
>    high-motion 区域误差
>
> 4. LARY-style probe:
>    z_t, z_future -> action chunk
>    或 z_delta -> action chunk
> ```
>
> 如果压缩后 latent 很小，但 action probe 还能恢复动作，说明它保留了 action-relevant information。
>
> **你现在最该做的具体事情**
> 我建议你下一步不是直接训模型，而是先做一个 `feature_tokenizer` 工作包：
>
> ```text
> 1. 写清楚 tokenizer 输入输出 shape。
> 2. 整理多数据源统一 dataloader。
> 3. 先支持 DINOv3/SVG-P frozen feature extraction。
> 4. 跑一个 per-frame S-VAE baseline。
> 5. 加 feature MSE/cosine/copy baseline/LARY-style action probe。
> 6. 再扩展 PV-VAE 和 DeltaTok。
> ```
>
> 一句话说：
>
> **现在的下一步是搭 feature tokenizer 的数据与评测地基：先把多数据集统一成 clip dataloader，再训练一个稳定的高压缩 S-VAE adapter，最后用 PV-VAE/DeltaTok 做更强的时序和变化压缩。**
</details>

明白。结合聊天记录和 `Feature tokenizer.pdf`，师兄现在给你的方向不是“马上做 FastWAM 主实验”，而是先做一个 **feature tokenizer 基础设施**。

也就是：

```text
大量视频/机器人数据
  -> frozen visual encoder: DINO/SVG/SigLIP
  -> 高维 feature
  -> 训练 tokenizer / adapter / VAE
  -> 得到高压缩 latent
  -> 之后再接 WAM / policy
```

**师兄的意思**
他主要在说几件事：

1. **现在要训 tokenizer**
   不是先训完整 world model，而是先把视觉 feature 压缩器训出来。

2. **已有小数据实验不够**
   你们可能已经在 LIBERO 或小规模数据上跑过 PV-VAE / DeltaTok smoke，但下一步需要更大、更杂的数据。

3. **需要整理统一数据格式**
   Droid、Behavior-1K、OXE、Ego4D 等都要转成同一种 clip 格式，这样一个 dataloader 就能读。

4. **目标是高压缩 VAE / adapter**
   因为 Wan2.2 的 3D VAE latent 还是太多，师兄希望参考 PV-VAE 或 Semantic-WM 的 S-VAE，做一个更高压缩的 feature tokenizer。

5. **tokenizer 质量可以用 LARY 思路验证**
   不只看 reconstruction loss，还要看压缩后的 latent 是否还能恢复 action、语义动作、任务进展。

**下一步怎么展开**
我建议分成 4 个阶段，不要一上来就做复杂模型。

**第一步：先定 tokenizer 的输入输出 contract**
先写清楚我们到底压什么。

建议先固定一种 teacher：

```text
DINOv3 或 SVG-DINO-P
```

然后统一 feature shape：

```text
RGB clip
  -> frozen encoder
  -> x: [B, V, T, N, D]
```

例如：

```text
V = camera 数
T = clip 帧数
N = spatial token 数，例如 16x16
D = feature dim，例如 384 / 1024
```

然后 tokenizer 输出：

```text
Per-frame S-VAE:
  [B,V,T,N,D] -> [B,V,T,N,d]

PV-VAE:
  [B,V,1+T,N,D] -> [B,V,1+T/4,N,d]

DeltaTok:
  x_t, x_{t+k} -> z_delta [B,V,M,d]
```

这一步非常重要，否则后面 dataloader、loss、eval 都会乱。

**第二步：整理统一数据 dataloader**
这是师兄现在最明确让你做的事。

把这些数据源统一成同一种格式：

```text
DROID
/data/user/jhe724/workspace/data/droid_success

Behavior-1K
/data/user/jhe724/workspace/data/2025-challenge-demos

OXE
/data/user/jhe724/workspace/data/OXE

Ego4D
/data/user/jhe724/workspace/data/Ego4D
```

统一 dataloader 最少返回：

```python
{
  "images": [T, V, C, H, W],
  "actions": [T-1, A] or None,
  "instruction": str or None,
  "dataset_name": str,
  "episode_id": str,
  "timestamps": [...]
}
```

注意分两类数据：

```text
有 action 的机器人数据:
  DROID / OXE / LIBERO / RoboCasa / Behavior-1K
  可以做 reconstruction + prediction + action probe

无 robot action 的自然视频:
  Ego4D
  可以做 feature reconstruction / temporal prediction
  但不能直接做 robot action recoverability
```

所以 dataloader 要允许 `actions=None`。

**第三步：先实现最稳的 tokenizer baseline**
我建议先做 **Per-frame S-VAE / Channel Adapter**，因为它最稳，最适合做基础设施验证。

目标：

```text
x_t -> encoder adapter -> z_t -> decoder adapter -> x_hat_t
```

loss：

```text
feature_mse(x_hat, x)
+ cosine_loss(x_hat, x)
+ beta * KL
```

先不要急着做 temporal compression。先证明：

```text
大规模数据能读
frozen teacher 能稳定抽 feature
adapter 能稳定压缩和重建
可视化/metric pipeline 能跑
```

这个 baseline 对应师兄说的：

```text
训练一个 adapter
这样可以接入高纬度的特征
```

**第四步：再做 PV-VAE 和 DeltaTok**
等数据和 S-VAE baseline 稳了，再做两个更有研究味的版本。

PV-VAE：

```text
17 frames -> 5 latent groups
```

重点看：

```text
future_mse
static_future_mse
delta_ratio
```

防止它只学 copy 当前帧。

DeltaTok：

```text
x_t, x_{t+k} -> z_delta
x_t, z_delta -> x_hat_{t+k}
```

这个最有潜力，因为它更符合机器人视频：

```text
背景多数不动
真正重要的是 gripper/object/contact 的变化
```

可以先 `M=1` 个 delta token，再扩展到 `M=4/8`。

**怎么验证 tokenizer 好不好**
不要只看 reconstruction loss。建议至少有四类指标：

```text
1. Feature reconstruction:
   MSE / cosine

2. Copy baseline gap:
   pred_mse vs copy_mse

3. Dynamics:
   delta_ratio
   high-motion 区域误差

4. LARY-style probe:
   z_t, z_future -> action chunk
   或 z_delta -> action chunk
```

如果压缩后 latent 很小，但 action probe 还能恢复动作，说明它保留了 action-relevant information。

**你现在最该做的具体事情**
我建议你下一步不是直接训模型，而是先做一个 `feature_tokenizer` 工作包：

```text
1. 写清楚 tokenizer 输入输出 shape。
2. 整理多数据源统一 dataloader。
3. 先支持 DINOv3/SVG-P frozen feature extraction。
4. 跑一个 per-frame S-VAE baseline。
5. 加 feature MSE/cosine/copy baseline/LARY-style action probe。
6. 再扩展 PV-VAE 和 DeltaTok。
```

一句话说：

**现在的下一步是搭 feature tokenizer 的数据与评测地基：先把多数据集统一成 clip dataloader，再训练一个稳定的高压缩 S-VAE adapter，最后用 PV-VAE/DeltaTok 做更强的时序和变化压缩。**

> 我怕你的研究方法和判断可能不是很准确，请你先读一下项目文件夹下的几个.md文档，看下师兄他们前期写的一些文档，里面可能有更加准确的研究进度、研究计划和方法等等，请你根据这些md文档，以及之前给你发的那些聊天记录截图，再来告诉我下一步该做什么
>
> User context
>
> Attachments:
> - current_fastwam_experiments.md: `./wam/current_fastwam_experiments.md`
> - dinov3_fastwam_plan.md: `./wam/dinov3_fastwam_plan.md`
> - fastwam_codebase_reading_guide.md: `./wam/fastwam_codebase_reading_guide.md`
> - interaction_aware_latent_action.md: `./wam/interaction_aware_latent_action.md`
> - novelty_check.md: `./wam/novelty_check.md`
> - pi05_native_denoising_gen_expert.md: `./wam/pi05_native_denoising_gen_expert.md`
> - plan.md: `./wam/plan.md`
> - research_refine.md: `./wam/research_refine.md`
> - story.md: `./wam/story.md`
> - tokenizer_methods.md: `./wam/tokenizer_methods.md`
> - action_salient_world_modeling.md: `./wam/action_salient_world_modeling.md`

<details><summary>5 previous messages</summary>

> 你担心得对，这类研究方向如果只凭几篇论文和聊天记录，很容易把“师兄真正想推进的工程节点”和“我们脑补的研究主线”混在一起。我现在按项目文档本身来校准：先读这些 `wam/*.md`，再把它们和聊天截图合并成下一步建议。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content wam\story.md`
> - Ran `Get-Content wam\research_refine.md`
> - Ran `Get-Content wam\novelty_check.md`
> - Ran `Get-Content wam\dinov3_fastwam_plan.md`
>
> </details>
>
> 到这里已经能看到前期文档的主线：一条是 FastWAM 表征目标的 controlled study，另一条是 tokenizer/adapter 作为“之后可接入更高维特征”的基础工作。继续读方法候选和当前实验状态，避免把两条线排反。
>
> <details><summary>Ran 7 commands</summary>
>
> - Ran `Get-Content wam\current_fastwam_experiments.md`
> - Ran `Get-Content wam\fastwam_codebase_reading_guide.md`
> - Ran `Get-Content wam\tokenizer_methods.md`
> - Ran `Get-Content wam\plan.md`
> - Ran `Get-Content wam\action_salient_world_modeling.md`
> - Ran `Get-Content wam\interaction_aware_latent_action.md`
> - Ran `Get-Content wam\pi05_native_denoising_gen_expert.md`
>
> </details>
>
> 你这个提醒是对的。读完这些 `.md` 后，我要修正一下前面的表述：**项目主线和师兄现在交代你的“下一步”不是完全同一层级。**
>
> 项目文档里的主线是：
>
> ```text
> FastWAM 表征目标研究：
> RGB / depth / seg / DINO / SigLIP / V-JEPA / SVG
> 在固定 FastWAM 协议下，比较哪个 future world target 更 action-aligned。
> ```
>
> 但师兄聊天和 `Feature tokenizer.pdf` 里给你的更直接任务是：
>
> ```text
> 先做 feature tokenizer / adapter：
> 把高维语义特征压缩成更小的 latent，
> 之后再接入 WAM 或更大规模训练。
> ```
>
> 所以，**你现在最该展开的是 feature tokenizer 这条支线的工程地基**，而不是马上继续堆 FastWAM 在线评测或 interaction-saliency 方法。
>
> **校准后的下一步**
> 第一优先级应该是：
>
> ```text
> 整理多数据集 -> 统一 dataloader -> 训练第一个稳定 feature adapter/tokenizer baseline
> ```
>
> 这和师兄说的几句话完全对应：
>
> ```text
> “现在要训 tokenizer”
> “需要用更大规模的数据集训练”
> “把数据都下好”
> “转成同样的格式应该就可以用同一个 dataloader”
> “训练一个 adapter”
> “接入高纬度的特征”
> ```
>
> **具体怎么做**
> 我建议你把下一步拆成 5 个很明确的阶段。
>
> **1. 先做数据清单和统一格式**
> 先不要训练模型。第一步是把师兄给的数据路径整理成一个 manifest。
>
> 数据源包括：
>
> ```text
> DROID:
> /data/user/jhe724/workspace/data/droid_success
>
> Behavior-1K:
> /data/user/jhe724/workspace/data/2025-challenge-demos
>
> OXE:
> /data/user/jhe724/workspace/data/OXE
>
> Ego4D:
> /data/user/jhe724/workspace/data/Ego4D
> ```
>
> 统一成一种 clip 样本格式：
>
> ```python
> {
>   "dataset": "droid",
>   "episode_id": "...",
>   "frames": [T, V, C, H, W],
>   "actions": [T-1, A] or None,
>   "instruction": str or None,
>   "fps": float,
>   "timestamps": [...]
> }
> ```
>
> 这里要注意：
>
> ```text
> DROID / OXE / Behavior / LIBERO:
>   有 action，可以做 LARY-style action probe
>
> Ego4D:
>   大概率无机器人 action，只适合做 feature reconstruction / temporal prediction
> ```
>
> 所以 dataloader 必须支持 `actions=None`。
>
> **2. 先确定 tokenizer 的输入输出 contract**
> 这一点必须写死，否则后面实验会乱。
>
> 当前推荐先用一种 teacher feature，例如：
>
> ```text
> DINOv3-S / SVG-DINO-P
> ```
>
> 输入：
>
> ```text
> RGB clip -> frozen encoder -> x
> x: [B, V, T, N, D]
> ```
>
> 输出根据 tokenizer 不同：
>
> ```text
> Per-frame S-VAE:
> [B,V,T,N,D] -> [B,V,T,N,d]
>
> PV-VAE:
> [B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
>
> DeltaTok:
> x_t, x_{t+k} -> z_delta [B,V,M,d]
> ```
>
> 第一版不要贪多，建议先固定：
>
> ```text
> T = 16 或 17
> image size = 224/256
> teacher = DINOv3 或 SVG-P
> d = 96 或 128
> ```
>
> **3. 先实现最稳的 Per-frame S-VAE / Channel Adapter**
> 从文档看，`tokenizer_methods.md` 里已经明确说：
>
> ```text
> Per-frame S-VAE 是最稳 baseline，但还没有 standalone 实现。
> ```
>
> 这正好对应师兄说的“训练一个 adapter”。
>
> 第一版模型就是：
>
> ```text
> x_t -> adapter encoder -> z_t -> adapter decoder -> x_hat_t
> ```
>
> loss：
>
> ```text
> feature_mse(x_hat, x)
> + cosine_loss(x_hat, x)
> + beta * KL
> ```
>
> 它不做时序压缩，所以最适合验证：
>
> ```text
> 数据能不能读
> teacher feature 能不能稳定抽
> adapter 能不能压缩
> feature reconstruction 是否正常
> ```
>
> 不要一上来就做 PV-VAE 或 DeltaTok，因为那样数据、模型、评测、动态 shortcut 会混在一起。
>
> **4. 同时补一个 tokenizer 质量评测，不只看 reconstruction**
> 这里要吸收 `Reconstruction or Semantics?` 和 LARY 的方法论。
>
> 最低限度要有：
>
> ```text
> feature_mse
> feature_cosine
> compression ratio
> PCA visualization
> ```
>
> 但这还不够。对于有 action 的数据，还要加：
>
> ```text
> LARY-style probe:
> z_t, z_{t+k} -> action chunk
> ```
>
> 也就是说，比较：
>
> ```text
> 原始 teacher feature 的 action recoverability
> 压缩后 z 的 action recoverability
> ```
>
> 如果 adapter reconstruction loss 很低，但 action probe 掉很多，那说明这个 tokenizer 把控制相关几何压坏了。
>
> **5. 等 baseline 稳了，再做 PV-VAE 和 DeltaTok**
> 第二阶段再比较：
>
> ```text
> PV-VAE-style temporal tokenizer
> Delta transition tokenizer
> ```
>
> PV-VAE 要重点防：
>
> ```text
> static copy shortcut
> ```
>
> 所以必须报：
>
> ```text
> future_mse
> static_future_mse / copy_mse
> delta_ratio
> ```
>
> DeltaTok 更像后续主线，因为它压的是变化：
>
> ```text
> x_t, x_{t+k} -> z_delta
> x_t, z_delta -> x_hat_future
> ```
>
> 这更适合机器人场景里的“静态背景 + 局部交互变化”。
>
> **我建议你现在的实际任务清单**
> 按优先级排：
>
> 1. 写一个 `feature tokenizer` 的数据 manifest 规范，列出 DROID / Behavior / OXE / Ego4D 每个数据源有什么字段、fps、camera、action、instruction。
>
> 2. 做统一 dataloader，先只保证能 sample 出统一 clip batch。
>
> 3. 确认当前 `external/openpi` 里 tokenizer 相关实现是否完整。
>    我刚才看到本地若干被文档引用的文件可能是空文件或镜像不完整，例如 `predictive_feature_vae.py`、`latent_action.py`、`delta_tokenizer.py`。落地前要和 HPC 上真实代码对齐一下。
>
> 4. 实现/整理 standalone `semantic_feature_vae.py`，先做 per-frame S-VAE / channel adapter。
>
> 5. 用一个小混合数据子集跑 smoke：
>    ```text
>    1k / 5k steps
>    batch shape 正确
>    loss 稳定下降
>    feature PCA 正常
>    ```
>
> 6. 加 LARY-style action probe，先在有 action 的数据上验证压缩 latent 是否保留 action 信息。
>
> 7. 再决定是上 PV-VAE 还是 DeltaTok。
>    如果 per-frame adapter 保留 action 信息很好，但压缩率不够，再推 PV-VAE；如果 PV-VAE copy shortcut 明显，就转 DeltaTok。
>
> **一句话结论**
> 根据项目文档和师兄聊天，下一步不是直接做新的 WAM 方法，也不是马上开大规模 LIBERO 评测，而是先把 **feature tokenizer 的数据、adapter baseline 和 action-aware 评测** 搭起来。这个东西做好后，才有资格接入 FastWAM，去验证“高压缩 semantic latent 是否比原始高维 feature / Wan VAE latent 更适合 robot world modeling”。
</details>

你这个提醒是对的。读完这些 `.md` 后，我要修正一下前面的表述：**项目主线和师兄现在交代你的“下一步”不是完全同一层级。**

项目文档里的主线是：

```text
FastWAM 表征目标研究：
RGB / depth / seg / DINO / SigLIP / V-JEPA / SVG
在固定 FastWAM 协议下，比较哪个 future world target 更 action-aligned。
```

但师兄聊天和 `Feature tokenizer.pdf` 里给你的更直接任务是：

```text
先做 feature tokenizer / adapter：
把高维语义特征压缩成更小的 latent，
之后再接入 WAM 或更大规模训练。
```

所以，**你现在最该展开的是 feature tokenizer 这条支线的工程地基**，而不是马上继续堆 FastWAM 在线评测或 interaction-saliency 方法。

**校准后的下一步**
第一优先级应该是：

```text
整理多数据集 -> 统一 dataloader -> 训练第一个稳定 feature adapter/tokenizer baseline
```

这和师兄说的几句话完全对应：

```text
“现在要训 tokenizer”
“需要用更大规模的数据集训练”
“把数据都下好”
“转成同样的格式应该就可以用同一个 dataloader”
“训练一个 adapter”
“接入高纬度的特征”
```

**具体怎么做**
我建议你把下一步拆成 5 个很明确的阶段。

**1. 先做数据清单和统一格式**
先不要训练模型。第一步是把师兄给的数据路径整理成一个 manifest。

数据源包括：

```text
DROID:
/data/user/jhe724/workspace/data/droid_success

Behavior-1K:
/data/user/jhe724/workspace/data/2025-challenge-demos

OXE:
/data/user/jhe724/workspace/data/OXE

Ego4D:
/data/user/jhe724/workspace/data/Ego4D
```

统一成一种 clip 样本格式：

```python
{
  "dataset": "droid",
  "episode_id": "...",
  "frames": [T, V, C, H, W],
  "actions": [T-1, A] or None,
  "instruction": str or None,
  "fps": float,
  "timestamps": [...]
}
```

这里要注意：

```text
DROID / OXE / Behavior / LIBERO:
  有 action，可以做 LARY-style action probe

Ego4D:
  大概率无机器人 action，只适合做 feature reconstruction / temporal prediction
```

所以 dataloader 必须支持 `actions=None`。

**2. 先确定 tokenizer 的输入输出 contract**
这一点必须写死，否则后面实验会乱。

当前推荐先用一种 teacher feature，例如：

```text
DINOv3-S / SVG-DINO-P
```

输入：

```text
RGB clip -> frozen encoder -> x
x: [B, V, T, N, D]
```

输出根据 tokenizer 不同：

```text
Per-frame S-VAE:
[B,V,T,N,D] -> [B,V,T,N,d]

PV-VAE:
[B,V,1+T,N,D] -> [B,V,1+T/4,N,d]

DeltaTok:
x_t, x_{t+k} -> z_delta [B,V,M,d]
```

第一版不要贪多，建议先固定：

```text
T = 16 或 17
image size = 224/256
teacher = DINOv3 或 SVG-P
d = 96 或 128
```

**3. 先实现最稳的 Per-frame S-VAE / Channel Adapter**
从文档看，`tokenizer_methods.md` 里已经明确说：

```text
Per-frame S-VAE 是最稳 baseline，但还没有 standalone 实现。
```

这正好对应师兄说的“训练一个 adapter”。

第一版模型就是：

```text
x_t -> adapter encoder -> z_t -> adapter decoder -> x_hat_t
```

loss：

```text
feature_mse(x_hat, x)
+ cosine_loss(x_hat, x)
+ beta * KL
```

它不做时序压缩，所以最适合验证：

```text
数据能不能读
teacher feature 能不能稳定抽
adapter 能不能压缩
feature reconstruction 是否正常
```

不要一上来就做 PV-VAE 或 DeltaTok，因为那样数据、模型、评测、动态 shortcut 会混在一起。

**4. 同时补一个 tokenizer 质量评测，不只看 reconstruction**
这里要吸收 `Reconstruction or Semantics?` 和 LARY 的方法论。

最低限度要有：

```text
feature_mse
feature_cosine
compression ratio
PCA visualization
```

但这还不够。对于有 action 的数据，还要加：

```text
LARY-style probe:
z_t, z_{t+k} -> action chunk
```

也就是说，比较：

```text
原始 teacher feature 的 action recoverability
压缩后 z 的 action recoverability
```

如果 adapter reconstruction loss 很低，但 action probe 掉很多，那说明这个 tokenizer 把控制相关几何压坏了。

**5. 等 baseline 稳了，再做 PV-VAE 和 DeltaTok**
第二阶段再比较：

```text
PV-VAE-style temporal tokenizer
Delta transition tokenizer
```

PV-VAE 要重点防：

```text
static copy shortcut
```

所以必须报：

```text
future_mse
static_future_mse / copy_mse
delta_ratio
```

DeltaTok 更像后续主线，因为它压的是变化：

```text
x_t, x_{t+k} -> z_delta
x_t, z_delta -> x_hat_future
```

这更适合机器人场景里的“静态背景 + 局部交互变化”。

**我建议你现在的实际任务清单**
按优先级排：

1. 写一个 `feature tokenizer` 的数据 manifest 规范，列出 DROID / Behavior / OXE / Ego4D 每个数据源有什么字段、fps、camera、action、instruction。

2. 做统一 dataloader，先只保证能 sample 出统一 clip batch。

3. 确认当前 `external/openpi` 里 tokenizer 相关实现是否完整。
   我刚才看到本地若干被文档引用的文件可能是空文件或镜像不完整，例如 `predictive_feature_vae.py`、`latent_action.py`、`delta_tokenizer.py`。落地前要和 HPC 上真实代码对齐一下。

4. 实现/整理 standalone `semantic_feature_vae.py`，先做 per-frame S-VAE / channel adapter。

5. 用一个小混合数据子集跑 smoke：
   ```text
   1k / 5k steps
   batch shape 正确
   loss 稳定下降
   feature PCA 正常
   ```

6. 加 LARY-style action probe，先在有 action 的数据上验证压缩 latent 是否保留 action 信息。

7. 再决定是上 PV-VAE 还是 DeltaTok。
   如果 per-frame adapter 保留 action 信息很好，但压缩率不够，再推 PV-VAE；如果 PV-VAE copy shortcut 明显，就转 DeltaTok。

**一句话结论**
根据项目文档和师兄聊天，下一步不是直接做新的 WAM 方法，也不是马上开大规模 LIBERO 评测，而是先把 **feature tokenizer 的数据、adapter baseline 和 action-aware 评测** 搭起来。这个东西做好后，才有资格接入 FastWAM，去验证“高压缩 semantic latent 是否比原始高维 feature / Wan VAE latent 更适合 robot world modeling”。

> 1. LARY-style action probe是什么？2.数据集要做成什么样的格式？ 3.tokenizer 的输入输出 contract 是什么？tokenizer_methods.md中难道没有规定嘛？Per-frame S-VAE:
> [B,V,T,N,D] -> [B,V,T,N,d]
>
> PV-VAE:
> [B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
>
> DeltaTok:
> x_t, x_{t+k} -> z_delta [B,V,M,d]具体参数指什么？4.tokenizer_methods.md请你再仔细读一下，里面有很多实验进度，当前代码实现了什么，以及建议入口，当前实验经验等等。请你给我讲一下现在具体的研究进度做到哪一步了。

我来把这几个问题拆开讲清楚，并且这次严格以 `tokenizer_methods.md` 和项目文档里的“当前代码状态/实验经验”为准，不把建议和已完成混在一起。

> 1. LARY-style action probe是什么？2.数据集要做成什么样的格式？ 3.tokenizer 的输入输出 contract 是什么？tokenizer_methods.md中难道没有规定嘛？Per-frame S-VAE:
> [B,V,T,N,D] -> [B,V,T,N,d]
>
> PV-VAE:
> [B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
>
> DeltaTok:
> x_t, x_{t+k} -> z_delta [B,V,M,d]具体参数指什么？4.tokenizer_methods.md请你再仔细读一下，里面有很多实验进度，当前代码实现了什么，以及建议入口，当前实验经验等等。请你给我讲一下现在具体的研究进度做到哪一步了。然后再根据之前的聊天记录以及你的分析，来为我重新设计该怎么展开

<details><summary>5 previous messages</summary>

> 我先把 `tokenizer_methods.md` 和当前本地代码状态再对一遍，特别是“文档说已实现”和“本地文件实际是否有内容”这两层要分清楚；这样后面给你的展开路线才不会飘。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-ChildItem external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py, external\openpi\src\openpi\models_pytorch\latent_action.py, external\openpi\src\openpi\models_pytorch\delta_tokenizer.py -ErrorAction SilentlyContinue | Select-Object FullName,Length`
> - Ran `Get-Content wam\tokenizer_methods.md`
> - Ran `rg -n "PredictiveFeatureVAE|FeatureLatentAction|FeatureDeltaTokenizer|train_predictive_feature_vae|train_lam_libero|train_deltatok" external\openpi wam runs -S`
> - Ran `Get-ChildItem runs\pvvae,runs\lam,runs\deltatok -Recurse -File | Select-Object FullName,Length | Select-Object -First 120`
>
> </details>
>
> 有个关键校准点：`tokenizer_methods.md` 记录的“已实现”很可能指 HPC/某个工作副本里的状态；当前本地镜像里 `predictive_feature_vae.py / latent_action.py / delta_tokenizer.py` 三个模型文件是 0 字节，但脚本、sbatch 和 synced run 日志还在，说明实验确实跑过，只是本地代码镜像不完整或被占位文件覆盖了。这个会影响下一步：先要恢复/同步真实模型实现。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content runs\deltatok\deltatok_libero_svg_p_stride4_bs64_3000_20260522_171438\args.json`
> - Ran `Get-Content runs\deltatok\deltatok_libero_svg_p_stride4_bs64_3000_20260522_171438\deltatok_s4.310239.err -Tail 80`
> - Ran `Get-Content external\openpi\scripts\train_deltatok_libero.py | Select-Object -Skip 130 -First 130`
> - Ran `Get-Content external\openpi\scripts\train_predictive_feature_vae_libero.py | Select-Object -Skip 300 -First 150`
>
> </details>
>
> 你说得对，`tokenizer_methods.md` 里其实已经规定了 tokenizer 的核心 contract。我前面说“先定 contract”不够准确，更准确应该是：**文档已经给了研究级 contract，下一步要把它落成代码级 batch schema、shape 约定、数据字段和 eval protocol。**
>
> **1. LARY-style Action Probe 是什么**
> 它不是一个 tokenizer，也不是 WAM 主模型，而是一个 **评估探针**：检查某种 feature / tokenizer latent 里面还保留了多少“动作相关信息”。
>
> 形式类似 inverse dynamics model：
>
> ```text
> z_t, z_{t+k}, instruction(optional) -> action chunk a_t:t+k
> ```
>
> 比如你有原始 SVG/DINO feature：
>
> ```text
> x_t, x_{t+4} -> probe -> action[0:4]
> ```
>
> 也可以对压缩后的 token 做：
>
> ```text
> z_t, z_{t+4} -> probe -> action[0:4]
> ```
>
> 它回答的问题是：
>
> ```text
> 压缩前 feature 能恢复动作吗？
> 压缩后 latent 还能恢复动作吗？
> tokenizer 有没有把控制相关信息压没？
> ```
>
> 所以它是 tokenizer 质量评估。不是只看 `feature_mse`，还看 `action recoverability`。这来自 LARY 和 `Reconstruction or Semantics?` 的思想。
>
> **2. 数据集要做成什么格式**
> 不要一开始就把所有数据物理转成同一个巨大格式，可以先做“统一 dataloader 输出协议”。
>
> 建议每个 sample 输出：
>
> ```python
> {
>     "images": Tensor[T, V, C, H, W],
>     "actions": Tensor[T-1, A] or None,
>     "instruction": str or None,
>     "dataset_name": str,
>     "episode_id": str,
>     "fps": float,
>     "timestamps": Tensor[T],
>     "camera_names": list[str],
> }
> ```
>
> 字段含义：
>
> ```text
> T: 采样的 clip 帧数
> V: 相机数 / view 数
> C,H,W: RGB 图像通道和尺寸
> A: action 维度
> actions=None: 对 Ego4D 这类无机器人动作数据允许为空
> ```
>
> 然后 tokenizer 训练时再做：
>
> ```text
> images -> frozen encoder -> feature x
> ```
>
> 也就是说，数据层统一的是 RGB clip + metadata；feature 可以在线抽，也可以以后缓存。
>
> 对不同数据集的角色：
>
> ```text
> DROID / OXE / Behavior / LIBERO:
>   有 action，能训练 tokenizer，也能做 action probe
>
> Ego4D:
>   大量自然视频，无 robot action，适合训练 reconstruction/predictive tokenizer，
>   但不能做 robot action probe
> ```
>
> **3. Tokenizer 输入输出 contract**
> `tokenizer_methods.md` 已经给了。我展开一下每个符号。
>
> 通用 teacher feature：
>
> ```text
> x: [B, V, T, N, D]
> ```
>
> 含义：
>
> ```text
> B: batch size
> V: camera/view 数
> T: 时间帧数
> N: 每帧 spatial token 数，例如 16x16=256
> D: frozen teacher feature 维度，例如 DINO/SVG 是 384，V-JEPA/Web-DINO 可到 1024
> d: tokenizer 压缩后的 channel 维度，例如 64/96/128
> M: delta token 数，例如 1/4/8
> ```
>
> Per-frame S-VAE：
>
> ```text
> [B,V,T,N,D] -> [B,V,T,N,d]
> ```
>
> 每帧独立压缩，只压通道，不压时间。适合做第一版 adapter baseline。
>
> PV-VAE：
>
> ```text
> [B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
> ```
>
> 这里 `1+T` 表示：
>
> ```text
> 第 0 帧 current/condition frame
> +
> T 个 future frames
> ```
>
> 如果有 16 个 future frames，`temporal_compression=4`：
>
> ```text
> 17 frames -> 5 latent groups
> group 0: x0
> group 1: x1-x4
> group 2: x5-x8
> group 3: x9-x12
> group 4: x13-x16
> ```
>
> DeltaTok：
>
> ```text
> x_t, x_{t+k} -> z_delta [B,V,M,d]
> ```
>
> 这里：
>
> ```text
> x_t: 当前 feature
> x_{t+k}: k 步后的 future feature
> z_delta: 从当前到未来的变化 token
> M: 用几个 token 表达变化
> d: 每个 delta token 的维度
> ```
>
> decode 时：
>
> ```text
> x_t, z_delta -> x_hat_{t+k}
> ```
>
> 核心区别是：DeltaTok 不让 token 记住整帧背景，只记变化。
>
> **4. 当前研究进度做到哪了**
> 按 `tokenizer_methods.md` 和本地状态综合看，进度是这样：
>
> Per-frame S-VAE / Channel Adapter：
>
> ```text
> 状态：还没实现 standalone S-VAE adapter。
> 已有基础：pi0.5/OpenPI 已有 future image teacher encode，
> 支持 raw SigLIP / raw DINOv3 target，以及 DINO32/128 随机投影。
> 建议入口：semantic_feature_vae.py + train_svae_libero.py。
> ```
>
> PV-VAE-style：
>
> ```text
> 状态：文档记录为已实现并在跑 LIBERO/SVG-P。
> 脚本存在：external/openpi/scripts/train_predictive_feature_vae_libero.py
> 支持：SVG-P/DINOv3 teacher、future_deltas、temporal_compression=4、
> PCA/SVG decoded visualization。
> 指标：future_mse, static_future_mse, delta_ratio 等。
> 经验：static_future_mse 很低，说明静态 copy shortcut 很强。
> ```
>
> 但是有一个重要风险：当前本地镜像里：
>
> ```text
> external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py
> ```
>
> 是 0 字节。说明真实模型代码可能在 HPC 或另一个工作副本里，本地同步不完整。
>
> LAM / DeltaTok-style：
>
> ```text
> 状态：文档说 LAM 原型存在，不是正式 DeltaTok。
> 脚本存在：train_lam_libero.py
> 用途：current_features + future_features -> latent action -> reconstruct future
> ```
>
> 另外，本地还看到一个后续 DeltaTok-style smoke：
>
> ```text
> runs/deltatok/deltatok_libero_svg_p_stride4_bs64_3000_20260522_171438
> ```
>
> 这个 run 用的是：
>
> ```text
> teacher=svg_p
> delta_stride=4
> batch_size=64
> max_steps=3000
> feature_dim=384
> model_dim=384
> encoder_layers=8
> decoder_layers=8
> ```
>
> 最后日志大概是：
>
> ```text
> step=3000 mse=0.007110
> copy_mse=0.008126
> delta_ratio=0.870
> ```
>
> 说明它比 copy baseline 略好，但还只是一个初步 prototype，不足以证明 DeltaTok 已经成熟。
>
> 同样，本地：
>
> ```text
> delta_tokenizer.py
> latent_action.py
> ```
>
> 也是 0 字节，所以要先恢复真实代码。
>
> **重新设计下一步**
> 我建议现在按这个顺序展开：
>
> 1. **先做代码状态核验**
>    确认 HPC/远端真实版本里这三个文件是否完整：
>
>    ```text
>    predictive_feature_vae.py
>    latent_action.py
>    delta_tokenizer.py
>    ```
>
>    这是第一步。不然你会在一个“文档说有、当前本地没有”的状态里继续推进。
>
> 2. **整理 tokenizer 数据协议**
>    先做统一 dataloader 输出：
>
>    ```text
>    images, actions, instruction, fps, timestamps, dataset_name
>    ```
>
>    不要急着全量转换，先每个数据集抽 100 clips 做 shape check。
>
> 3. **补 Per-frame S-VAE baseline**
>    这是师兄说的“训练一个 adapter”最直接对应物：
>
>    ```text
>    SVG/DINO feature [N,D] -> [N,d] -> [N,D]
>    ```
>
>    它最稳，也能作为所有后续 PV-VAE/DeltaTok 的压缩 baseline。
>
> 4. **复现实验表，而不是只跑 loss**
>    对 S-VAE / PV-VAE / DeltaTok 都统一报：
>
>    ```text
>    feature_mse
>    feature_cosine
>    copy_mse / static_future_mse
>    delta_ratio
>    PCA / SVG decode visualization
>    action probe error
>    ```
>
> 5. **再决定主线**
>    如果目标是“高压缩 VAE/adapter”，先推 S-VAE 和 PV-VAE。
>    如果目标是“方法创新和机器人交互变化”，再推 DeltaTok。
>    这两条不是冲突，而是 baseline 和主线的关系。
>
> 最实在的一句话：**现在不是重新发明方向，而是把已有 tokenizer 线的代码和数据地基补完整：先恢复真实实现，再统一多数据集 dataloader，然后用 S-VAE 做稳定 adapter baseline，最后用 PV-VAE/DeltaTok 比较高压缩和 transition-level 压缩到底哪个更保留 action-relevant 信息。**
</details>

你说得对，`tokenizer_methods.md` 里其实已经规定了 tokenizer 的核心 contract。我前面说“先定 contract”不够准确，更准确应该是：**文档已经给了研究级 contract，下一步要把它落成代码级 batch schema、shape 约定、数据字段和 eval protocol。**

**1. LARY-style Action Probe 是什么**
它不是一个 tokenizer，也不是 WAM 主模型，而是一个 **评估探针**：检查某种 feature / tokenizer latent 里面还保留了多少“动作相关信息”。

形式类似 inverse dynamics model：

```text
z_t, z_{t+k}, instruction(optional) -> action chunk a_t:t+k
```

比如你有原始 SVG/DINO feature：

```text
x_t, x_{t+4} -> probe -> action[0:4]
```

也可以对压缩后的 token 做：

```text
z_t, z_{t+4} -> probe -> action[0:4]
```

它回答的问题是：

```text
压缩前 feature 能恢复动作吗？
压缩后 latent 还能恢复动作吗？
tokenizer 有没有把控制相关信息压没？
```

所以它是 tokenizer 质量评估。不是只看 `feature_mse`，还看 `action recoverability`。这来自 LARY 和 `Reconstruction or Semantics?` 的思想。

**2. 数据集要做成什么格式**
不要一开始就把所有数据物理转成同一个巨大格式，可以先做“统一 dataloader 输出协议”。

建议每个 sample 输出：

```python
{
    "images": Tensor[T, V, C, H, W],
    "actions": Tensor[T-1, A] or None,
    "instruction": str or None,
    "dataset_name": str,
    "episode_id": str,
    "fps": float,
    "timestamps": Tensor[T],
    "camera_names": list[str],
}
```

字段含义：

```text
T: 采样的 clip 帧数
V: 相机数 / view 数
C,H,W: RGB 图像通道和尺寸
A: action 维度
actions=None: 对 Ego4D 这类无机器人动作数据允许为空
```

然后 tokenizer 训练时再做：

```text
images -> frozen encoder -> feature x
```

也就是说，数据层统一的是 RGB clip + metadata；feature 可以在线抽，也可以以后缓存。

对不同数据集的角色：

```text
DROID / OXE / Behavior / LIBERO:
  有 action，能训练 tokenizer，也能做 action probe

Ego4D:
  大量自然视频，无 robot action，适合训练 reconstruction/predictive tokenizer，
  但不能做 robot action probe
```

**3. Tokenizer 输入输出 contract**
`tokenizer_methods.md` 已经给了。我展开一下每个符号。

通用 teacher feature：

```text
x: [B, V, T, N, D]
```

含义：

```text
B: batch size
V: camera/view 数
T: 时间帧数
N: 每帧 spatial token 数，例如 16x16=256
D: frozen teacher feature 维度，例如 DINO/SVG 是 384，V-JEPA/Web-DINO 可到 1024
d: tokenizer 压缩后的 channel 维度，例如 64/96/128
M: delta token 数，例如 1/4/8
```

Per-frame S-VAE：

```text
[B,V,T,N,D] -> [B,V,T,N,d]
```

每帧独立压缩，只压通道，不压时间。适合做第一版 adapter baseline。

PV-VAE：

```text
[B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
```

这里 `1+T` 表示：

```text
第 0 帧 current/condition frame
+
T 个 future frames
```

如果有 16 个 future frames，`temporal_compression=4`：

```text
17 frames -> 5 latent groups
group 0: x0
group 1: x1-x4
group 2: x5-x8
group 3: x9-x12
group 4: x13-x16
```

DeltaTok：

```text
x_t, x_{t+k} -> z_delta [B,V,M,d]
```

这里：

```text
x_t: 当前 feature
x_{t+k}: k 步后的 future feature
z_delta: 从当前到未来的变化 token
M: 用几个 token 表达变化
d: 每个 delta token 的维度
```

decode 时：

```text
x_t, z_delta -> x_hat_{t+k}
```

核心区别是：DeltaTok 不让 token 记住整帧背景，只记变化。

**4. 当前研究进度做到哪了**
按 `tokenizer_methods.md` 和本地状态综合看，进度是这样：

Per-frame S-VAE / Channel Adapter：

```text
状态：还没实现 standalone S-VAE adapter。
已有基础：pi0.5/OpenPI 已有 future image teacher encode，
支持 raw SigLIP / raw DINOv3 target，以及 DINO32/128 随机投影。
建议入口：semantic_feature_vae.py + train_svae_libero.py。
```

PV-VAE-style：

```text
状态：文档记录为已实现并在跑 LIBERO/SVG-P。
脚本存在：external/openpi/scripts/train_predictive_feature_vae_libero.py
支持：SVG-P/DINOv3 teacher、future_deltas、temporal_compression=4、
PCA/SVG decoded visualization。
指标：future_mse, static_future_mse, delta_ratio 等。
经验：static_future_mse 很低，说明静态 copy shortcut 很强。
```

但是有一个重要风险：当前本地镜像里：

```text
external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py
```

是 0 字节。说明真实模型代码可能在 HPC 或另一个工作副本里，本地同步不完整。

LAM / DeltaTok-style：

```text
状态：文档说 LAM 原型存在，不是正式 DeltaTok。
脚本存在：train_lam_libero.py
用途：current_features + future_features -> latent action -> reconstruct future
```

另外，本地还看到一个后续 DeltaTok-style smoke：

```text
runs/deltatok/deltatok_libero_svg_p_stride4_bs64_3000_20260522_171438
```

这个 run 用的是：

```text
teacher=svg_p
delta_stride=4
batch_size=64
max_steps=3000
feature_dim=384
model_dim=384
encoder_layers=8
decoder_layers=8
```

最后日志大概是：

```text
step=3000 mse=0.007110
copy_mse=0.008126
delta_ratio=0.870
```

说明它比 copy baseline 略好，但还只是一个初步 prototype，不足以证明 DeltaTok 已经成熟。

同样，本地：

```text
delta_tokenizer.py
latent_action.py
```

也是 0 字节，所以要先恢复真实代码。

**重新设计下一步**
我建议现在按这个顺序展开：

1. **先做代码状态核验**
   确认 HPC/远端真实版本里这三个文件是否完整：

   ```text
   predictive_feature_vae.py
   latent_action.py
   delta_tokenizer.py
   ```

   这是第一步。不然你会在一个“文档说有、当前本地没有”的状态里继续推进。

2. **整理 tokenizer 数据协议**
   先做统一 dataloader 输出：

   ```text
   images, actions, instruction, fps, timestamps, dataset_name
   ```

   不要急着全量转换，先每个数据集抽 100 clips 做 shape check。

3. **补 Per-frame S-VAE baseline**
   这是师兄说的“训练一个 adapter”最直接对应物：

   ```text
   SVG/DINO feature [N,D] -> [N,d] -> [N,D]
   ```

   它最稳，也能作为所有后续 PV-VAE/DeltaTok 的压缩 baseline。

4. **复现实验表，而不是只跑 loss**
   对 S-VAE / PV-VAE / DeltaTok 都统一报：

   ```text
   feature_mse
   feature_cosine
   copy_mse / static_future_mse
   delta_ratio
   PCA / SVG decode visualization
   action probe error
   ```

5. **再决定主线**
   如果目标是“高压缩 VAE/adapter”，先推 S-VAE 和 PV-VAE。
   如果目标是“方法创新和机器人交互变化”，再推 DeltaTok。
   这两条不是冲突，而是 baseline 和主线的关系。

最实在的一句话：**现在不是重新发明方向，而是把已有 tokenizer 线的代码和数据地基补完整：先恢复真实实现，再统一多数据集 dataloader，然后用 S-VAE 做稳定 adapter baseline，最后用 PV-VAE/DeltaTok 比较高压缩和 transition-level 压缩到底哪个更保留 action-relevant 信息。**

> 好的，服务器上的项目我已经建了一个GitHub repo，之后我们本地改代码，把更新push到github上，用这种方式来协作。[YuhengNa/awesome-wam](https://github.com/YuhengNa/awesome-wam)

<details><summary>5 previous messages</summary>

> 好，这样协作会清晰很多：本地负责改代码和整理文档，GitHub 做同步源，服务器/HPC 从 GitHub 拉取或 checkout 指定 commit 来跑实验。我先看一下当前本地仓库的 git remote 和状态，确认是否已经接上这个 repo。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `git status -sb`
> - Ran `git remote -v`
> - Ran `git branch --show-current`
>
> </details>
>
> 当前这个 `E:\awesome_wam` 在 shell 里没有被识别成 git 仓库，虽然之前目录列表里看到过 `.git`。我再确认一下 `.git` 是否是真目录、文件，或者是不是同步时丢了元数据。
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `Get-ChildItem -Force .git | Select-Object -First 20`
> - Ran `Get-Item -Force .git | Format-List FullName,Mode,Length,Attributes`
> - Ran `Get-ChildItem -Force | Select-Object Name,Mode,Length`
> - Ran `Get-ChildItem -Force .git -Recurse | Select-Object -First 30 FullName,Mode,Length`
> - Ran `Get-ChildItem -Force .git | Measure-Object`
>
> </details>
>
> 好，这个协作方式很合适。
>
> 我刚看了一下当前本地 `E:\awesome_wam`：里面有 `.git` 文件夹，但它是空的，所以现在还不是一个有效 git 仓库。也就是说后面需要做一次初始化/重新 clone/绑定 remote。
>
> 建议工作流：
>
> ```bash
> cd E:\awesome_wam
> git init
> git remote add origin https://github.com/YuhengNa/awesome-wam.git
> git status
> ```
>
> 然后重点注意：**不要把大文件、实验结果、数据、checkpoint push 上去**。这个项目里尤其要排除：
>
> ```text
> runs/
> external/*/.git/
> checkpoints/
> *.pt
> *.pth
> *.ckpt
> *.mp4
> *.zip
> __pycache__/
> ```
>
> 我们后续可以这样协作：
>
> ```text
> 本地改代码/文档
>   -> git diff 检查
>   -> commit
>   -> push GitHub
>   -> 服务器 pull GitHub
>   -> 跑实验
>   -> 只同步关键日志/summary/md，不同步大 checkpoint
> ```
>
> 下一步我建议先做两件小事：
>
> 1. 建一个靠谱的 `.gitignore`，保护数据和实验产物。
> 2. 把当前 `wam/`、`scripts/`、必要的 `external/openpi` / `external/FastWAM` 改动纳入版本管理，但不要误提交大 repo 子目录里的历史和缓存。
>
> 之后你让我改代码时，我就可以按“改动 -> 测试 -> git diff 摘要 -> 你确认后 push/commit”的方式来做。
</details>

好，这个协作方式很合适。

我刚看了一下当前本地 `E:\awesome_wam`：里面有 `.git` 文件夹，但它是空的，所以现在还不是一个有效 git 仓库。也就是说后面需要做一次初始化/重新 clone/绑定 remote。

建议工作流：

```bash
cd E:\awesome_wam
git init
git remote add origin https://github.com/YuhengNa/awesome-wam.git
git status
```

然后重点注意：**不要把大文件、实验结果、数据、checkpoint push 上去**。这个项目里尤其要排除：

```text
runs/
external/*/.git/
checkpoints/
*.pt
*.pth
*.ckpt
*.mp4
*.zip
__pycache__/
```

我们后续可以这样协作：

```text
本地改代码/文档
  -> git diff 检查
  -> commit
  -> push GitHub
  -> 服务器 pull GitHub
  -> 跑实验
  -> 只同步关键日志/summary/md，不同步大 checkpoint
```

下一步我建议先做两件小事：

1. 建一个靠谱的 `.gitignore`，保护数据和实验产物。
2. 把当前 `wam/`、`scripts/`、必要的 `external/openpi` / `external/FastWAM` 改动纳入版本管理，但不要误提交大 repo 子目录里的历史和缓存。

之后你让我改代码时，我就可以按“改动 -> 测试 -> git diff 摘要 -> 你确认后 push/commit”的方式来做。

> emm，有一个点需要补充，其实本地你现在看到的这个E:\awesome_wam，是我之前手动从服务器下载下来的，可能不太全。现在你可以不用管E:\awesome_wam的内容，可以clone下来[YuhengNa/awesome-wam](https://github.com/YuhengNa/awesome-wam)的仓库

<details><summary>5 previous messages</summary>

> 明白，那我们把现在这个 `E:\awesome_wam` 当作旧的本地快照，不再以它为准。我会把 GitHub 仓库 clone 到一个新的子目录里，避免覆盖你现有文件。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `git clone https://github.com/YuhengNa/awesome-wam.git awesome-wam-github`
>
> </details>
>
> 仓库已经 clone 下来了。我检查一下新仓库的目录、分支和状态，确认后续就在这个新目录里工作。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-ChildItem -Force`
> - Ran `git remote -v`
> - Ran `git status -sb`
> - Ran `rg --files`
>
> </details>
>
> 已经 clone 好了，位置是：
>
> [E:\awesome_wam\awesome-wam-github](/E:/awesome_wam/awesome-wam-github)
>
> 这个新目录是干净的 GitHub 仓库：
>
> ```text
> branch: main
> remote: https://github.com/YuhengNa/awesome-wam.git
> status: clean
> ```
>
> 后续我会以这个目录为准，不再把旧的 `E:\awesome_wam` 快照当作权威代码。旧目录可以继续留着当参考/备份。
</details>

已经 clone 好了，位置是：

[E:\awesome_wam\awesome-wam-github](/E:/awesome_wam/awesome-wam-github)

这个新目录是干净的 GitHub 仓库：

```text
branch: main
remote: https://github.com/YuhengNa/awesome-wam.git
status: clean
```

后续我会以这个目录为准，不再把旧的 `E:\awesome_wam` 快照当作权威代码。旧目录可以继续留着当参考/备份。

> 好的，现在可以进入到E:\awesome_wam\awesome-wam-github，继续咱们刚刚所探讨的研究计划

<details><summary>1 previous message</summary>

> 好，现在以后我们以这个仓库为准：
>
> [E:\awesome_wam\awesome-wam-github](/E:/awesome_wam/awesome-wam-github)
>
> 我对当前研究计划的校准如下。
>
> 当前最应该推进的不是再开一个新想法，而是把 **feature tokenizer 这条线从“想法/小实验”整理成可复现工程线**。师兄聊天里说的重点很明确：
>
> ```text
> 训 tokenizer
> 准备更大规模数据
> 统一数据格式
> 训练高压缩 VAE / adapter
> 参考 PV-VAE / Semantic-WM / DeltaTok
> 用 LARY 思路验证 tokenizer 质量
> ```
>
> **下一步主线**
> 建议分成三个阶段。
>
> **阶段 0：代码与资产对齐**
> 现在 GitHub repo 里主要是文档、论文和脚本，没有 `external/openpi` / `external/FastWAM` 实际代码。旧快照里虽然有这些目录，但一些关键模型文件是空的，所以第一步应该在服务器确认真实代码在哪里。
>
> 要确认：
>
> ```text
> predictive_feature_vae.py
> latent_action.py
> delta_tokenizer.py
> train_predictive_feature_vae_libero.py
> train_lam_libero.py
> train_deltatok_libero.py
> ```
>
> 目标是把“已跑过的小实验”恢复成 GitHub 上可追踪的代码状态。否则文档说 PV-VAE/DeltaTok 已实现，但本地无法复现。
>
> **阶段 1：统一 feature tokenizer 数据协议**
> 先别急着训模型。先定义所有数据集输出同一种 clip batch：
>
> ```python
> {
>     "images": [T, V, C, H, W],
>     "actions": [T-1, A] or None,
>     "instruction": str or None,
>     "dataset_name": str,
>     "episode_id": str,
>     "fps": float,
>     "timestamps": [T],
>     "camera_names": list[str],
> }
> ```
>
> 数据集角色：
>
> ```text
> DROID / OXE / Behavior / LIBERO:
>   有 robot action，可训练 tokenizer，也可做 action probe
>
> Ego4D:
>   视频量大，但无机器人 action，适合训练视觉 tokenizer，不适合 action probe
> ```
>
> 这一阶段产物应该是一个清楚的 data contract 文档，以及每个数据集 100 个 clip 的 shape/sample 检查。
>
> **阶段 2：先做最稳的 Per-frame S-VAE / Adapter**
> 这对应师兄说的“训练一个 adapter”。
>
> 输入输出：
>
> ```text
> x: [B,V,T,N,D]
> z: [B,V,T,N,d]
> x_hat: [B,V,T,N,D]
> ```
>
> 先只压通道，不压时间：
>
> ```text
> D -> d，例如 384 -> 96 或 1024 -> 96
> ```
>
> loss：
>
> ```text
> feature_mse
> + cosine_loss
> + beta * KL
> ```
>
> 这个 baseline 的意义是验证：
>
> ```text
> 大数据能否稳定读
> teacher feature 能否稳定抽
> 高维 feature 能否压缩
> 压缩后是否保留 action-relevant 信息
> ```
>
> **阶段 3：再比较 PV-VAE 和 DeltaTok**
> 等 S-VAE 稳了，再做更有研究性的 tokenizer：
>
> PV-VAE：
>
> ```text
> [B,V,17,N,D] -> [B,V,5,N,d]
> ```
>
> 重点防止 copy shortcut：
>
> ```text
> future_mse
> static_future_mse
> delta_ratio
> ```
>
> DeltaTok：
>
> ```text
> x_t, x_{t+k} -> z_delta [B,V,M,d]
> x_t, z_delta -> x_hat_{t+k}
> ```
>
> 它更像方法主线，因为它专门压缩“变化”，更适合机器人视频里的局部交互。
>
> **评测必须补上**
> 每个 tokenizer 不只看 reconstruction loss。至少要有：
>
> ```text
> feature_mse / cosine
> copy_mse 或 static_future_mse
> delta_ratio
> PCA / SVG decoded visualization
> LARY-style action probe
> ```
>
> 其中 LARY-style action probe 就是：
>
> ```text
> z_t, z_{t+k} -> action chunk
> ```
>
> 它检查压缩 latent 是否还保留动作信息。
>
> **我建议现在先做的具体文件**
> 可以在 GitHub repo 里新增一份研究执行文档，比如：
>
> ```text
> wam/feature_tokenizer_execution_plan.md
> ```
>
> 里面固定：
>
> ```text
> 1. 数据 contract
> 2. tokenizer 输入输出 shape
> 3. 当前代码缺口
> 4. S-VAE baseline 训练计划
> 5. PV-VAE / DeltaTok 后续对照
> 6. action probe 评测方式
> ```
>
> 这样你和师兄对齐时，不会只停留在口头讨论。下一步最好先把这个执行计划写清楚，然后再开始动代码。
</details>

好，现在以后我们以这个仓库为准：

[E:\awesome_wam\awesome-wam-github](/E:/awesome_wam/awesome-wam-github)

我对当前研究计划的校准如下。

当前最应该推进的不是再开一个新想法，而是把 **feature tokenizer 这条线从“想法/小实验”整理成可复现工程线**。师兄聊天里说的重点很明确：

```text
训 tokenizer
准备更大规模数据
统一数据格式
训练高压缩 VAE / adapter
参考 PV-VAE / Semantic-WM / DeltaTok
用 LARY 思路验证 tokenizer 质量
```

**下一步主线**
建议分成三个阶段。

**阶段 0：代码与资产对齐**
现在 GitHub repo 里主要是文档、论文和脚本，没有 `external/openpi` / `external/FastWAM` 实际代码。旧快照里虽然有这些目录，但一些关键模型文件是空的，所以第一步应该在服务器确认真实代码在哪里。

要确认：

```text
predictive_feature_vae.py
latent_action.py
delta_tokenizer.py
train_predictive_feature_vae_libero.py
train_lam_libero.py
train_deltatok_libero.py
```

目标是把“已跑过的小实验”恢复成 GitHub 上可追踪的代码状态。否则文档说 PV-VAE/DeltaTok 已实现，但本地无法复现。

**阶段 1：统一 feature tokenizer 数据协议**
先别急着训模型。先定义所有数据集输出同一种 clip batch：

```python
{
    "images": [T, V, C, H, W],
    "actions": [T-1, A] or None,
    "instruction": str or None,
    "dataset_name": str,
    "episode_id": str,
    "fps": float,
    "timestamps": [T],
    "camera_names": list[str],
}
```

数据集角色：

```text
DROID / OXE / Behavior / LIBERO:
  有 robot action，可训练 tokenizer，也可做 action probe

Ego4D:
  视频量大，但无机器人 action，适合训练视觉 tokenizer，不适合 action probe
```

这一阶段产物应该是一个清楚的 data contract 文档，以及每个数据集 100 个 clip 的 shape/sample 检查。

**阶段 2：先做最稳的 Per-frame S-VAE / Adapter**
这对应师兄说的“训练一个 adapter”。

输入输出：

```text
x: [B,V,T,N,D]
z: [B,V,T,N,d]
x_hat: [B,V,T,N,D]
```

先只压通道，不压时间：

```text
D -> d，例如 384 -> 96 或 1024 -> 96
```

loss：

```text
feature_mse
+ cosine_loss
+ beta * KL
```

这个 baseline 的意义是验证：

```text
大数据能否稳定读
teacher feature 能否稳定抽
高维 feature 能否压缩
压缩后是否保留 action-relevant 信息
```

**阶段 3：再比较 PV-VAE 和 DeltaTok**
等 S-VAE 稳了，再做更有研究性的 tokenizer：

PV-VAE：

```text
[B,V,17,N,D] -> [B,V,5,N,d]
```

重点防止 copy shortcut：

```text
future_mse
static_future_mse
delta_ratio
```

DeltaTok：

```text
x_t, x_{t+k} -> z_delta [B,V,M,d]
x_t, z_delta -> x_hat_{t+k}
```

它更像方法主线，因为它专门压缩“变化”，更适合机器人视频里的局部交互。

**评测必须补上**
每个 tokenizer 不只看 reconstruction loss。至少要有：

```text
feature_mse / cosine
copy_mse 或 static_future_mse
delta_ratio
PCA / SVG decoded visualization
LARY-style action probe
```

其中 LARY-style action probe 就是：

```text
z_t, z_{t+k} -> action chunk
```

它检查压缩 latent 是否还保留动作信息。

**我建议现在先做的具体文件**
可以在 GitHub repo 里新增一份研究执行文档，比如：

```text
wam/feature_tokenizer_execution_plan.md
```

里面固定：

```text
1. 数据 contract
2. tokenizer 输入输出 shape
3. 当前代码缺口
4. S-VAE baseline 训练计划
5. PV-VAE / DeltaTok 后续对照
6. action probe 评测方式
```

这样你和师兄对齐时，不会只停留在口头讨论。下一步最好先把这个执行计划写清楚，然后再开始动代码。

对，这个提醒很关键。`external/` 不能随手整个 `git add`。

原因是：

```text
external/FastWAM
```

如果是软链接，Git 只会记录“这是一个链接”，不会把真实代码上传。

而：

```text
external/DeltaTok
external/DreamDojo
external/openpi
external/FastWAM
```

如果它们本身是 Git 仓库，直接 `git add external` 很可能变成 submodule / gitlink，而不是把代码内容纳入当前 repo。

所以后续我们要分清三类东西。

**建议策略**

1. **文档、脚本、计划**
   放在当前 `awesome-wam` 主仓库里正常提交：

   ```text
   wam/
   docs/
   scripts/
   AGENTS.md
   ```

2. **第三方完整仓库**
   不直接塞进主仓库。保留为外部依赖，在文档里记录：

   ```text
   repo URL
   commit hash
   expected path
   setup command
   ```

   比如写到：

   ```text
   docs/external_repos.md
   ```

3. **我们对外部仓库的修改**
   不能粗暴提交整个 external。更稳的方式是：

   ```text
   patches/
   或
   forks/submodules 明确管理
   或
   在 README 里记录需要 clone 的 fork 地址
   ```

如果师兄希望把 `external/openpi` 或某个修改版 FastWAM 也一起版本化，那最好单独建 fork，例如：

```text
YuhengNa/openpi-awam
YuhengNa/FastWAM-awam
```

然后当前 `awesome-wam` 只记录依赖关系和 patch/配置。

**我们现在下一步可以做**
我建议先在新 repo 里加一个小文档：

```text
docs/external_code_policy.md
```

里面写清楚：

```text
external/ 目录用途
哪些不提交
哪些用 GitHub URL + commit 管理
如何同步服务器代码
如何保存我们自己的 patch
```

这样之后不会误把软链接、子仓库或大文件提交进去。当前这个 repo 最适合先作为 **研究计划 + 实验脚本 + patch/配置管理仓库**，不要直接变成所有外部代码的大杂烩。

> 不想这么麻烦了，你现在还差哪些代码，你需要这些代码是要干什么

现在真正“缺”的不是为了继续理解研究方向，而是为了 **把 tokenizer 方案落到可运行/可复现实验**。

我目前缺的关键代码主要是这几类：

```text
1. external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py
2. external/openpi/src/openpi/models_pytorch/latent_action.py
3. external/openpi/src/openpi/models_pytorch/delta_tokenizer.py
```

在你旧快照里，这三个文件是空的，但文档和脚本都引用了它们。

它们分别用来做：

**1. `predictive_feature_vae.py`**
这是 PV-VAE-style tokenizer 的模型定义。

用途：

```text
输入一段 SVG/DINO feature clip
把 17 帧压成 5 个 temporal latent groups
再 decoder 回完整 feature clip
```

如果没有它，`train_predictive_feature_vae_libero.py` 这个训练脚本跑不了。

**2. `latent_action.py`**
这是 LAM / DreamDojo-style latent action feature model。

用途：

```text
current feature + future feature
  -> latent action
  -> reconstruct future feature
```

它是 DeltaTok 之前的原型，也能用于 action probe / transition representation。

如果没有它，`train_lam_libero.py` 跑不了。

**3. `delta_tokenizer.py`**
这是 DeltaTok-style tokenizer 的模型定义。

用途：

```text
x_t, x_{t+k} -> z_delta
x_t, z_delta -> x_hat_{t+k}
```

也就是当前最值得推进的 transition tokenizer 原型。

如果没有它，`train_deltatok_libero.py` 跑不了。

**我需要这些代码是为了什么**
主要有三个目的：

1. **判断当前真实进度**
   现在文档说 PV-VAE 已实现、LAM 有原型、DeltaTok 有 smoke，但本地模型文件空了。没有代码，我只能根据文档和日志推断，不能准确判断模型到底怎么写的。

2. **继续改或补实验**
   如果要做 S-VAE adapter、改 DeltaTok 的 token 数 `M`、加 action probe、换数据集 loader，都需要基于这些模型/脚本改。

3. **避免重复造轮子**
   这些代码如果服务器上已经有，我不应该重新写一遍。应该先拿真实版本，然后在上面做小改动。

**如果你嫌 external 麻烦**
最简单的方式不是上传整个 `external/`，而是只把这几个文件发进 repo 或发给我：

```text
external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py
external/openpi/src/openpi/models_pytorch/latent_action.py
external/openpi/src/openpi/models_pytorch/delta_tokenizer.py
external/openpi/scripts/train_predictive_feature_vae_libero.py
external/openpi/scripts/train_lam_libero.py
external/openpi/scripts/train_deltatok_libero.py
```

脚本其实 GitHub repo 里已经有一部分了，最缺的是前三个模型文件。

一句话：**我需要这些代码不是为了读论文，而是为了确认现有 tokenizer 实验到底实现到什么程度，并在不重写已有工作的情况下继续推进 S-VAE / PV-VAE / DeltaTok。**

> 好的，我已经成功上传到github了，我本地为什么PS E:\awesome_wam> cd .\awesome-wam-github\
> PS E:\awesome_wam\awesome-wam-github> git pull
> fatal: detected dubious ownership in repository at 'E:/awesome_wam/awesome-wam-github'
> 'E:/awesome_wam/awesome-wam-github' is owned by:
>         LAPTOP-T57PQCA7/CodexSandboxOffline (S-1-5-21-3479451055-80557564-1509539465-1010)
> but the current user is:
>         LAPTOP-T57PQCA7/Lenovo (S-1-5-21-3479451055-80557564-1509539465-1001)
> To add an exception for this directory, call:
>
>         git config --global --add safe.directory E:/awesome_wam/awesome-wam-github

这是 Git 的安全机制，不是仓库坏了。

原因是：这个仓库刚才是我这边用 `CodexSandboxOffline` 用户 clone 下来的，但你现在在 PowerShell 里是 `Lenovo` 用户执行 `git pull`。Git 发现“仓库目录所有者”和“当前执行用户”不一致，就拒绝操作，提示 `dubious ownership`。

直接按它提示执行一次就行：

```powershell
git config --global --add safe.directory E:/awesome_wam/awesome-wam-github
```

然后再：

```powershell
cd E:\awesome_wam\awesome-wam-github
git pull
```

这表示你信任这个目录，之后 `Lenovo` 用户就能正常操作这个 repo 了。

如果你想确认是否添加成功：

```powershell
git config --global --get-all safe.directory
```

> 我现在已经把你需要的代码拉到文件夹里了，请你继续咱们刚刚的研究计划

<details><summary>7 previous messages</summary>

> 好，我现在切到 GitHub 版仓库里重新检查新增的代码，重点看那几个 tokenizer 模型文件和脚本是否完整，然后我会把“真实代码进度”和下一步计划重新校准。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `git status -sb`
> - Ran `rg --files`
> - Ran `Get-ChildItem external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py, external\openpi\src\openpi\models_pytorch\latent_action.py, external\openpi\src\openpi\models_pytorch\delta_tokenizer.py -ErrorAction SilentlyContinue | Select-Object FullName,Length`
> - Ran `Get-ChildItem external -Force -ErrorAction SilentlyContinue`
>
> </details>
>
> 代码已经在 `external/openpi` 里了，关键的三个模型文件现在都有内容。`rg --files` 没列出来是因为 `external/` 很可能被 `.gitignore` 忽略，但文件本身可读。我继续读模型实现，看看它们和 `tokenizer_methods.md` 的描述是否一致。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py -TotalCount 260`
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\latent_action.py -TotalCount 240`
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\delta_tokenizer.py -TotalCount 260`
> - Ran `Get-Content .gitignore`
>
> </details>
>
> 实现看起来和文档基本一致：PV-VAE 是 `[B,V,F,N,D]` 的时空分组 VAE，LAM 是 VAE-style global latent action，Delta tokenizer 是 deterministic one-token transition autoencoder。下一步我会读完 PV-VAE 的 loss 细节和训练脚本参数，确认哪些指标已经有，哪些还缺。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content external\openpi\src\openpi\models_pytorch\predictive_feature_vae.py | Select-Object -Skip 240 -First 220`
> - Ran `Get-Content external\openpi\scripts\train_lam_libero.py -TotalCount 120`
> - Ran `Get-Content external\openpi\scripts\train_deltatok_libero.py -TotalCount 120`
> - Ran `Get-Content external\openpi\scripts\train_predictive_feature_vae_libero.py -TotalCount 120`
>
> </details>
>
> 我检查完 GitHub 版仓库里的新代码了。现在判断更准确了：**tokenizer 线不是停在想法阶段，已经有 3 个可读的原型模型和 3 条训练脚本，但还没形成一个统一、可复现实验计划。**
>
> 当前真实进度如下。
>
> **已有代码**
> 现在这三个核心文件已经存在：
>
> ```text
> external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py
> external/openpi/src/openpi/models_pytorch/latent_action.py
> external/openpi/src/openpi/models_pytorch/delta_tokenizer.py
> ```
>
> 它们分别对应：
>
> ```text
> PV-VAE-style temporal predictive tokenizer
> DreamDojo-style LAM feature transition VAE
> DeltaTok-style deterministic transition tokenizer
> ```
>
> 其中：
>
> **1. PV-VAE 已经是完整模型**
> 输入：
>
> ```text
> features: [B, V, F, N, D]
> ```
>
> 其中 `F = 1 + T_future`。
>
> 它做的是：
>
> ```text
> current frame 单独一组
> future frames 每 temporal_compression 帧压成一组
> ```
>
> 默认：
>
> ```text
> temporal_compression = 4
> 17 frames -> 5 latent groups
> ```
>
> loss 已经有：
>
> ```text
> recon_loss
> cosine_loss
> delta_loss
> kl_loss
> future_mse
> static_future_mse
> delta_ratio
> ```
>
> 所以 PV-VAE 这条不是“没做”，而是 **已经实现并做过 LIBERO/SVG-P 小实验**。
>
> **2. LAM 已经是一个 feature latent-action VAE**
> 输入：
>
> ```text
> current_features: [B,V,N,D]
> future_features:  [B,V,N,D]
> ```
>
> 它编码成一个 global latent action：
>
> ```text
> current, future -> z_action
> current, z_action -> pred_future
> ```
>
> 默认：
>
> ```text
> latent_dim = 32
> model_dim = 512
> ```
>
> 它更像 DreamDojo/LAM，不是正式 DeltaTok，但可以作为 transition representation baseline。
>
> **3. Delta tokenizer 已经有初版**
> 输入：
>
> ```text
> current_features: [B,V,N,D]
> future_features:  [B,V,N,D]
> ```
>
> 输出：
>
> ```text
> z_delta: [B, token_dim]
> pred_future: [B,V,N,D]
> ```
>
> 注意：当前实现是 **one global delta token per sample**，不是 `[B,V,M,d]` 的多 token 版本。
>
> 也就是说，文档里说的：
>
> ```text
> z_delta [B,V,M,d]
> ```
>
> 是后续更完整、更结构化的设计；当前代码是最简原型：
>
> ```text
> z_delta [B,d]
> ```
>
> 这点很重要。
>
> **当前实验做到哪里**
> 从文档和 synced run 来看：
>
> ```text
> PV-VAE:
>   已实现，已在 LIBERO/SVG-P 上跑过 smoke / 3k / 30k 类实验。
>   发现 static_future_mse 很低，说明静态 copy shortcut 很强。
>
> LAM:
>   已实现，跑过 DINO/SVG-P stride4 smoke 和长训可视化。
>
> DeltaTok-style:
>   已实现 deterministic one-token prototype。
>   已跑过 SVG-P stride4 bs64 3000 steps。
>   最后大概 mse=0.0071, copy_mse=0.0081, delta_ratio=0.87。
> ```
>
> 所以目前不是“从零开始做 tokenizer”，而是：
>
> ```text
> 已有三个原型，但还缺：
> 1. 大规模多数据集 dataloader
> 2. Per-frame S-VAE / channel adapter baseline
> 3. 多 token DeltaTok 版本
> 4. LARY-style action probe
> 5. 统一实验表和复现实验协议
> ```
>
> **接下来该怎么展开**
> 我建议现在不要再开新模型，先把这条线整理成一个严谨的三层研究计划。
>
> **第一层：补最稳 baseline**
> 先实现文档里还没做的：
>
> ```text
> Per-frame S-VAE / Channel Adapter
> ```
>
> 原因很简单：师兄说“训练一个 adapter”，这就是最直接对应。
>
> 目标：
>
> ```text
> x:     [B,V,T,N,D]
> z:     [B,V,T,N,d]
> x_hat: [B,V,T,N,D]
> ```
>
> 比如：
>
> ```text
> D=384 -> d=96
> D=1024 -> d=96
> ```
>
> 它不做 temporal compression，所以能回答最基础问题：
>
> ```text
> 高维 teacher feature 能不能稳定压缩？
> 压缩后 reconstruction 质量如何？
> 压缩后 action information 是否还在？
> ```
>
> 这应该作为 tokenizer 线的 baseline。
>
> **第二层：整理已有 PV-VAE / LAM / DeltaTok**
> 现在已有代码，但缺统一对比。应该把三者放到同一张表里：
>
> ```text
> S-VAE:
>   单帧语义状态压缩
>
> PV-VAE:
>   clip-level temporal compression
>
> LAM:
>   global latent action transition VAE
>
> DeltaTok:
>   deterministic transition token
> ```
>
> 每个方法统一报：
>
> ```text
> feature_mse
> feature_cosine
> copy_mse / static_future_mse
> delta_ratio
> compression ratio
> PCA / SVG decode visualization
> ```
>
> 有 action 的数据再加：
>
> ```text
> action probe L1/L2
> ```
>
> **第三层：再扩展 DeltaTok**
> 当前 DeltaTok 是 one global token：
>
> ```text
> [B,d]
> ```
>
> 后续可以升级成文档里的正式版本：
>
> ```text
> [B,V,M,d]
> ```
>
> 其中：
>
> ```text
> V: 每个 view 单独 token，避免 base/wrist 混在一起
> M: 每个 view 多个 delta tokens，表达多个物体/区域变化
> d: 每个 token 维度
> ```
>
> 建议 ablation：
>
> ```text
> M = 1, 4, 8
> token_dim = 96, 128, 384
> stride = 1, 2, 4
> teacher = SVG-P, DINOv3
> ```
>
> 但这个应该放在 S-VAE baseline 和当前 one-token DeltaTok 对齐之后。
>
> **数据层下一步**
> 你们师兄强调“把数据转成同样格式，用同一个 dataloader”。这里应该先做一个 tokenizer-specific data contract：
>
> ```python
> {
>     "images": Tensor[B,T,V,C,H,W],
>     "actions": Tensor[B,T-1,A] or None,
>     "instruction": list[str] or None,
>     "dataset_name": list[str],
>     "episode_id": list[str],
>     "fps": Tensor[B],
> }
> ```
>
> 训练 tokenizer 时：
>
> ```text
> images -> frozen teacher -> features
> ```
>
> 不要一开始缓存所有 features。先在线 encode 跑通，再考虑缓存。
>
> 数据集分工：
>
> ```text
> LIBERO:
>   先做调试和 action probe
>
> DROID / OXE / Behavior:
>   扩大机器人数据规模
>
> Ego4D:
>   可用于视觉 tokenizer 预训练，但不能做 robot action probe
> ```
>
> **我建议马上做的具体事项**
> 下一步最实际的是写一份执行文档并开始补代码：
>
> ```text
> wam/feature_tokenizer_execution_plan.md
> ```
>
> 内容固定为：
>
> ```text
> 1. 当前已有实现清单
> 2. 三类 tokenizer 输入输出 shape
> 3. 缺失的 S-VAE baseline
> 4. 数据 dataloader contract
> 5. 统一评测指标
> 6. 第一轮实验矩阵
> ```
>
> 然后代码层第一步补：
>
> ```text
> external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
> external/openpi/scripts/train_svae_libero.py
> ```
>
> 这两个正好是 `tokenizer_methods.md` 里建议的入口。
>
> 一个小提醒：当前 `.gitignore` 里有：
>
> ```text
> external/
> ```
>
> 所以你放进 `external/openpi` 的代码不会自动被 git 跟踪。现在这不影响我阅读和帮你设计，但如果后面要 push 这些代码，需要临时 `git add -f external/openpi/...` 或改 `.gitignore`。
</details>

我检查完 GitHub 版仓库里的新代码了。现在判断更准确了：**tokenizer 线不是停在想法阶段，已经有 3 个可读的原型模型和 3 条训练脚本，但还没形成一个统一、可复现实验计划。**

当前真实进度如下。

**已有代码**
现在这三个核心文件已经存在：

```text
external/openpi/src/openpi/models_pytorch/predictive_feature_vae.py
external/openpi/src/openpi/models_pytorch/latent_action.py
external/openpi/src/openpi/models_pytorch/delta_tokenizer.py
```

它们分别对应：

```text
PV-VAE-style temporal predictive tokenizer
DreamDojo-style LAM feature transition VAE
DeltaTok-style deterministic transition tokenizer
```

其中：

**1. PV-VAE 已经是完整模型**
输入：

```text
features: [B, V, F, N, D]
```

其中 `F = 1 + T_future`。

它做的是：

```text
current frame 单独一组
future frames 每 temporal_compression 帧压成一组
```

默认：

```text
temporal_compression = 4
17 frames -> 5 latent groups
```

loss 已经有：

```text
recon_loss
cosine_loss
delta_loss
kl_loss
future_mse
static_future_mse
delta_ratio
```

所以 PV-VAE 这条不是“没做”，而是 **已经实现并做过 LIBERO/SVG-P 小实验**。

**2. LAM 已经是一个 feature latent-action VAE**
输入：

```text
current_features: [B,V,N,D]
future_features:  [B,V,N,D]
```

它编码成一个 global latent action：

```text
current, future -> z_action
current, z_action -> pred_future
```

默认：

```text
latent_dim = 32
model_dim = 512
```

它更像 DreamDojo/LAM，不是正式 DeltaTok，但可以作为 transition representation baseline。

**3. Delta tokenizer 已经有初版**
输入：

```text
current_features: [B,V,N,D]
future_features:  [B,V,N,D]
```

输出：

```text
z_delta: [B, token_dim]
pred_future: [B,V,N,D]
```

注意：当前实现是 **one global delta token per sample**，不是 `[B,V,M,d]` 的多 token 版本。

也就是说，文档里说的：

```text
z_delta [B,V,M,d]
```

是后续更完整、更结构化的设计；当前代码是最简原型：

```text
z_delta [B,d]
```

这点很重要。

**当前实验做到哪里**
从文档和 synced run 来看：

```text
PV-VAE:
  已实现，已在 LIBERO/SVG-P 上跑过 smoke / 3k / 30k 类实验。
  发现 static_future_mse 很低，说明静态 copy shortcut 很强。

LAM:
  已实现，跑过 DINO/SVG-P stride4 smoke 和长训可视化。

DeltaTok-style:
  已实现 deterministic one-token prototype。
  已跑过 SVG-P stride4 bs64 3000 steps。
  最后大概 mse=0.0071, copy_mse=0.0081, delta_ratio=0.87。
```

所以目前不是“从零开始做 tokenizer”，而是：

```text
已有三个原型，但还缺：
1. 大规模多数据集 dataloader
2. Per-frame S-VAE / channel adapter baseline
3. 多 token DeltaTok 版本
4. LARY-style action probe
5. 统一实验表和复现实验协议
```

**接下来该怎么展开**
我建议现在不要再开新模型，先把这条线整理成一个严谨的三层研究计划。

**第一层：补最稳 baseline**
先实现文档里还没做的：

```text
Per-frame S-VAE / Channel Adapter
```

原因很简单：师兄说“训练一个 adapter”，这就是最直接对应。

目标：

```text
x:     [B,V,T,N,D]
z:     [B,V,T,N,d]
x_hat: [B,V,T,N,D]
```

比如：

```text
D=384 -> d=96
D=1024 -> d=96
```

它不做 temporal compression，所以能回答最基础问题：

```text
高维 teacher feature 能不能稳定压缩？
压缩后 reconstruction 质量如何？
压缩后 action information 是否还在？
```

这应该作为 tokenizer 线的 baseline。

**第二层：整理已有 PV-VAE / LAM / DeltaTok**
现在已有代码，但缺统一对比。应该把三者放到同一张表里：

```text
S-VAE:
  单帧语义状态压缩

PV-VAE:
  clip-level temporal compression

LAM:
  global latent action transition VAE

DeltaTok:
  deterministic transition token
```

每个方法统一报：

```text
feature_mse
feature_cosine
copy_mse / static_future_mse
delta_ratio
compression ratio
PCA / SVG decode visualization
```

有 action 的数据再加：

```text
action probe L1/L2
```

**第三层：再扩展 DeltaTok**
当前 DeltaTok 是 one global token：

```text
[B,d]
```

后续可以升级成文档里的正式版本：

```text
[B,V,M,d]
```

其中：

```text
V: 每个 view 单独 token，避免 base/wrist 混在一起
M: 每个 view 多个 delta tokens，表达多个物体/区域变化
d: 每个 token 维度
```

建议 ablation：

```text
M = 1, 4, 8
token_dim = 96, 128, 384
stride = 1, 2, 4
teacher = SVG-P, DINOv3
```

但这个应该放在 S-VAE baseline 和当前 one-token DeltaTok 对齐之后。

**数据层下一步**
你们师兄强调“把数据转成同样格式，用同一个 dataloader”。这里应该先做一个 tokenizer-specific data contract：

```python
{
    "images": Tensor[B,T,V,C,H,W],
    "actions": Tensor[B,T-1,A] or None,
    "instruction": list[str] or None,
    "dataset_name": list[str],
    "episode_id": list[str],
    "fps": Tensor[B],
}
```

训练 tokenizer 时：

```text
images -> frozen teacher -> features
```

不要一开始缓存所有 features。先在线 encode 跑通，再考虑缓存。

数据集分工：

```text
LIBERO:
  先做调试和 action probe

DROID / OXE / Behavior:
  扩大机器人数据规模

Ego4D:
  可用于视觉 tokenizer 预训练，但不能做 robot action probe
```

**我建议马上做的具体事项**
下一步最实际的是写一份执行文档并开始补代码：

```text
wam/feature_tokenizer_execution_plan.md
```

内容固定为：

```text
1. 当前已有实现清单
2. 三类 tokenizer 输入输出 shape
3. 缺失的 S-VAE baseline
4. 数据 dataloader contract
5. 统一评测指标
6. 第一轮实验矩阵
```

然后代码层第一步补：

```text
external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
external/openpi/scripts/train_svae_libero.py
```

这两个正好是 `tokenizer_methods.md` 里建议的入口。

一个小提醒：当前 `.gitignore` 里有：

```text
external/
```

所以你放进 `external/openpi` 的代码不会自动被 git 跟踪。现在这不影响我阅读和帮你设计，但如果后面要 push 这些代码，需要临时 `git add -f external/openpi/...` 或改 `.gitignore`。

> 好的，写一份执行文档并开始补代码

<details><summary>25 previous messages</summary>

> 我来做两件事：先新增一份 `feature_tokenizer_execution_plan.md` 把路线钉住；然后补一个最小可跑的 S-VAE/channel adapter 模型和 LIBERO 训练脚本，尽量复用现有 `train_lam_libero.py` 的数据、teacher encode、可视化工具。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content external\openpi\scripts\train_lam_libero.py | Select-Object -Skip 120 -First 260`
> - Ran `Get-Content external\openpi\scripts\train_deltatok_libero.py | Select-Object -Skip 240 -First 80`
> - Ran `Get-Content external\openpi\scripts\train_predictive_feature_vae_libero.py | Select-Object -Skip 120 -First 180`
> - Ran `Get-ChildItem external\openpi\src\openpi\models_pytorch | Select-Object Name,Length | Sort-Object Name`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我会先把“执行文档”落到仓库里，作为后续和师兄对齐的主线；然后补一个最小可跑的 Per-frame S-VAE/tokenizer 代码入口，让当前三条 tokenizer 路线从文档变成可比较的实验骨架。
>
> <details><summary>Created 2 files</summary>
>
> - Wrote `./awesome-wam-github/external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py` (+82 -0)
> - Wrote `./awesome-wam-github/wam/feature_tokenizer_execution_plan.md` (+316 -0)
>
> </details>
>
> Stream error
>
> Reconnecting... 1/5
> stream disconnected before completion: Transport error: network error: error decoding response body
>
> <details><summary>Created 1 file</summary>
>
> - Wrote `./awesome-wam-github/wam/feature_tokenizer_execution_plan.md` (+167 -0)
>
> </details>
>
> 执行文档已经落下来了。接下来我补 S-VAE 本体和训练入口，尽量复用现有 PV-VAE/LAM 脚本里的 dataloader、SVG/DINO 特征提取和日志风格，这样后面比较实验不会散成三套口径。
>
> <details><summary>Created 1 file</summary>
>
> - Wrote `./awesome-wam-github/external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py` (+161 -0)
>
> </details>
>
> S-VAE 模型已经补上：它只做逐帧语义特征压缩，不偷混时间信息，所以后面可以和 PV-VAE/DeltaTok 很干净地分工比较。现在我继续补 `train_svae_libero.py`，把 LIBERO 图像批次转成 teacher feature clip，再 flatten 成逐帧训练样本。
>
> <details><summary>Created 1 file, ran 6 commands</summary>
>
> - Ran `Get-Content -Path external/openpi/scripts/train_lam_libero.py -TotalCount 260`
> - Ran `Get-Content -Path external/openpi/scripts/train_predictive_feature_vae_libero.py -TotalCount 260`
> - Ran `Get-Content -Path external/openpi/scripts/train_deltatok_libero.py -TotalCount 220`
> - Ran `Select-String -Path external/openpi/scripts/train_lam_libero.py -Pattern "def save_visualization" -Context 0,80`
> - Ran `Select-String -Path external/openpi/scripts/train_lam_libero.py -Pattern "def load_svg_decoder" -Context 0,60`
> - Ran `Select-String -Path external/openpi/scripts/train_predictive_feature_vae_libero.py -Pattern "def main" -Context 0,190`
> - Wrote `./awesome-wam-github/external/openpi/scripts/train_svae_libero.py` (+250 -0)
>
> </details>
>
> 训练入口也补好了。现在我做一轮轻量检查：先编译新 Python 文件看有没有语法问题，再看一下 git 状态，确认哪些东西会被仓库追踪、哪些因为 `external/` 规则暂时还不会进提交。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `git status -sb`
> - Ran `git status --ignored -sb`
> - Ran `python -m py_compile external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py external/openpi/scripts/train_svae_libero.py`
>
> </details>
>
> 语法编译遇到的是 Windows 对 `external/.../__pycache__` 的写权限拒绝，不是代码解析错误。我换成不写 `.pyc` 的 AST 解析检查，专门验证源码语法。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in ['external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py','external/openpi/scripts/train_svae_libero.py']]; print('syntax ok')"`
> - Ran `git ls-files external/openpi | Select-Object -First 20`
> - Ran `git ls-files external | Select-Object -First 20`
> - Ran `Get-Content -Path .gitignore`
>
> </details>
>
> 我发现一个协作层面的坑：仓库已经追踪了几个 `external/openpi` 里的核心脚本，但 `.gitignore` 仍然整体忽略 `external/`，所以我刚补的新 S-VAE 文件默认不会出现在普通 `git add` 里。我会把 `.gitignore` 改成“只允许这些实验相关文件露出来”的白名单，避免之后每次都要 `git add -f`。
>
> <details><summary>Edited 1 file, ran 8 commands</summary>
>
> - Wrote `./awesome-wam-github/.gitignore` (+23 -2)
> - Ran `git status -sb`
> - Ran `git status --ignored -sb`
> - Ran `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in ['external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py','external/openpi/scripts/train_svae_libero.py']]; print('syntax ok')"`
> - Ran `Get-Content -Path wam/feature_tokenizer_execution_plan_zh.md -TotalCount 80`
> - Ran `Get-Content -Path wam/feature_tokenizer_execution_plan_zh.md | Select-Object -Last 80`
> - Ran `git diff -- .gitignore wam/feature_tokenizer_execution_plan.md external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py external/openpi/scripts/train_svae_libero.py`
> - Ran `Get-Content -Path external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py -TotalCount 220`
> - Ran `Get-Content -Path external/openpi/scripts/train_svae_libero.py -TotalCount 260`
>
> </details>
>
> AST 语法检查过了。为了多踩一步，我再用一个随机张量实例化 S-VAE 跑一次 forward/loss，确认 shape contract 真能走通。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `$env:PYTHONPATH='external/openpi/src'; python -c "import torch; from openpi.models_pytorch.semantic_feature_vae import SemanticFeatureVAE, SemanticFeatureVAEConfig; m=SemanticFeatureVAE(SemanticFeatureVAEConfig(feature_dim=8, model_dim=16, latent_dim=4, num_encoder_layers=1, num_decoder_layers=1, num_heads=4)); x=torch.randn(2,1,9,8); y=m.compute_loss(x); print(tuple(y['pred'].shape), tuple(y['z'].shape), float(y['loss']))"`
>
> </details>
>
> 我看到工作区里已经有一份中文执行计划草稿。为了你后面直接拿给师兄看，我补一个“本轮落地状态”小节，让它和刚补的代码对应起来。
>
> <details><summary>Edited 1 file, ran 2 commands</summary>
>
> - Wrote `./awesome-wam-github/wam/feature_tokenizer_execution_plan_zh.md` (+23 -5)
> - Ran `git status -sb`
> - Ran `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in ['external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py','external/openpi/scripts/train_svae_libero.py']]; print('syntax ok')"`
>
> </details>
>
> 还有一个实际跑实验时容易踩的细节：`svg_p` teacher 需要传 SVG autoencoder 的 root/config/checkpoint/weights。文档里的 smoke 命令我补上提示，否则服务器上第一次跑会因为缺 SVG 参数直接停住。
>
> <details><summary>Edited 2 files, ran 2 commands</summary>
>
> - Wrote `./awesome-wam-github/wam/feature_tokenizer_execution_plan.md` (+13 -0)
> - Wrote `./awesome-wam-github/wam/feature_tokenizer_execution_plan_zh.md` (+23 -10)
> - Ran `git status -sb`
> - Ran `$env:PYTHONPATH='external/openpi/src'; python -c "import torch; from openpi.models_pytorch.semantic_feature_vae import SemanticFeatureVAE, SemanticFeatureVAEConfig; m=SemanticFeatureVAE(SemanticFeatureVAEConfig(feature_dim=8, model_dim=16, latent_dim=4, num_encoder_layers=1, num_decoder_layers=1, num_heads=4)); x=torch.randn(2,1,9,8); y=m.compute_loss(x); print(tuple(y['pred'].shape), tuple(y['z'].shape))"`
>
> </details>
>
> 已经开始落地了，当前这一轮做了四件事：
>
> 1. 写了执行文档
>    [英文版](./awesome-wam-github/wam/feature_tokenizer_execution_plan.md)
>    [中文版](./awesome-wam-github/wam/feature_tokenizer_execution_plan_zh.md)
>
> 2. 补了 Per-frame S-VAE tokenizer 模型
>    [semantic_feature_vae.py](./awesome-wam-github/external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py)
>    现在 contract 是：`[B,V,N,D] -> z [B,V,N,d] -> recon [B,V,N,D]`。
>
> 3. 补了 LIBERO 训练入口
>    [train_svae_libero.py](./awesome-wam-github/external/openpi/scripts/train_svae_libero.py)
>    它会把 clip feature `[B,V,F,N,D]` 展平成 `[B*F,V,N,D]` 来训练逐帧 S-VAE，复用现有 PV-VAE/LAM 的 dataloader、SVG-P/DINO feature encoder、可视化和 checkpoint 逻辑。
>
> 4. 调整了 [.gitignore](./awesome-wam-github/.gitignore)
>    仍然默认忽略整个 `external/`，但白名单放出当前项目实际维护的 OpenPI tokenizer/训练脚本文件，之后普通 `git add` 能看到这次新增的代码。
>
> 验证过两步：
>
> ```text
> AST syntax ok
> random tensor test: pred (2,1,9,8), z (2,1,9,4)
> ```
>
> `python -m py_compile` 没跑通是因为 Windows 对 `external/.../__pycache__` 写入拒绝，不是代码语法问题；我改用了不写 `.pyc` 的 AST 检查。
>
> 下一步在服务器上先跑一个 smoke：
>
> ```bash
> python external/openpi/scripts/train_svae_libero.py \
>   --output-dir runs/svae/smoke_svgp \
>   --teacher svg_p \
>   --views base_0_rgb \
>   --future-deltas 1,3,6,9 \
>   --batch-size 8 \
>   --max-steps 100 \
>   --latent-dim 96 \
>   --model-dim 384 \
>   --svg-autoencoder-root ... \
>   --svg-config ... \
>   --svg-checkpoint ... \
>   --svg-dinov3-weights ...
> ```
>
> 当前 `git status` 里应看到：`.gitignore` 修改，以及 4 个新增文件。建议下一步就是把这版提交上去，然后在服务器跑 S-VAE smoke，拿它和已有 PV-VAE / DeltaTok 指标做第一张对比表。
>
> <details><summary>Plan</summary>
>
> - [x] 写 feature tokenizer 执行文档
> - [x] 实现 Per-frame S-VAE 模型
> - [x] 补 LIBERO 训练脚本入口
> - [x] 做语法检查并汇总下一步
>
> </details>
</details>

已经开始落地了，当前这一轮做了四件事：

1. 写了执行文档
   [英文版](./awesome-wam-github/wam/feature_tokenizer_execution_plan.md)
   [中文版](./awesome-wam-github/wam/feature_tokenizer_execution_plan_zh.md)

2. 补了 Per-frame S-VAE tokenizer 模型
   [semantic_feature_vae.py](./awesome-wam-github/external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py)
   现在 contract 是：`[B,V,N,D] -> z [B,V,N,d] -> recon [B,V,N,D]`。

3. 补了 LIBERO 训练入口
   [train_svae_libero.py](./awesome-wam-github/external/openpi/scripts/train_svae_libero.py)
   它会把 clip feature `[B,V,F,N,D]` 展平成 `[B*F,V,N,D]` 来训练逐帧 S-VAE，复用现有 PV-VAE/LAM 的 dataloader、SVG-P/DINO feature encoder、可视化和 checkpoint 逻辑。

4. 调整了 [.gitignore](./awesome-wam-github/.gitignore)
   仍然默认忽略整个 `external/`，但白名单放出当前项目实际维护的 OpenPI tokenizer/训练脚本文件，之后普通 `git add` 能看到这次新增的代码。

验证过两步：

```text
AST syntax ok
random tensor test: pred (2,1,9,8), z (2,1,9,4)
```

`python -m py_compile` 没跑通是因为 Windows 对 `external/.../__pycache__` 写入拒绝，不是代码语法问题；我改用了不写 `.pyc` 的 AST 检查。

下一步在服务器上先跑一个 smoke：

```bash
python external/openpi/scripts/train_svae_libero.py \
  --output-dir runs/svae/smoke_svgp \
  --teacher svg_p \
  --views base_0_rgb \
  --future-deltas 1,3,6,9 \
  --batch-size 8 \
  --max-steps 100 \
  --latent-dim 96 \
  --model-dim 384 \
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

当前 `git status` 里应看到：`.gitignore` 修改，以及 4 个新增文件。建议下一步就是把这版提交上去，然后在服务器跑 S-VAE smoke，拿它和已有 PV-VAE / DeltaTok 指标做第一张对比表。

> 我现在思路有点乱。所以你这一步做的工作，和之前师兄说的，以及整个的研究计划有什么关系

这一步做的不是“正式实验结论”，而是在补齐师兄说的第一件基础设施：**先训练一个 tokenizer**。

师兄聊天里的主线大概是：

1. 数据已经在准备/整理；
2. 想要一个高压缩 VAE；
3. 现在 VAE latent 太多，想参考 PV-VAE；
4. 也可以参考“重构还是语义”那篇，用语义潜空间；
5. 训练一个 adapter/tokenizer；
6. 之后接入高纬度特征；
7. 数据格式统一后，用同一个 dataloader 跑。

我这一步对应的是第 2、4、5、7 点：先补一个**最基础的 Per-frame S-VAE / channel adapter tokenizer**。

它和整个研究计划的关系是：

```text
图像
 -> SVG / DINO / SigLIP 等视觉 encoder 提取高维特征
 -> tokenizer / VAE 把高维特征压缩成 latent tokens
 -> world model / FastWAM 使用这些 latent tokens 做预测、动作建模、评测
```

之前已有代码里已经有两条路线：

```text
PV-VAE:
[B,V,1+T,N,D] -> [B,V,1+T/4,N,d]
关注：时间压缩 + 未来预测

DeltaTok:
x_t, x_{t+k} -> z_delta
关注：状态转移 / 动作相关变化
```

但缺一条最朴素、最容易解释的 baseline：

```text
Per-frame S-VAE:
[B,V,T,N,D] -> [B,V,T,N,d]
关注：每一帧的语义特征压缩
```

所以我补的就是这条 baseline。

它回答的问题是：

> 如果我不建模时间、不预测未来，只把每一帧的高维视觉特征压缩，能不能保留语义信息？

这个 baseline 很重要，因为后面你比较 PV-VAE / DeltaTok 时，才知道提升到底来自哪里：

```text
如果 S-VAE 已经很好：
说明主要问题可能是语义压缩，不一定需要复杂时间建模。

如果 PV-VAE 更好：
说明时间分组/预测确实有用。

如果 DeltaTok 在 action probe 上更好：
说明动作相关的状态变化比完整重构更关键。
```

所以它在研究计划里的位置是：

```text
阶段 0：统一数据格式和 teacher feature 提取
阶段 1：补齐三个 tokenizer baseline
        - Per-frame S-VAE     ← 我这一步补的
        - PV-VAE              ← 之前已有
        - DeltaTok            ← 之前已有
阶段 2：跑同一批 LIBERO 数据，做指标对比
阶段 3：加 LARY-style action probe，看哪个 latent 更动作相关
阶段 4：把最有希望的 tokenizer 接入 FastWAM / OpenPI
```

一句话总结：

**我这一步是在把“我们要训 tokenizer”这件事变成可执行的第一条 baseline。它不是偏离师兄计划，而是在给 PV-VAE、DeltaTok 和未来 FastWAM 接入建立一个可比较的参照组。**

> 请你根据这份实验进度计划说明的md，来看下，你之前的分析以及所做的工作，是否正确
>
> User context
>
> Attachments:
> - tokenizer_methods.md: `./awesome-wam-github/wam/tokenizer_methods.md`

<details><summary>3 previous messages</summary>

> 我先重新按这份 `tokenizer_methods.md` 做一次“对账”：看它真正规定了哪些实验路线、当前实现状态和建议入口，再判断我刚才补的 S-VAE 是否踩在计划线上，还是有偏差需要修正。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path wam/current_fastwam_experiments.md -TotalCount 220`
> - Ran `Get-Content -Path wam/research_refine.md -TotalCount 220`
> - Ran `Get-Content -Path wam/tokenizer_methods.md`
>
> </details>
>
> 看完 `tokenizer_methods.md` 后，我的判断是：
>
> **我之前的分析方向基本正确，但主次需要重新摆清楚。**
>
> 这份 md 里其实已经写得很明确：当前有三条 tokenizer 路线。
>
> ```text
> 1. Per-frame S-VAE / Channel Adapter
>    状态：尚未实现 standalone
>    作用：最稳 baseline
>
> 2. PV-VAE-style Temporal Predictive Feature VAE
>    状态：已实现，已在 LIBERO/SVG-P 跑
>    作用：时序压缩实验，但有 static shortcut 风险
>
> 3. Delta Transition Tokenizer / DeltaTok-style
>    状态：只有 LAM 原型，正式 DeltaTok 未实现
>    作用：最像方法创新主线
> ```
>
> 所以我刚才补的东西：
>
> ```text
> semantic_feature_vae.py
> train_svae_libero.py
> ```
>
> 正好对应 md 里的这一段建议：
>
> ```text
> 建议实现入口：
> - 新增 external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
> - 新增 external/openpi/scripts/train_svae_libero.py
> - 复用 train_predictive_feature_vae_libero.py 和 train_lam_libero.py ...
> ```
>
> 也就是说，**我做的工作是正确的，但它是“补齐最稳 baseline”，不是当前最有创新性的主线。**
>
> 更准确地说：
>
> ```text
> 我这一步 = 把 Per-frame S-VAE 从“文档里说尚未实现”推进到“已有代码入口”
> ```
>
> 它的研究意义是：
>
> ```text
> 先确认：只做每帧 semantic feature 压缩，到底能做到什么水平？
> ```
>
> 后面才能公平判断：
>
> ```text
> PV-VAE 比 S-VAE 好吗？
> 如果好，是不是因为时序压缩真的有用？
>
> DeltaTok 比 S-VAE / PV-VAE 好吗？
> 如果好，是不是因为 transition bottleneck 更动作相关？
> ```
>
> 但有一个地方我需要修正我的表达：
> 之前我说“下一步先跑 S-VAE smoke”是合理的工程顺序，但如果严格按 `tokenizer_methods.md` 的研究判断，**真正主线下一步应该是 Delta tokenizer**，因为文档里写了：
>
> ```text
> 当前判断：
> 1. 最稳 baseline：per-frame S-VAE / channel adapter
> 2. 已实现但风险高：PV-VAE-style temporal compression
> 3. 最像方法创新主线：Delta transition tokenizer
>
> 建议下一步把 Delta tokenizer 作为主线原型
> ```
>
> 所以正确的路线应该是：
>
> ```text
> 短期工程：
> 补齐 S-VAE baseline  ← 我刚做的，正确
>
> 紧接着：
> 实现正式 feature_delta_tokenizer.py  ← 这是主线
>
> 然后：
> S-VAE / PV-VAE / DeltaTok 三者同数据、同 teacher、同指标对比
> ```
>
> 还有一个小技术偏差要注意：
> 我实现的 S-VAE 是一个带 Transformer 的 per-frame feature VAE，它会在同一帧内混合 `V*N` tokens。它满足：
>
> ```text
> [B,V,N,D] -> [B,V,N,d]
> ```
>
> 但如果师兄说的 “Channel Adapter” 想要非常严格地接近 Semantic-WM，那可能还需要一个更轻的版本：
>
> ```text
> 每个 token 独立做 D -> d -> D
> 不做空间 token attention
> ```
>
> 所以后面可以把 S-VAE 分成两个 baseline：
>
> ```text
> S-VAE-MLP / channel-only：最纯 adapter baseline
> S-VAE-Transformer：更强一点的 per-frame baseline
> ```
>
> 结论：
>
> **我做的代码不是偏题，正好补了文档里明确说缺失的 S-VAE baseline；但它不是最终主线。接下来应该把重心转到正式 Delta Transition Tokenizer，因为那才是 md 里判断最有方法创新性的方向。**
</details>

看完 `tokenizer_methods.md` 后，我的判断是：

**我之前的分析方向基本正确，但主次需要重新摆清楚。**

这份 md 里其实已经写得很明确：当前有三条 tokenizer 路线。

```text
1. Per-frame S-VAE / Channel Adapter
   状态：尚未实现 standalone
   作用：最稳 baseline

2. PV-VAE-style Temporal Predictive Feature VAE
   状态：已实现，已在 LIBERO/SVG-P 跑
   作用：时序压缩实验，但有 static shortcut 风险

3. Delta Transition Tokenizer / DeltaTok-style
   状态：只有 LAM 原型，正式 DeltaTok 未实现
   作用：最像方法创新主线
```

所以我刚才补的东西：

```text
semantic_feature_vae.py
train_svae_libero.py
```

正好对应 md 里的这一段建议：

```text
建议实现入口：
- 新增 external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
- 新增 external/openpi/scripts/train_svae_libero.py
- 复用 train_predictive_feature_vae_libero.py 和 train_lam_libero.py ...
```

也就是说，**我做的工作是正确的，但它是“补齐最稳 baseline”，不是当前最有创新性的主线。**

更准确地说：

```text
我这一步 = 把 Per-frame S-VAE 从“文档里说尚未实现”推进到“已有代码入口”
```

它的研究意义是：

```text
先确认：只做每帧 semantic feature 压缩，到底能做到什么水平？
```

后面才能公平判断：

```text
PV-VAE 比 S-VAE 好吗？
如果好，是不是因为时序压缩真的有用？

DeltaTok 比 S-VAE / PV-VAE 好吗？
如果好，是不是因为 transition bottleneck 更动作相关？
```

但有一个地方我需要修正我的表达：
之前我说“下一步先跑 S-VAE smoke”是合理的工程顺序，但如果严格按 `tokenizer_methods.md` 的研究判断，**真正主线下一步应该是 Delta tokenizer**，因为文档里写了：

```text
当前判断：
1. 最稳 baseline：per-frame S-VAE / channel adapter
2. 已实现但风险高：PV-VAE-style temporal compression
3. 最像方法创新主线：Delta transition tokenizer

建议下一步把 Delta tokenizer 作为主线原型
```

所以正确的路线应该是：

```text
短期工程：
补齐 S-VAE baseline  ← 我刚做的，正确

紧接着：
实现正式 feature_delta_tokenizer.py  ← 这是主线

然后：
S-VAE / PV-VAE / DeltaTok 三者同数据、同 teacher、同指标对比
```

还有一个小技术偏差要注意：
我实现的 S-VAE 是一个带 Transformer 的 per-frame feature VAE，它会在同一帧内混合 `V*N` tokens。它满足：

```text
[B,V,N,D] -> [B,V,N,d]
```

但如果师兄说的 “Channel Adapter” 想要非常严格地接近 Semantic-WM，那可能还需要一个更轻的版本：

```text
每个 token 独立做 D -> d -> D
不做空间 token attention
```

所以后面可以把 S-VAE 分成两个 baseline：

```text
S-VAE-MLP / channel-only：最纯 adapter baseline
S-VAE-Transformer：更强一点的 per-frame baseline
```

结论：

**我做的代码不是偏题，正好补了文档里明确说缺失的 S-VAE baseline；但它不是最终主线。接下来应该把重心转到正式 Delta Transition Tokenizer，因为那才是 md 里判断最有方法创新性的方向。**

> 好的，请你要牢记这份文档里的研究进度与计划，不能偏离计划。咱们做到哪了来着，我有点网络

> 好的，请你要牢记这份文档里的研究进度与计划，不能偏离计划。咱们做到哪了来着，我有点忘了

好的，我会按 `tokenizer_methods.md` 这份进度来，不再把主线带偏。

我们现在做到这里了：

```text
总计划：
1. S-VAE / Channel Adapter baseline
2. PV-VAE temporal predictive tokenizer
3. Delta Transition Tokenizer 主线
4. 三者统一评测，对比哪个 latent 更适合机器人 world model/action
```

当前状态：

```text
1. Per-frame S-VAE
文档原状态：尚未实现 standalone
现在状态：我刚补了第一版代码
- semantic_feature_vae.py
- train_svae_libero.py
作用：最稳 baseline，不是主创新点

2. PV-VAE
文档原状态：已实现并在跑 LIBERO/SVG-P
现在状态：未改动
作用：时序压缩 baseline
风险：static-copy shortcut 很强

3. DeltaTok / Delta tokenizer
文档原状态：只有 LAM 原型，还没有正式 DeltaTok
现在状态：还没开始补正式版
作用：最像方法创新主线
下一步应该做这个
```

所以我们当前处在：

```text
已经补齐 S-VAE baseline 的代码入口
下一步要转向正式 Delta Transition Tokenizer
```

更具体一点，刚刚我做了：

```text
新增：
- E:/awesome_wam/awesome-wam-github/wam/feature_tokenizer_execution_plan.md
- E:/awesome_wam/awesome-wam-github/wam/feature_tokenizer_execution_plan_zh.md
- E:/awesome_wam/awesome-wam-github/external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
- E:/awesome_wam/awesome-wam-github/external/openpi/scripts/train_svae_libero.py

修改：
- E:/awesome_wam/awesome-wam-github/.gitignore
```

已经验证：

```text
S-VAE 语法检查通过
随机张量 forward/loss shape 通过
[B,V,N,D] -> pred [B,V,N,D], z [B,V,N,d]
```

但还没有做：

```text
1. 服务器上跑 S-VAE smoke
2. 实现正式 feature_delta_tokenizer.py
3. 实现正式 train_feature_delta_tokenizer_libero.py
4. 建三者统一评测表
5. 做 LARY-style action probe
```

如果按文档最不偏航的顺序，下一步应该是：

```text
Step A：先提交/保存当前 S-VAE baseline
Step B：实现正式 Delta Transition Tokenizer
Step C：用同一套 LIBERO + SVG-P/DINO teacher 跑：
        S-VAE vs PV-VAE vs DeltaTok
Step D：看 static-copy baseline gap、feature MSE/cosine、delta_ratio、可视化
Step E：再进入 action probe / FastWAM 接入
```

一句话：
**我们刚刚把“最稳 baseline：Per-frame S-VAE”补上了；现在真正该进入文档里说的主线：Delta Transition Tokenizer。**

> 好的，我现在已经把代码改动上传到github上了，下一步

<details><summary>9 previous messages</summary>

> 既然 S-VAE 这一版已经同步到 GitHub，下一步就按 `tokenizer_methods.md` 的主线走：补正式的 Delta Transition Tokenizer，而不是继续在 S-VAE 上打转。我先确认本地工作区状态和现有 Delta/LAM 原型，然后开始补主线代码。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `git status -sb`
> - Ran `Get-Content -Path external/openpi/src/openpi/models_pytorch/delta_tokenizer.py`
> - Ran `Get-Content -Path external/openpi/scripts/train_deltatok_libero.py -TotalCount 260`
> - Ran `Get-Content -Path wam/tokenizer_methods.md`
>
> </details>
>
> 本地现在是干净的，说明你上传后这边没有残留未提交改动。现有 `delta_tokenizer.py` 还是“一枚全局 token”的原型，和文档要求的正式入口 `feature_delta_tokenizer.py`、输出 `[B,M,d]` 还有差距；我接下来补这个正式版，并让训练脚本能选择多 delta tokens。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `Get-Content -Path external/openpi/scripts/train_deltatok_libero.py | Select-Object -Skip 260 -First 180`
>
> </details>
>
> 我会保留旧的 `delta_tokenizer.py` 作为历史原型，同时新增文档里点名的 `feature_delta_tokenizer.py`。训练入口会继续用现有 `train_deltatok_libero.py`，但切到正式模型，并增加 `--num-delta-tokens`，这样 M=1/4/8 可以直接做消融。
>
> <details><summary>Created 1 file</summary>
>
> - Wrote `./awesome-wam-github/external/openpi/src/openpi/models_pytorch/feature_delta_tokenizer.py` (+174 -0)
>
> </details>
>
> 正式 Delta tokenizer 本体已经补上，关键变化是 `z_delta` 从旧原型的 `[B,d]` 变成文档要求的 `[B,M,d]`。现在我把训练脚本接到这个新模型上，并把日志指标改成直接报告 `copy_mse / delta_ratio / token_norm` 这些文档指定的诊断量。
>
> <details><summary>Edited 1 file</summary>
>
> - Wrote `./awesome-wam-github/external/openpi/scripts/train_deltatok_libero.py` (+11 -12)
>
> </details>
>
> <details><summary>Plan</summary>
>
> - [x] 确认 DeltaTok 目标 contract
> - [ ] 实现正式 feature_delta_tokenizer.py
> - [ ] 补训练入口和指标
> - [ ] 语法/shape 检查
> - [ ] 更新执行文档
>
> </details>
</details>

> 等一下，你刚刚新增的代码，我只是同步到了github，我还没有在服务器运行。我需要把刚刚增加的- E:/awesome_wam/awesome-wam-github/wam/feature_tokenizer_execution_plan.md
> - E:/awesome_wam/awesome-wam-github/wam/feature_tokenizer_execution_plan_zh.md
> - E:/awesome_wam/awesome-wam-github/external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
> - E:/awesome_wam/awesome-wam-github/external/openpi/scripts/train_svae_libero.py在服务器里面运行嘛

<details><summary>1 previous message</summary>

> 要跑，但**不是全部都“运行”**。
>
> 这四个文件里：
>
> ```text
> wam/feature_tokenizer_execution_plan.md
> wam/feature_tokenizer_execution_plan_zh.md
> ```
>
> 这是文档，不需要在服务器运行，只是给你和师兄对齐研究计划用。
>
> 真正需要在服务器 smoke test 的是：
>
> ```text
> external/openpi/scripts/train_svae_libero.py
> ```
>
> 它会自动 import：
>
> ```text
> external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
> ```
>
> 所以服务器上要做的是：**拉最新 GitHub 代码，然后跑一个很小的 S-VAE smoke 实验**，确认 dataloader、teacher encoder、feature shape、loss、checkpoint 都能走通。
>
> 先不要跑长实验。建议第一步只跑 50-100 steps。
>
> 如果服务器上 SVG-P 路径都准备好了，跑：
>
> ```bash
> python external/openpi/scripts/train_svae_libero.py \
>   --output-dir runs/svae/smoke_svgp \
>   --teacher svg_p \
>   --views base_0_rgb \
>   --future-deltas 1,3,6,9 \
>   --batch-size 8 \
>   --max-steps 100 \
>   --latent-dim 96 \
>   --model-dim 384 \
>   --svg-autoencoder-root ... \
>   --svg-config ... \
>   --svg-checkpoint ... \
>   --svg-dinov3-weights ...
> ```
>
> 如果 SVG-P 的路径你暂时不确定，先用 DINOv3 smoke 更简单：
>
> ```bash
> python external/openpi/scripts/train_svae_libero.py \
>   --output-dir runs/svae/smoke_dinov3 \
>   --teacher dinov3_vits16 \
>   --dinov3-path assets/dinov3-vits16-pretrain-lvd1689m \
>   --views base_0_rgb \
>   --future-deltas 1,3,6,9 \
>   --batch-size 8 \
>   --max-steps 100 \
>   --latent-dim 96 \
>   --model-dim 384 \
>   --no-decode-svg-rgb
> ```
>
> 这一步的目的不是出结果，而是确认：
>
> ```text
> 1. 服务器代码能 import
> 2. LIBERO dataloader 能出 current/future image
> 3. SVG-P/DINO teacher 能提 feature
> 4. S-VAE 的 [B,V,N,D] -> [B,V,N,d] -> [B,V,N,D] 能训练
> 5. loss 正常下降或至少不 NaN
> 6. checkpoint / args.json / 可视化目录能生成
> ```
>
> 然后我们再进入文档里的主线：**正式 Delta Transition Tokenizer**。
>
> 另外提醒一下：我刚刚在你打断前已经开始新增了一个 DeltaTok 文件 `feature_delta_tokenizer.py`，那是下一步的本地未完成改动，不属于你刚才已经同步 GitHub 的 S-VAE 版本。现在先别管它，先把 S-VAE smoke 在服务器跑通。
</details>

要跑，但**不是全部都“运行”**。

这四个文件里：

```text
wam/feature_tokenizer_execution_plan.md
wam/feature_tokenizer_execution_plan_zh.md
```

这是文档，不需要在服务器运行，只是给你和师兄对齐研究计划用。

真正需要在服务器 smoke test 的是：

```text
external/openpi/scripts/train_svae_libero.py
```

它会自动 import：

```text
external/openpi/src/openpi/models_pytorch/semantic_feature_vae.py
```

所以服务器上要做的是：**拉最新 GitHub 代码，然后跑一个很小的 S-VAE smoke 实验**，确认 dataloader、teacher encoder、feature shape、loss、checkpoint 都能走通。

先不要跑长实验。建议第一步只跑 50-100 steps。

如果服务器上 SVG-P 路径都准备好了，跑：

```bash
python external/openpi/scripts/train_svae_libero.py \
  --output-dir runs/svae/smoke_svgp \
  --teacher svg_p \
  --views base_0_rgb \
  --future-deltas 1,3,6,9 \
  --batch-size 8 \
  --max-steps 100 \
  --latent-dim 96 \
  --model-dim 384 \
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

如果 SVG-P 的路径你暂时不确定，先用 DINOv3 smoke 更简单：

```bash
python external/openpi/scripts/train_svae_libero.py \
  --output-dir runs/svae/smoke_dinov3 \
  --teacher dinov3_vits16 \
  --dinov3-path assets/dinov3-vits16-pretrain-lvd1689m \
  --views base_0_rgb \
  --future-deltas 1,3,6,9 \
  --batch-size 8 \
  --max-steps 100 \
  --latent-dim 96 \
  --model-dim 384 \
  --no-decode-svg-rgb
```

这一步的目的不是出结果，而是确认：

```text
1. 服务器代码能 import
2. LIBERO dataloader 能出 current/future image
3. SVG-P/DINO teacher 能提 feature
4. S-VAE 的 [B,V,N,D] -> [B,V,N,d] -> [B,V,N,D] 能训练
5. loss 正常下降或至少不 NaN
6. checkpoint / args.json / 可视化目录能生成
```

然后我们再进入文档里的主线：**正式 Delta Transition Tokenizer**。

另外提醒一下：我刚刚在你打断前已经开始新增了一个 DeltaTok 文件 `feature_delta_tokenizer.py`，那是下一步的本地未完成改动，不属于你刚才已经同步 GitHub 的 S-VAE 版本。现在先别管它，先把 S-VAE smoke 在服务器跑通。

> python external/openpi/scripts/train_svae_libero.py \
>   --output-dir runs/svae/smoke_svgp \
>   --teacher svg_p \
>   --views base_0_rgb \
>   --future-deltas 1,3,6,9 \
>   --batch-size 8 \
>   --max-steps 100 \
>   --latent-dim 96 \
>   --model-dim 384 \
>   --svg-autoencoder-root ... \
>   --svg-config ... \
>   --svg-checkpoint ... \
>   --svg-dinov3-weights ...
> Traceback (most recent call last):
>   File "/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/external/openpi/scripts/train_svae_libero.py", line 27, in <module>
>     from openpi.models_pytorch.dinov3_vit import load_dinov3_patch_encoder
> ModuleNotFoundError: No module named 'openpi'

<details><summary>2 previous messages</summary>

> 这个报错很清楚：脚本文件找到了，但 Python 的 import 路径里没有 `external/openpi/src`，所以找不到 `openpi` 包。先不用怀疑 S-VAE 代码本身，这是运行环境/PYTHONPATH 问题。
>
> <details><summary>Ran 6 commands</summary>
>
> - Ran `Get-ChildItem -Path external/openpi -Filter pyproject.toml -Recurse | Select-Object -First 10 FullName`
> - Ran `Get-ChildItem -Path external/openpi -Force | Select-Object Name,Mode`
> - Ran `Get-Content -Path external/openpi/pyproject.toml -TotalCount 120`
> - Ran `Get-ChildItem -Path external/openpi/src/openpi/models_pytorch -Force | Select-Object -First 20 Name,Mode`
> - Ran `Get-ChildItem -Path external/openpi/src/openpi -Force | Select-Object Name,Mode`
> - Ran `Get-ChildItem -Path external/openpi/src/openpi/training -Force -ErrorAction SilentlyContinue | Select-Object -First 20 Name,Mode`
>
> </details>
</details>
