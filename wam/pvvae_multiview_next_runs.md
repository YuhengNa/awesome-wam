# PV-VAE Multi-Dataset Multi-View Next Runs

当前目标不是接 FastWAM，而是继续评估 frozen SVG-P + trainable PV-VAE adapter 的压缩表征质量。

## 为什么分 main-view 和 multi-view

- `pvvae_multidataset_mainview_spec.json`：稳定 baseline。每个数据集只选一个最可靠主视角，方便确认多数据集训练本身是否成立。
- `pvvae_multidataset_multiview_spec.json`：更接近 WAM。Bridge 使用 2 视角，Behavior 使用 head + left/right wrist，Fractal/DROID 使用当前审计可靠的单视角。

由于不同 source 的视角数不同，multi-view mixed training 需要：

```text
--mixture-source-batch-mode homogeneous
```

这样每个 batch 内来自同一个 source，视角数一致；不同 batch 仍然按 source 权重混合。

## 1. Loader Dry-Run

目的：确认多数据集、多视角、不同 fps 对齐、黑屏过滤都能正常采样。

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/train_predictive_feature_vae_lerobot.py \
  --dataset-spec-json wam/pvvae_multidataset_multiview_spec.json \
  --output-dir runs/pvvae/multidataset_multiview_dryrun \
  --teacher svg_p \
  --time-sampling-mode duration_sec \
  --clip-duration-sec 5.333 \
  --num-future-frames 16 \
  --batch-size 4 \
  --num-workers 4 \
  --max-episodes 64 \
  --samples-per-episode 8 \
  --mixture-source-batch-mode homogeneous \
  --min-rgb-delta 0.02 \
  --min-rgb-mean 0.02 \
  --min-rgb-std 0.02 \
  --dry-run-loader
```

## 2. Loader Visualization

目的：肉眼确认每个 source 的每个 view 是否合理，尤其 Behavior wrist、Bridge image_1、DROID external。

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/visualize_lerobot_loader_clips.py \
  --dataset-spec-json wam/pvvae_multidataset_multiview_spec.json \
  --output-dir runs/loader_vis/multidataset_multiview \
  --time-sampling-mode duration_sec \
  --clip-duration-sec 5.333 \
  --num-future-frames 16 \
  --batch-size 4 \
  --num-workers 0 \
  --max-episodes 64 \
  --samples-per-episode 8 \
  --mixture-source-batch-mode homogeneous \
  --min-rgb-delta 0.02 \
  --min-rgb-mean 0.02 \
  --min-rgb-std 0.02 \
  --num-clips 32
```

## 3. Multi-View Feature Stats

目的：为 multi-view mixed adapter 训练生成匹配的 SVG-P feature channel mean/std。

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/compute_lerobot_feature_stats.py \
  --dataset-spec-json wam/pvvae_multidataset_multiview_spec.json \
  --output runs/pvvae_stats/multidataset_multiview_svgp_5p33s.pt \
  --summary-json runs/pvvae_stats/multidataset_multiview_svgp_5p33s.json \
  --teacher svg_p \
  --time-sampling-mode duration_sec \
  --clip-duration-sec 5.333 \
  --num-future-frames 16 \
  --batch-size 4 \
  --num-workers 4 \
  --max-episodes 512 \
  --samples-per-episode 16 \
  --mixture-source-batch-mode homogeneous \
  --min-rgb-delta 0.02 \
  --min-rgb-mean 0.02 \
  --min-rgb-std 0.02 \
  --max-batches 200 \
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

## 4. 1k Smoke Training

目的：确认多视角 mixed PV-VAE 能正常反传、保存、可视化。

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/train_predictive_feature_vae_lerobot.py \
  --dataset-spec-json wam/pvvae_multidataset_multiview_spec.json \
  --output-dir runs/pvvae/multidataset_multiview_svgp_obs3_5_smoke \
  --teacher svg_p \
  --time-sampling-mode duration_sec \
  --clip-duration-sec 5.333 \
  --num-future-frames 16 \
  --batch-size 4 \
  --num-workers 4 \
  --max-episodes 512 \
  --samples-per-episode 16 \
  --mixture-source-batch-mode homogeneous \
  --observed-groups 0 \
  --min-observed-groups 3 \
  --feature-normalization channel_standard \
  --feature-stats runs/pvvae_stats/multidataset_multiview_svgp_5p33s.pt \
  --min-rgb-delta 0.02 \
  --min-rgb-mean 0.02 \
  --min-rgb-std 0.02 \
  --delta-weight 0.5 \
  --future-loss-weight 1.0 \
  --max-steps 1000 \
  --save-interval 1000 \
  --vis-interval 500 \
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

## 5. 正式训练候选

如果 1k smoke 正常，再启动 20k baseline：

```bash
PYTHONPATH=$PWD/external/openpi/src:$PYTHONPATH python external/openpi/scripts/train_predictive_feature_vae_lerobot.py \
  --dataset-spec-json wam/pvvae_multidataset_multiview_spec.json \
  --output-dir runs/pvvae/multidataset_multiview_svgp_obs3_5_20k \
  --teacher svg_p \
  --time-sampling-mode duration_sec \
  --clip-duration-sec 5.333 \
  --num-future-frames 16 \
  --batch-size 4 \
  --num-workers 4 \
  --max-episodes 0 \
  --samples-per-episode 16 \
  --mixture-source-batch-mode homogeneous \
  --observed-groups 0 \
  --min-observed-groups 3 \
  --feature-normalization channel_standard \
  --feature-stats runs/pvvae_stats/multidataset_multiview_svgp_5p33s.pt \
  --min-rgb-delta 0.02 \
  --min-rgb-mean 0.02 \
  --min-rgb-std 0.02 \
  --delta-weight 0.5 \
  --future-loss-weight 1.0 \
  --max-steps 20000 \
  --save-interval 5000 \
  --vis-interval 2000 \
  --svg-autoencoder-root ... \
  --svg-config ... \
  --svg-checkpoint ... \
  --svg-dinov3-weights ...
```

## 当前需要观察的结果

- dry-run: `images` shape 是否随 source 变成 `[B,1,T,C,H,W]` / `[B,2,T,C,H,W]` / `[B,3,T,C,H,W]`。
- loader visualization: 是否仍有黑屏/占位视角混入。
- stats: channel std 是否仍明显不均衡，确认继续使用 `channel_standard`。
- smoke training: loss 是否下降，`future_mse`、`delta_ratio`、SVG decoded visualization 是否正常。
