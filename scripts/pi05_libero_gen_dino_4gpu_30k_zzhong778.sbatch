#!/bin/bash
#SBATCH -J pi05_dino_30k
#SBATCH -p acd_u
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH -t 2-00:00:00
#SBATCH -o /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_dino_30k.%j.log
#SBATCH -e /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_dino_30k.%j.err

set -euo pipefail

cd /data/user/zzhong778/git-project/openpi_awam_smoke_20260516
mkdir -p logs

export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/third_party"
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

EXP_NAME="${EXP_NAME:-pi05_libero_gen_dino_bs64_gw05_30k_$(date +%Y%m%d_%H%M%S)}"

echo "repo=$PWD"
echo "exp_name=$EXP_NAME"
echo "config=pi05_libero_gen_dino"
echo "world_size=4 global_bs=64 future=(3,9) action_horizon=10 gen_target=raw_dinov3_vits16 gen_target_norm=fixed_stats gen_time_sampling=svg_shift gen_svg_infer_shift=6 gen_loss_weight=0.5 save_interval=5000 steps=30000"

.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  scripts/train_pytorch.py \
  pi05_libero_gen_dino \
  --exp_name "$EXP_NAME"
