#!/bin/bash
#SBATCH -J pi05_gen_overfit
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -t 0-04:00:00
#SBATCH -o /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_gen_overfit.%j.log
#SBATCH -e /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_gen_overfit.%j.err

set -euo pipefail

cd /data/user/zzhong778/git-project/openpi_awam_smoke_20260516
mkdir -p logs runs/pi05_gen_overfit

export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/third_party"
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

RUN_NAME="${RUN_NAME:-rawsiglip_svg_b2_s1000_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="runs/pi05_gen_overfit/${RUN_NAME}"

echo "repo=$PWD"
echo "out_dir=$OUT_DIR"
echo "config=pi05_libero_gen target=raw_siglip norm=fixed_stats sampling=svg_shift batch=2 steps=1000 gen_loss_weight=1.0"

.venv/bin/python scripts/overfit_pi05_gen.py \
  --config pi05_libero_gen \
  --output-dir "$OUT_DIR" \
  --batch-size 2 \
  --steps 1000 \
  --lr 5e-5 \
  --gen-loss-weight 1.0 \
  --log-interval 20 \
  --vis-interval 100 \
  --sampling-steps 20 \
  --max-vis-items 1 \
  --num-workers 0 \
  --device cuda
