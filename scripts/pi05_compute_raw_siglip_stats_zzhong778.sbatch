#!/bin/bash
#SBATCH -J pi05_siglip_stats
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH -t 0-04:00:00
#SBATCH -o /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_siglip_stats.%j.log
#SBATCH -e /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_siglip_stats.%j.err

set -euo pipefail

cd /data/user/zzhong778/git-project/openpi_awam_smoke_20260516
mkdir -p logs assets/pi05_libero_gen/physical-intelligence/libero

export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/third_party"
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "repo=$PWD"
echo "config=pi05_libero_gen target=raw_siglip stats=assets/pi05_libero_gen/physical-intelligence/libero/raw_siglip_stats.pt"

.venv/bin/python scripts/compute_pi05_siglip_stats.py \
  --config pi05_libero_gen \
  --output assets/pi05_libero_gen/physical-intelligence/libero/raw_siglip_stats.pt \
  --batch-size 8 \
  --num-batches 200 \
  --num-workers 4 \
  --device cuda
