#!/bin/bash
#SBATCH -J pi05_gen_vis_fix
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH -o /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_gen_vis_fix.%j.log
#SBATCH -e /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_gen_vis_fix.%j.err

set -euo pipefail

cd /data/user/zzhong778/git-project/openpi_awam_smoke_20260516

export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/third_party"
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

RUN_DIR="$PWD/checkpoints/pi05_libero_gen/pi05_libero_gen39_bs64_gw05_30k_20260517_024928"
OUT_DIR="$PWD/runs/pi05_gen39_posthoc_vis_fixed_pos_step15000"

.venv/bin/python -u scripts/visualize_pi05_gen_posthoc.py \
  --config pi05_libero_gen \
  --run-dir "$RUN_DIR" \
  --ckpt-step 15000 \
  --output-dir "$OUT_DIR" \
  --batch-size 1 \
  --num-steps 20 \
  --max-items 1
