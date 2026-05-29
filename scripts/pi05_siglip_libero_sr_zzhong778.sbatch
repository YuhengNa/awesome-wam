#!/usr/bin/env bash
#SBATCH -J pi05_siglip_sr
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=180G
#SBATCH -t 2-00:00:00
#SBATCH -o /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_siglip_sr_%x_%j.out
#SBATCH -e /data/user/zzhong778/git-project/openpi_awam_smoke_20260516/logs/pi05_siglip_sr_%x_%j.err

set -euo pipefail

REPO="/data/user/zzhong778/git-project/openpi_awam_smoke_20260516"
CKPT="${REPO}/checkpoints/pi05_libero_gen/pi05_libero_gen39_bs64_gw05_30k_20260518_142909/30000"
CLIENT_PY="/data/user/jhe724/.conda/envs/fastwam/bin/python"
LIBERO_HOME="/data/user/zzhong778/LIBERO"
SITE_BRIDGE="/data/user/zzhong778/fastwam_jobs/hpc_zzhong_sitecustomize"

SUITE="${SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
PORT="${PORT:-$((18000 + SLURM_JOB_ID % 10000))}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO}/runs/libero_sr/pi05_siglip_step30000_rp${REPLAN_STEPS}_${STAMP}}"
OUT_DIR="${OUT_ROOT}/${SUITE}"

mkdir -p "${REPO}/logs" "${OUT_DIR}"
cd "${REPO}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/user/zzhong778/.cache/openpi}"
export HF_HOME="${HF_HOME:-/data/user/zzhong778/.cache/huggingface}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true

SERVER_LOG="${OUT_DIR}/server.log"
CLIENT_LOG="${OUT_DIR}/client.log"

echo "[eval] suite=${SUITE}"
echo "[eval] trials=${NUM_TRIALS}"
echo "[eval] replan_steps=${REPLAN_STEPS}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] out_dir=${OUT_DIR}"
echo "[eval] port=${PORT}"
echo "[eval] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

PYTHONPATH="${REPO}/scripts:${REPO}/src:${REPO}/third_party" \
  "${REPO}/.venv/bin/python" scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config=pi05_libero_gen \
  --policy.dir="${CKPT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${REPO}/.venv/bin/python" - <<PY
import socket
import time
host = "127.0.0.1"
port = ${PORT}
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=3):
            raise SystemExit(0)
    except OSError:
        time.sleep(5)
raise SystemExit(f"server did not open {host}:{port}")
PY

export PYTHONPATH="${SITE_BRIDGE}:${LIBERO_HOME}:${REPO}/packages/openpi-client/src:${REPO}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="/data/user/zzhong778/.libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/openpi_libero_numba_${USER}}"
mkdir -p "${NUMBA_CACHE_DIR}"

"${CLIENT_PY}" examples/libero/eval_sr_no_tyro.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --task-suite-name "${SUITE}" \
  --num-trials-per-task "${NUM_TRIALS}" \
  --replan-steps "${REPLAN_STEPS}" \
  --resize-size 224 \
  --output-dir "${OUT_DIR}" \
  >"${CLIENT_LOG}" 2>&1

echo "[eval] done suite=${SUITE}"
cat "${OUT_DIR}/results.json"
