#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="HPC3_zzhong778"
REMOTE_DIR="/data/user/jhe724/workspace/FastWAM/runs/libero_svg_resvit_2cam256_future4_1e-4/zz_svg_future4_10ep_299097_20260513_200104/eval"
LOCAL_DIR="/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/runs/libero_svg_resvit_2cam256_future4_1e-4/zz_svg_future4_10ep_299097_20260513_200104/eval"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
SYNC_ONCE="${SYNC_ONCE:-0}"

mkdir -p "${LOCAL_DIR}"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync start"
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "find '${REMOTE_DIR}' -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sort" |
  while IFS= read -r filename; do
    [[ -z "${filename}" ]] && continue
    if [[ ! -f "${LOCAL_DIR}/${filename}" ]]; then
      echo "copy ${filename}"
      scp -q -o BatchMode=yes "${REMOTE_HOST}:${REMOTE_DIR}/${filename}" "${LOCAL_DIR}/"
    fi
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync done"
  if [[ "${SYNC_ONCE}" == "1" ]]; then
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
