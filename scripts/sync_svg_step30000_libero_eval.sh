#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-HPC3_zzhong778}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/user/jhe724/workspace/FastWAM/runs/libero_eval}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/runs/libero_eval}"

PATTERNS=(
  "libero_svg_dino_p_2cam256_future4_1e-4_step_030000_ah16_rp10_full_zzhong778_*"
  "libero_svg_dino_p_vaecond_2cam256_future4_1e-4_step_030000_ah16_rp10_full_zzhong778_*"
)

mkdir -p "${LOCAL_ROOT}"

for pattern in "${PATTERNS[@]}"; do
  mapfile -t remote_dirs < <(
    ssh -o BatchMode=yes "${REMOTE}" \
      "find '${REMOTE_ROOT}' -maxdepth 1 -type d -name '${pattern}' -printf '%f\n' 2>/dev/null | sort" \
      2>/dev/null || true
  )

  for dir_name in "${remote_dirs[@]}"; do
    [[ -n "${dir_name}" ]] || continue
    mkdir -p "${LOCAL_ROOT}/${dir_name}"
    rsync -a "${REMOTE}:${REMOTE_ROOT}/${dir_name}/" "${LOCAL_ROOT}/${dir_name}/"
    echo "Synced ${REMOTE_ROOT}/${dir_name} -> ${LOCAL_ROOT}/${dir_name}"
  done
done
