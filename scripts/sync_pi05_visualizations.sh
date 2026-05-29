#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-HPC3_zzhong778}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/user/zzhong778/git-project/openpi_awam_smoke_20260516/checkpoints}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/runs}"

sync_run() {
  local remote_rel="$1"
  local local_name="$2"
  local remote_dir="${REMOTE_ROOT}/${remote_rel}"
  local local_dir="${LOCAL_ROOT}/${local_name}"

  mkdir -p "${local_dir}/gen_visualizations" "${local_dir}/prefix_hidden_visualizations"

  for subdir in gen_visualizations prefix_hidden_visualizations; do
    if ssh "${REMOTE}" "test -d '${remote_dir}/${subdir}'"; then
      rsync -a --ignore-existing \
        "${REMOTE}:${remote_dir}/${subdir}/" \
        "${local_dir}/${subdir}/"
    else
      echo "skip missing ${remote_dir}/${subdir}"
    fi
  done
}

sync_run \
  "pi05_libero_gen/pi05_libero_gen39_bs64_gw05_30k_20260518_142909" \
  "pi05_libero_gen39_bs64_gw05_30k_20260518_142909"

sync_run \
  "pi05_libero_gen_dino/pi05_libero_gen_dino_bs64_gw05_30k_20260518_185723" \
  "pi05_libero_gen_dino_bs64_gw05_30k_20260518_185723"

sync_run \
  "pi05_libero_gen_dino32/pi05_libero_gen_dino32_bs64_gw05_30k_20260518_205359" \
  "pi05_libero_gen_dino32_bs64_gw05_30k_20260518_205359"

latest_dino128="$(
  ssh "${REMOTE}" "ls -dt '${REMOTE_ROOT}'/pi05_libero_gen_dino128/pi05_libero_gen_dino128_bs64_gw05_30k_* 2>/dev/null | head -1" \
    | sed "s#^${REMOTE_ROOT}/##"
)"
if [[ -n "${latest_dino128}" ]]; then
  sync_run "${latest_dino128}" "$(basename "${latest_dino128}")"
else
  echo "skip missing latest pi05_libero_gen_dino128 run"
fi
