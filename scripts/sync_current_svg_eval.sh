#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-HPC3_zzhong778}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/runs}"

REMOTE_RUNS=(
  "/data/user/jhe724/workspace/FastWAM/runs/libero_svg_dino_p_2cam256_future4_1e-4/zz_svgp_future4_10ep_300362_20260514_175129"
  "/data/user/jhe724/workspace/FastWAM/runs/libero_svg_dino_p_vaecond_2cam256_future4_1e-4/zz_svgp_vaecond_future4_10ep_300363_20260514_175133"
  "/data/user/jhe724/workspace/FastWAM/runs/libero_svg_dino_p_compact_2cam256_future4_1e-4/zz_svgp_compact_future4_10ep_301745_20260515_171741"
)

for remote_run in "${REMOTE_RUNS[@]}"; do
  task="$(basename "$(dirname "${remote_run}")")"
  run_name="$(basename "${remote_run}")"
  local_run="${LOCAL_ROOT}/${task}/${run_name}"
  local_eval="${local_run}/eval"
  local_logs="${LOCAL_ROOT}/${task}/_slurm_logs"

  mkdir -p "${local_eval}"
  mkdir -p "${local_logs}"
  scp -q "${REMOTE}:${remote_run}/dataset_stats.json" "${local_run}/" 2>/dev/null || true
  scp -q "${REMOTE}:$(dirname "${remote_run}")/_slurm_logs/"*.out "${local_logs}/" 2>/dev/null || true
  scp -q "${REMOTE}:$(dirname "${remote_run}")/_slurm_logs/"*.err "${local_logs}/" 2>/dev/null || true

  mapfile -t eval_files < <(
    ssh -o BatchMode=yes "${REMOTE}" \
      "find '${remote_run}/eval' -maxdepth 1 -type f \\( -name '*.mp4' -o -name '*.json' \\) -printf '%f\n' 2>/dev/null | sort" \
      2>/dev/null || true
  )

  for file_name in "${eval_files[@]}"; do
    [[ -n "${file_name}" ]] || continue
    if [[ ! -s "${local_eval}/${file_name}" ]]; then
      scp -q "${REMOTE}:${remote_run}/eval/${file_name}" "${local_eval}/" 2>/dev/null || true
    fi
  done
done

if [[ -f "scripts/monitor_compact_feature_loss.py" ]]; then
  python scripts/monitor_compact_feature_loss.py \
    --output "${LOCAL_ROOT}/libero_svg_dino_p_compact_2cam256_future4_1e-4/compact_vs_full_feature_loss.md" \
    >/dev/null 2>&1 || true
fi
