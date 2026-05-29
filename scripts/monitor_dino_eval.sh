#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-HPC3_jhe724}"
JOB_ID="${JOB_ID:-297695}"
TASK="${TASK:-libero_dinov3_vitb_2cam224_1e-4}"
RUN_ID="${RUN_ID:-2026-05-12_23-16-48}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"

REMOTE_REPO="${REMOTE_REPO:-/data/user/jhe724/workspace/FastWAM}"
REMOTE_RUN="${REMOTE_REPO}/runs/${TASK}/${RUN_ID}"
REMOTE_EVAL="${REMOTE_RUN}/eval"
REMOTE_LOG="${REMOTE_LOG:-}"

LOCAL_ROOT="${LOCAL_ROOT:-/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/runs}"
LOCAL_RUN="${LOCAL_ROOT}/${TASK}/${RUN_ID}"
LOCAL_EVAL="${LOCAL_RUN}/eval"
LOCAL_LOG_DIR="${LOCAL_RUN}/monitor"
LOCAL_MONITOR_LOG="${LOCAL_LOG_DIR}/monitor.log"

mkdir -p "${LOCAL_EVAL}" "${LOCAL_LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOCAL_MONITOR_LOG}"
}

sync_eval() {
  if ssh -o BatchMode=yes "${REMOTE}" "find '${REMOTE_EVAL}' -maxdepth 1 -type f -name '*.mp4' -print -quit 2>/dev/null" | grep -q .; then
    scp -q -o BatchMode=yes "${REMOTE}:${REMOTE_EVAL}/"*.mp4 "${LOCAL_EVAL}/"
    log "synced eval videos; local_count=$(find "${LOCAL_EVAL}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  else
    log "no eval videos found yet"
  fi
}

while true; do
  log "checking job ${JOB_ID}"
  ssh -o BatchMode=yes "${REMOTE}" "squeue -j '${JOB_ID}' -o '%.18i %.9P %.24j %.8T %.10M %.6D %R'" \
    | tee -a "${LOCAL_MONITOR_LOG}" || true
  if [[ -z "${REMOTE_LOG}" ]]; then
    REMOTE_LOG="$(ssh -o BatchMode=yes "${REMOTE}" "ls -t '${REMOTE_REPO}/runs/${TASK}/_launcher_logs'/train_*.log 2>/dev/null | head -n 1" || true)"
  fi
  if [[ -n "${REMOTE_LOG}" ]]; then
    ssh -o BatchMode=yes "${REMOTE}" "tail -n 40 '${REMOTE_LOG}' 2>/dev/null" \
      | tee -a "${LOCAL_MONITOR_LOG}" || true
  else
    log "launcher log not found yet"
  fi

  if [[ ! -d "${LOCAL_EVAL}" ]]; then
    mkdir -p "${LOCAL_EVAL}"
  fi
  if ! ssh -o BatchMode=yes "${REMOTE}" "test -d '${REMOTE_EVAL}'"; then
    latest_run="$(ssh -o BatchMode=yes "${REMOTE}" "find '${REMOTE_REPO}/runs/${TASK}' -maxdepth 1 -mindepth 1 -type d -name '20*' -printf '%f\n' 2>/dev/null | sort | tail -n 1" || true)"
    if [[ -n "${latest_run}" && "${latest_run}" != "${RUN_ID}" ]]; then
      log "remote eval dir not found for RUN_ID=${RUN_ID}; latest run appears to be ${latest_run}"
    fi
  fi

  sync_eval

  if ! ssh -o BatchMode=yes "${REMOTE}" "squeue -h -j '${JOB_ID}'" | grep -q .; then
    log "job ${JOB_ID} is no longer in squeue; final sacct follows"
    ssh -o BatchMode=yes "${REMOTE}" \
      "sacct -j '${JOB_ID}' --format=JobID,JobName%24,State,Elapsed,ExitCode -P 2>/dev/null" \
      | tee -a "${LOCAL_MONITOR_LOG}" || true
    sync_eval
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
