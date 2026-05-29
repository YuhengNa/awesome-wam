#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam"
PID_FILE="/tmp/pi05_vis_sync_loop.pid"
LOG_FILE="${ROOT}/runs/pi05_visualization_sync.log"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "sync loop already running: ${old_pid}"
    exit 0
  fi
fi

echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

while true; do
  {
    date '+%Y-%m-%d %H:%M:%S'
    "${ROOT}/scripts/sync_pi05_visualizations.sh"
    echo
  } >> "${LOG_FILE}" 2>&1
  sleep "${SYNC_INTERVAL_SECONDS:-600}"
done
