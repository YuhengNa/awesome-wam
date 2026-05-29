#!/bin/bash
set -euo pipefail

REMOTE="${REMOTE:-HPC3_zzhong778}"
JOB_ID="${JOB_ID:-303646}"
REMOTE_REPO="${REMOTE_REPO:-/data/user/zzhong778/git-project/openpi_awam_smoke_20260516}"
RUN_NAME="${RUN_NAME:-pi05_libero_gen39_bs64_gw05_30k_20260517_024928}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-$REMOTE_REPO/checkpoints/pi05_libero_gen/$RUN_NAME}"
REMOTE_LOG_ERR="${REMOTE_LOG_ERR:-$REMOTE_REPO/logs/pi05_gen39_30k.$JOB_ID.err}"
REMOTE_LOG_OUT="${REMOTE_LOG_OUT:-$REMOTE_REPO/logs/pi05_gen39_30k.$JOB_ID.log}"
LOCAL_DIR="${LOCAL_DIR:-/data/LFT-W02_data/zhongzd/cc_projects/awesome_wam/runs/pi05_libero_gen39_bs64_gw05_30k_20260517_024928}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"

mkdir -p "$LOCAL_DIR/logs"

echo "monitor_start $(date '+%F %T')"
echo "remote=$REMOTE"
echo "job_id=$JOB_ID"
echo "remote_run_dir=$REMOTE_RUN_DIR"
echo "local_dir=$LOCAL_DIR"
echo "interval_sec=$INTERVAL_SEC"

while true; do
  echo
  echo "===== $(date '+%F %T') ====="

  ssh -o BatchMode=yes "$REMOTE" "
    set -e
    echo '[squeue]'
    squeue -j '$JOB_ID' -o '%.18i %.9P %.32j %.8T %.10M %.6D %R' || true
    echo '[latest_structured_loss]'
    if [ -f '$REMOTE_LOG_ERR' ]; then
      tr '\r' '\n' < '$REMOTE_LOG_ERR' | grep -aoE '[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} \[I\] step=[0-9]+ loss=[^)]+' | tail -20 || true
    else
      echo 'missing err log: $REMOTE_LOG_ERR'
    fi
    echo '[latest_tqdm_step]'
    if [ -f '$REMOTE_LOG_ERR' ]; then
      tr '\r' '\n' < '$REMOTE_LOG_ERR' | grep -aoE 'step=[0-9]+' | tail -1 || true
    fi
    echo '[error_scan]'
    if [ -f '$REMOTE_LOG_ERR' ]; then
      tr '\r' '\n' < '$REMOTE_LOG_ERR' | grep -aE 'OutOfMemory|Traceback|nan|inf' | tail -20 || true
    fi
    echo '[visual_files]'
    if [ -d '$REMOTE_RUN_DIR' ]; then
      find '$REMOTE_RUN_DIR' -maxdepth 3 -type f | grep -E 'gen_visualizations|prefix_hidden_visualizations' | sort | tail -20 || true
    else
      echo 'missing run dir: $REMOTE_RUN_DIR'
    fi
  " | tee -a "$LOCAL_DIR/logs/monitor.log"

  scp -q -o BatchMode=yes "$REMOTE:$REMOTE_LOG_ERR" "$LOCAL_DIR/logs/" || true
  scp -q -o BatchMode=yes "$REMOTE:$REMOTE_LOG_OUT" "$LOCAL_DIR/logs/" || true

  if ssh -o BatchMode=yes "$REMOTE" "test -d '$REMOTE_RUN_DIR/gen_visualizations'"; then
    scp -q -r -o BatchMode=yes "$REMOTE:$REMOTE_RUN_DIR/gen_visualizations" "$LOCAL_DIR/" || true
  fi

  if ssh -o BatchMode=yes "$REMOTE" "test -d '$REMOTE_RUN_DIR/prefix_hidden_visualizations'"; then
    scp -q -r -o BatchMode=yes "$REMOTE:$REMOTE_RUN_DIR/prefix_hidden_visualizations" "$LOCAL_DIR/" || true
  fi

  if ! ssh -o BatchMode=yes "$REMOTE" "squeue -h -j '$JOB_ID' | grep -q '$JOB_ID'"; then
    echo "job_not_running $(date '+%F %T')" | tee -a "$LOCAL_DIR/logs/monitor.log"
    break
  fi

  sleep "$INTERVAL_SEC"
done
