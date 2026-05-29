#!/usr/bin/env bash
set -euo pipefail

LOCAL_FASTWAM="${LOCAL_FASTWAM:-external/FastWAM}"
REMOTE_HOST="${REMOTE_HOST:-HPC3_jhe724}"
REMOTE_FASTWAM="${REMOTE_FASTWAM:-/data/user/jhe724/workspace/FastWAM}"

if [[ ! -d "${LOCAL_FASTWAM}/src" ]]; then
  echo "FastWAM source not found: ${LOCAL_FASTWAM}" >&2
  exit 1
fi

rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.log' \
  "${LOCAL_FASTWAM}/src" \
  "${LOCAL_FASTWAM}/configs" \
  "${LOCAL_FASTWAM}/experiments" \
  "${LOCAL_FASTWAM}/scripts" \
  "${LOCAL_FASTWAM}/docs" \
  "${LOCAL_FASTWAM}/pyproject.toml" \
  "${LOCAL_FASTWAM}/README.md" \
  "${LOCAL_FASTWAM}/README_zh.md" \
  "${LOCAL_FASTWAM}/.gitignore" \
  "${REMOTE_HOST}:${REMOTE_FASTWAM}/"

echo "Synced FastWAM code to ${REMOTE_HOST}:${REMOTE_FASTWAM}"
