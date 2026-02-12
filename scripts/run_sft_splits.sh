#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <dataset>

Environment variables:
  SFT_SPLITS    Number of splits to launch (default: 4)
  SFT_GPUS      Comma-separated GPU ids to round-robin across (default: 0,1,2,3)
  PYTHON_BIN    Python interpreter (default: python3)

Example:
  SFT_SPLITS=4 SFT_GPUS=0,1 PYTHON_BIN=$(which python3) $0 vlcs
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

DATASET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${ROOT_DIR}/experiments/configs"
SPLITS="${SFT_SPLITS:-4}"
GPU_STR="${SFT_GPUS:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

IFS=',' read -ra GPU_LIST <<<"${GPU_STR}"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs provided via SFT_GPUS" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/third_party/domainbed"

LOG_DIR="${ROOT_DIR}/logs/sft_${DATASET}"
mkdir -p "${LOG_DIR}"

pids=()

for ((split=0; split<${SPLITS}; ++split)); do
  CONFIG="${CONFIG_DIR}/sft_${DATASET}_split${split}.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config ${CONFIG}. Run scripts/create_sft_split_configs.py first." >&2
    exit 1
  fi
  GPU="${GPU_LIST[$((split % ${#GPU_LIST[@]}))]}"
  LOG_FILE="${LOG_DIR}/split${split}.log"
  echo "[$(date '+%F %T')] Launching ${DATASET} split ${split} on GPU ${GPU} -> ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train.py" \
    --config "${CONFIG}" \
    --device cuda >"${LOG_FILE}" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit ${status}
