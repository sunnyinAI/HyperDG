#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <backbone:{vitb16|vitl14}> <dataset> <num_splits>"
  exit 1
fi

BACKBONE="$1"
DATASET="$2"
NSPLITS="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/third_party/domainbed"

LOG_DIR="${ROOT_DIR}/logs/${BACKBONE}_${DATASET}"
mkdir -p "${LOG_DIR}"

for ((split=0; split<NSPLITS; ++split)); do
  gpu=${GPUS[$((split % ${#GPUS[@]}))]}
  config="${ROOT_DIR}/experiments/configs/${BACKBONE}_hyperdg_${DATASET}_split${split}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config ${config}" >&2
    exit 1
  fi
  log_file="${LOG_DIR}/split${split}.log"
  echo "[$(date)] ${BACKBONE}/${DATASET} split ${split} -> GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} \
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train.py" \
      --config "${config}" \
      --device cuda > "${log_file}" 2>&1 &
done

wait
echo "[$(date)] Completed ${BACKBONE}/${DATASET}"
