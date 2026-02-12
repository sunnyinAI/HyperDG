#!/usr/bin/env bash
set -euo pipefail
DATASET="$1"
NSPLITS="$2"
GPUS=(0 1 2 3)
ROOT_DIR="$(pwd)"
LOG_DIR="${ROOT_DIR}/logs/${DATASET}_hyperdg"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/third_party/domainbed"
for split in $(seq 0 $((NSPLITS-1))); do
  gpu=${GPUS[$((split % ${#GPUS[@]}))]}
  LOG="${LOG_DIR}/split${split}.log"
  echo "[$(date)] ${DATASET} split ${split} -> GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} \
    python3 scripts/train.py \
      --config "experiments/configs/hyperdg_${DATASET}_split${split}.yaml" \
      --device cuda >"${LOG}" 2>&1 &
done
wait
