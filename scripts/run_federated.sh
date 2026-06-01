#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset|all>" >&2
  exit 1
fi

TARGET="$1"
ROOT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASETS=()
if [ "${TARGET}" = "all" ]; then
  DATASETS=(vlcs pacs officehome terraincognita domainnet)
else
  DATASETS=("${TARGET}")
fi

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/third_party/domainbed"

for dataset in "${DATASETS[@]}"; do
  CONFIG="${ROOT_DIR}/configs/federated/resnet50_hyperdg_fed_${dataset}.yaml"
  if [ ! -f "${CONFIG}" ]; then
    echo "Missing config ${CONFIG}" >&2
    exit 1
  fi
  LOG_DIR="${ROOT_DIR}/logs/federated_${dataset}"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/run.log"
  echo "[$(date '+%F %T')] Running federated ${dataset} -> ${LOG_FILE}"
  ${PYTHON_BIN} scripts/train_federated.py --config "${CONFIG}" --device cuda >"${LOG_FILE}" 2>&1
done
