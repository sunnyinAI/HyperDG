#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <dataset>"
  echo "dataset ∈ {vlcs, pacs}"
  exit 1
fi

DATASET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/sftdg/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(which python)"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found. Set PYTHON_BIN to the desired python path."
  exit 1
fi

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/third_party/domainbed"

LOG_DIR="${ROOT_DIR}/logs/${DATASET}"
mkdir -p "${LOG_DIR}"

for split in 0 1 2 3; do
  CONFIG="${ROOT_DIR}/experiments/configs/${DATASET}_split${split}.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config ${CONFIG}" >&2
    exit 1
  fi

  LOG_FILE="${LOG_DIR}/split${split}.log"
  echo "[$(date)] Running ${DATASET} split ${split} -> ${LOG_FILE}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda 2>&1 | tee "${LOG_FILE}"
done
