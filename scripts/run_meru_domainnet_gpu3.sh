#!/usr/bin/env bash
# GPU 3 — MERU experiments: DomainNet single curvature (6 splits)
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=3

LOG_ROOT="${ROOT_DIR}/logs/reviewer_hQnG"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1" TAG="$2"
  local LOG="${LOG_ROOT}/${TAG}.log"
  echo "[GPU3] $(date '+%H:%M:%S')  Starting  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
    2>&1 | tee "${LOG}"
  echo "[GPU3] $(date '+%H:%M:%S')  Finished  ${TAG}"
}

echo "=== MERU single-curv: DomainNet ==="
for S in 0 1 2 3 4 5; do
  run_exp "${ROOT_DIR}/configs/reviewer_hQnG/meru_single/domainnet_split${S}.yaml" \
    "meru_single_domainnet_split${S}"
done

echo "[GPU3] ALL DONE at $(date)"
