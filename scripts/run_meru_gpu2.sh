#!/usr/bin/env bash
# GPU 2 — MERU experiments: PACS (per-domain + single curvature)
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=2

LOG_ROOT="${ROOT_DIR}/logs/reviewer_hQnG"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1" TAG="$2"
  local LOG="${LOG_ROOT}/${TAG}.log"
  echo "[GPU2] $(date '+%H:%M:%S')  Starting  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
    2>&1 | tee "${LOG}"
  echo "[GPU2] $(date '+%H:%M:%S')  Finished  ${TAG}"
}

echo "=== MERU per-domain: PACS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/reviewer_hQnG/meru/pacs_split${S}.yaml" "meru_pacs_split${S}"
done

echo "=== MERU single-curv: PACS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/reviewer_hQnG/meru_single/pacs_split${S}.yaml" "meru_single_pacs_split${S}"
done

echo "[GPU2] ALL DONE at $(date)"
