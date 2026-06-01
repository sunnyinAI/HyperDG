#!/usr/bin/env bash
# TerraIncognita single-curvature ablation — runs on whichever GPU is free
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_terra_single_curv.sh
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"

LOG_ROOT="${ROOT_DIR}/logs/reviewer_hQnG"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1" TAG="$2"
  local LOG="${LOG_ROOT}/${TAG}.log"
  echo "[TERRA] $(date '+%H:%M:%S')  Starting  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
    2>&1 | tee "${LOG}"
  echo "[TERRA] $(date '+%H:%M:%S')  Finished  ${TAG}"
}

echo "=== Single-curvature ablation: TerraIncognita ==="
for S in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/terraincognita_split${S}.yaml" \
    "single_curv_terra_split${S}"
done

echo "[TERRA] ALL DONE at $(date)"
