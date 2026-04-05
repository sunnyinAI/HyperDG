#!/usr/bin/env bash
# GPU 1 — MERU experiments: TerraIncognita (per-domain + single curvature)
# Also picks up single_curv Terra splits 2 & 3 first (fast ResNet runs)
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1

LOG_ROOT="${ROOT_DIR}/logs/reviewer_hQnG"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1" TAG="$2"
  local LOG="${LOG_ROOT}/${TAG}.log"
  echo "[GPU1] $(date '+%H:%M:%S')  Starting  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
    2>&1 | tee "${LOG}"
  echo "[GPU1] $(date '+%H:%M:%S')  Finished  ${TAG}"
}

echo "=== single_curv Terra: missing splits 2 & 3 ==="
for S in 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/terraincognita_split${S}.yaml" \
    "single_curv_terra_split${S}"
done

echo "=== MERU per-domain: TerraIncognita ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/reviewer_hQnG/meru/terraincognita_split${S}.yaml" \
    "meru_terra_split${S}"
done

echo "=== MERU single-curv: TerraIncognita ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/reviewer_hQnG/meru_single/terraincognita_split${S}.yaml" \
    "meru_single_terra_split${S}"
done

echo "[GPU1] ALL DONE at $(date)"
