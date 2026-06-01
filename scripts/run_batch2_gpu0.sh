#!/usr/bin/env bash
# ============================================================
# GPU 0 — Batch-2 leave-one-out (THIRD A, 14 runs)
#   * erm        : PACS                          (4)
#   * no_align   : PACS                          (4)
#   * no_sharp   : PACS                          (4)
#   * no_flat    : PACS  splits 0-1              (2)
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

LOG_ROOT="${ROOT_DIR}/logs/batch2/gpu0"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1"
  local TAG="$2"
  local OUTPUT_DIR="$3"
  local LOG="${LOG_ROOT}/${TAG}.log"
  if [[ -f "${ROOT_DIR}/${OUTPUT_DIR}/model_last.pt" ]]; then
    echo "[GPU0] $(date '+%H:%M:%S')  SKIP (done) ${TAG}"
    return
  fi
  echo "[GPU0] $(date '+%H:%M:%S')  START  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU0] $(date '+%H:%M:%S')  DONE   ${TAG}"
}

echo "=== erm: PACS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/erm/pacs_split${S}.yaml" \
          "erm_pacs_split${S}" "experiments/batch2/erm/pacs_split${S}"
done

echo "=== no_align: PACS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/no_align/pacs_split${S}.yaml" \
          "no_align_pacs_split${S}" "experiments/batch2/no_align/pacs_split${S}"
done

echo "=== no_sharp: PACS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/no_sharp/pacs_split${S}.yaml" \
          "no_sharp_pacs_split${S}" "experiments/batch2/no_sharp/pacs_split${S}"
done

echo "=== no_flat: PACS splits 0-1 ==="
for S in 0 1; do
  run_exp "${ROOT_DIR}/configs/batch2/no_flat/pacs_split${S}.yaml" \
          "no_flat_pacs_split${S}" "experiments/batch2/no_flat/pacs_split${S}"
done

echo "[GPU0] All assigned runs finished at $(date)"
