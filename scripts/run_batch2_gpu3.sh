#!/usr/bin/env bash
# ============================================================
# GPU 3 — Batch-2 leave-one-out (THIRD C, 13 runs)
#   * erm        : OfficeHome                    (4)
#   * no_align   : OfficeHome                    (4)
#   * no_flat    : OfficeHome                    (4)
#   * no_sharp   : OfficeHome split 3            (1)
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

LOG_ROOT="${ROOT_DIR}/logs/batch2/gpu3"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1"
  local TAG="$2"
  local OUTPUT_DIR="$3"
  local LOG="${LOG_ROOT}/${TAG}.log"
  if [[ -f "${ROOT_DIR}/${OUTPUT_DIR}/model_last.pt" ]]; then
    echo "[GPU3] $(date '+%H:%M:%S')  SKIP (done) ${TAG}"
    return
  fi
  echo "[GPU3] $(date '+%H:%M:%S')  START  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU3] $(date '+%H:%M:%S')  DONE   ${TAG}"
}

echo "=== erm: OfficeHome ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/erm/officehome_split${S}.yaml" \
          "erm_officehome_split${S}" "experiments/batch2/erm/officehome_split${S}"
done

echo "=== no_align: OfficeHome ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/no_align/officehome_split${S}.yaml" \
          "no_align_officehome_split${S}" "experiments/batch2/no_align/officehome_split${S}"
done

echo "=== no_flat: OfficeHome ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/no_flat/officehome_split${S}.yaml" \
          "no_flat_officehome_split${S}" "experiments/batch2/no_flat/officehome_split${S}"
done

echo "=== no_sharp: OfficeHome split 3 ==="
run_exp "${ROOT_DIR}/configs/batch2/no_sharp/officehome_split3.yaml" \
        "no_sharp_officehome_split3" "experiments/batch2/no_sharp/officehome_split3"

echo "[GPU3] All assigned runs finished at $(date)"
