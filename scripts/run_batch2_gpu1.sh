#!/usr/bin/env bash
# ============================================================
# GPU 1 — Batch-2 leave-one-out (THIRD B, 13 runs)
#   * erm        : VLCS                          (4)
#   * erm        : TerraIncognita                (4)
#   * no_flat    : PACS splits 2-3               (2)
#   * no_sharp   : OfficeHome splits 0-2         (3)
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

LOG_ROOT="${ROOT_DIR}/logs/batch2/gpu1"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1"
  local TAG="$2"
  local OUTPUT_DIR="$3"
  local LOG="${LOG_ROOT}/${TAG}.log"
  if [[ -f "${ROOT_DIR}/${OUTPUT_DIR}/model_last.pt" ]]; then
    echo "[GPU1] $(date '+%H:%M:%S')  SKIP (done) ${TAG}"
    return
  fi
  echo "[GPU1] $(date '+%H:%M:%S')  START  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" --config "${CONFIG}" --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU1] $(date '+%H:%M:%S')  DONE   ${TAG}"
}

echo "=== erm: VLCS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/erm/vlcs_split${S}.yaml" \
          "erm_vlcs_split${S}" "experiments/batch2/erm/vlcs_split${S}"
done

echo "=== erm: TerraIncognita ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/erm/terraincognita_split${S}.yaml" \
          "erm_terraincognita_split${S}" "experiments/batch2/erm/terraincognita_split${S}"
done

echo "=== no_flat: PACS splits 2-3 ==="
for S in 2 3; do
  run_exp "${ROOT_DIR}/configs/batch2/no_flat/pacs_split${S}.yaml" \
          "no_flat_pacs_split${S}" "experiments/batch2/no_flat/pacs_split${S}"
done

echo "=== no_sharp: OfficeHome splits 0-2 ==="
for S in 0 1 2; do
  run_exp "${ROOT_DIR}/configs/batch2/no_sharp/officehome_split${S}.yaml" \
          "no_sharp_officehome_split${S}" "experiments/batch2/no_sharp/officehome_split${S}"
done

echo "[GPU1] All assigned runs finished at $(date)"
