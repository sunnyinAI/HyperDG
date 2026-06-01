#!/usr/bin/env bash
# ============================================================
# GPU 3 — Batch-3 baseline trajectory runs (THIRD C, 4 runs)
#   * baseline (alpha_curv=0.2) : TerraIncognita 0-3  (4)
#
# Waits for Batch-2 GPU 3 master log to report completion before starting.
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

LOG_ROOT="${ROOT_DIR}/logs/batch3/gpu3"
mkdir -p "${LOG_ROOT}"

BATCH2_MASTER="${ROOT_DIR}/logs/batch2/gpu3/MASTER.log"
echo "[GPU3] waiting for Batch-2 GPU 3 to finish..."
while ! grep -q "All assigned runs finished" "${BATCH2_MASTER}" 2>/dev/null; do
  sleep 60
done
echo "[GPU3] Batch-2 GPU 3 finished. Starting Batch-3."

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

echo "=== baseline: TerraIncognita ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch3/baseline/terraincognita_split${S}.yaml" \
          "baseline_terraincognita_split${S}" \
          "experiments/batch3/baseline/terraincognita_split${S}"
done

echo "[GPU3] All assigned runs finished at $(date)"
