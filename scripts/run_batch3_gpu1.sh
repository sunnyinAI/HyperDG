#!/usr/bin/env bash
# ============================================================
# GPU 1 — Batch-3 baseline trajectory runs (THIRD B, 6 runs)
#   * baseline (alpha_curv=0.2) : VLCS 2-3            (2)
#   * baseline (alpha_curv=0.2) : OfficeHome 0-3      (4)
#
# Waits for Batch-2 GPU 1 master log to report completion before starting.
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

LOG_ROOT="${ROOT_DIR}/logs/batch3/gpu1"
mkdir -p "${LOG_ROOT}"

BATCH2_MASTER="${ROOT_DIR}/logs/batch2/gpu1/MASTER.log"
echo "[GPU1] waiting for Batch-2 GPU 1 to finish..."
while ! grep -q "All assigned runs finished" "${BATCH2_MASTER}" 2>/dev/null; do
  sleep 60
done
echo "[GPU1] Batch-2 GPU 1 finished. Starting Batch-3."

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

echo "=== baseline: VLCS splits 2-3 ==="
for S in 2 3; do
  run_exp "${ROOT_DIR}/configs/batch3/baseline/vlcs_split${S}.yaml" \
          "baseline_vlcs_split${S}" \
          "experiments/batch3/baseline/vlcs_split${S}"
done

echo "=== baseline: OfficeHome ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch3/baseline/officehome_split${S}.yaml" \
          "baseline_officehome_split${S}" \
          "experiments/batch3/baseline/officehome_split${S}"
done

echo "[GPU1] All assigned runs finished at $(date)"
