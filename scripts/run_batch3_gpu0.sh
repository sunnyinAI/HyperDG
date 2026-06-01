#!/usr/bin/env bash
# ============================================================
# GPU 0 — Batch-3 baseline trajectory runs (THIRD A, 6 runs)
#   * baseline (alpha_curv=0.2) : PACS 0-3            (4)
#   * baseline (alpha_curv=0.2) : VLCS 0-1            (2)
#
# This script *waits* until any prior Batch-2 job on GPU 0 has finished
# (detected via the master log) before kicking off, so it can be queued
# in parallel without colliding.
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

LOG_ROOT="${ROOT_DIR}/logs/batch3/gpu0"
mkdir -p "${LOG_ROOT}"

# ── Wait for Batch-2 GPU 0 to be done ─────────────────────────────────
BATCH2_MASTER="${ROOT_DIR}/logs/batch2/gpu0/MASTER.log"
echo "[GPU0] waiting for Batch-2 GPU 0 to finish (looking for 'All assigned runs finished')..."
while ! grep -q "All assigned runs finished" "${BATCH2_MASTER}" 2>/dev/null; do
  sleep 60
done
echo "[GPU0] Batch-2 GPU 0 finished. Starting Batch-3."

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

echo "=== baseline: PACS ==="
for S in 0 1 2 3; do
  run_exp "${ROOT_DIR}/configs/batch3/baseline/pacs_split${S}.yaml" \
          "baseline_pacs_split${S}" \
          "experiments/batch3/baseline/pacs_split${S}"
done

echo "=== baseline: VLCS splits 0-1 ==="
for S in 0 1; do
  run_exp "${ROOT_DIR}/configs/batch3/baseline/vlcs_split${S}.yaml" \
          "baseline_vlcs_split${S}" \
          "experiments/batch3/baseline/vlcs_split${S}"
done

echo "[GPU0] All assigned runs finished at $(date)"
