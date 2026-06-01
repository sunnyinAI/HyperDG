#!/usr/bin/env bash
# ============================================================
# GPU 0 — Batch-1 reviewer-response experiments (HALF A)
#   * novar: per-domain α with α_curv = 0           (16 runs)
#   * lvar : α_curv ∈ {0.05} on PACS, 4 splits      ( 4 runs)
#   * lvar : α_curv ∈ {0.5}  on PACS, 4 splits      ( 4 runs)
# Total: 24 runs.
#
# Output:  experiments/batch1/<family>/<tag>/<ds>_split<n>/
# Logs:    logs/batch1/gpu0/<exp>.log
#
# Resumable: each run writes its own log; ones already finished
# can be skipped by checking model_last.pt existence.
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
# Force CUDA numbering to match `nvidia-smi` (physical PCI bus order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

LOG_ROOT="${ROOT_DIR}/logs/batch1/gpu0"
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
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" \
      --config "${CONFIG}" \
      --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU0] $(date '+%H:%M:%S')  DONE   ${TAG}"
}

echo "=== novar (per-domain α, α_curv=0): PACS / VLCS / OfficeHome / TerraIncognita ==="
for DS in pacs vlcs officehome terraincognita; do
  for S in 0 1 2 3; do
    run_exp \
      "${ROOT_DIR}/configs/batch1/novar/${DS}_split${S}.yaml" \
      "novar_${DS}_split${S}" \
      "experiments/batch1/novar/a0/${DS}_split${S}"
  done
done

echo ""
echo "=== lvar sweep on PACS: α_curv ∈ {0.05, 0.5} ==="
for A in a0p05 a0p5; do
  for S in 0 1 2 3; do
    run_exp \
      "${ROOT_DIR}/configs/batch1/lvar/${A}_pacs_split${S}.yaml" \
      "lvar_${A}_pacs_split${S}" \
      "experiments/batch1/lvar/${A}/pacs_split${S}"
  done
done

echo ""
echo "[GPU0] All assigned runs finished at $(date)"
