#!/usr/bin/env bash
# ============================================================
# GPU 3 — Batch-1 reviewer-response experiments (HALF B)
#   * fixed: curvature_lr=0 (frozen), per-domain      (16 runs)
#   * lvar : α_curv ∈ {2.0} on PACS, 4 splits         ( 4 runs)
# Total: 20 runs.
#
# Output:  experiments/batch1/<family>/<tag>/<ds>_split<n>/
# Logs:    logs/batch1/gpu3/<exp>.log
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
# Force CUDA numbering to match `nvidia-smi` (physical PCI bus order).
# Without this, CUDA's default FASTEST_FIRST reorders devices and
# CUDA_VISIBLE_DEVICES=3 ends up pointing at the Blackwell card (sm_120)
# which is not supported by this PyTorch build.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

LOG_ROOT="${ROOT_DIR}/logs/batch1/gpu3"
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
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" \
      --config "${CONFIG}" \
      --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU3] $(date '+%H:%M:%S')  DONE   ${TAG}"
}

echo "=== fixed (curvature frozen at init): PACS / VLCS / OfficeHome / TerraIncognita ==="
for DS in pacs vlcs officehome terraincognita; do
  for S in 0 1 2 3; do
    run_exp \
      "${ROOT_DIR}/configs/batch1/fixed/${DS}_split${S}.yaml" \
      "fixed_${DS}_split${S}" \
      "experiments/batch1/fixed/frozen/${DS}_split${S}"
  done
done

echo ""
echo "=== lvar sweep on PACS: α_curv ∈ {2.0} ==="
for S in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/batch1/lvar/a2p0_pacs_split${S}.yaml" \
    "lvar_a2p0_pacs_split${S}" \
    "experiments/batch1/lvar/a2p0/pacs_split${S}"
done

echo ""
echo "[GPU3] All assigned runs finished at $(date)"
