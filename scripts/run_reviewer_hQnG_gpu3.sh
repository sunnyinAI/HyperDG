#!/usr/bin/env bash
# ============================================================
# GPU 3 — Reviewer hQnG experiments
# Runs: single_curv (OfficeHome, DomainNet) + MERU (VLCS, OfficeHome)
# ============================================================
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=3

LOG_ROOT="${ROOT_DIR}/logs/reviewer_hQnG"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1"
  local TAG="$2"
  local LOG="${LOG_ROOT}/${TAG}.log"
  echo "[GPU3] $(date '+%H:%M:%S')  Starting  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" \
      --config "${CONFIG}" \
      --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU3] $(date '+%H:%M:%S')  Finished  ${TAG}"
}

# ── Experiment 1d: Single curvature — OfficeHome (4 splits) ───────────────
echo "=== Single-curvature ablation: OfficeHome ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/officehome_split${SPLIT}.yaml" \
    "single_curv_officehome_split${SPLIT}"
done

# ── Experiment 1e: Single curvature — DomainNet (6 splits) ────────────────
echo "=== Single-curvature ablation: DomainNet ==="
for SPLIT in 0 1 2 3 4 5; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/domainnet_split${SPLIT}.yaml" \
    "single_curv_domainnet_split${SPLIT}"
done

# ── Experiment 2c: MERU per-domain-curvature — VLCS (4 splits) ────────────
echo "=== MERU per-domain curvature: VLCS ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/meru/vlcs_split${SPLIT}.yaml" \
    "meru_vlcs_split${SPLIT}"
done

# ── Experiment 2d: MERU single-curvature — VLCS (4 splits) ────────────────
echo "=== MERU single curvature: VLCS ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/meru_single/vlcs_split${SPLIT}.yaml" \
    "meru_single_vlcs_split${SPLIT}"
done

# ── Experiment 2e: MERU per-domain — OfficeHome (4 splits) ────────────────
echo "=== MERU per-domain curvature: OfficeHome ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/meru/officehome_split${SPLIT}.yaml" \
    "meru_officehome_split${SPLIT}"
done

# ── Experiment 2f: MERU single — OfficeHome (4 splits) ────────────────────
echo "=== MERU single curvature: OfficeHome ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/meru_single/officehome_split${SPLIT}.yaml" \
    "meru_single_officehome_split${SPLIT}"
done

echo ""
echo "[GPU3] ALL DONE at $(date)"
