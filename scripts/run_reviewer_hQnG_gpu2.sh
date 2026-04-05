#!/usr/bin/env bash
# ============================================================
# GPU 2 — Reviewer hQnG experiments
# Runs: single_curv (PACS, VLCS, TerraIncognita) + MERU (PACS)
# ============================================================
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=2

LOG_ROOT="${ROOT_DIR}/logs/reviewer_hQnG"
mkdir -p "${LOG_ROOT}"

run_exp() {
  local CONFIG="$1"
  local TAG="$2"
  local LOG="${LOG_ROOT}/${TAG}.log"
  echo "[GPU2] $(date '+%H:%M:%S')  Starting  ${TAG}"
  "${PYBIN}" "${ROOT_DIR}/scripts/train.py" \
      --config "${CONFIG}" \
      --device cuda \
      2>&1 | tee "${LOG}"
  echo "[GPU2] $(date '+%H:%M:%S')  Finished  ${TAG}"
}

# ── Experiment 1a: Single curvature — PACS (4 splits) ─────────────────────
echo "=== Single-curvature ablation: PACS ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/pacs_split${SPLIT}.yaml" \
    "single_curv_pacs_split${SPLIT}"
done

# ── Experiment 1b: Single curvature — VLCS (4 splits) ─────────────────────
echo "=== Single-curvature ablation: VLCS ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/vlcs_split${SPLIT}.yaml" \
    "single_curv_vlcs_split${SPLIT}"
done

# ── Experiment 1c: Single curvature — TerraIncognita (4 splits) ───────────
echo "=== Single-curvature ablation: TerraIncognita ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/single_curv/terraincognita_split${SPLIT}.yaml" \
    "single_curv_terra_split${SPLIT}"
done

# ── Experiment 2a: MERU per-domain-curvature — PACS (4 splits) ────────────
echo "=== MERU per-domain curvature: PACS ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/meru/pacs_split${SPLIT}.yaml" \
    "meru_pacs_split${SPLIT}"
done

# ── Experiment 2b: MERU single-curvature — PACS (4 splits) ────────────────
echo "=== MERU single curvature: PACS ==="
for SPLIT in 0 1 2 3; do
  run_exp \
    "${ROOT_DIR}/configs/reviewer_hQnG/meru_single/pacs_split${SPLIT}.yaml" \
    "meru_single_pacs_split${SPLIT}"
done

echo ""
echo "[GPU2] ALL DONE at $(date)"
