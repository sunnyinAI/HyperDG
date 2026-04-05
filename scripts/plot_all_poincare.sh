#!/usr/bin/env bash
# Generate ALL Poincaré disk plots for reviewer hQnG.
# Safe to run at any time — skips any checkpoint that doesn't exist yet.
# Run again after experiments complete to pick up newly finished runs.
#
# Usage:
#   bash scripts/plot_all_poincare.sh [--device cuda:2]
set -euo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
SCRIPT="${ROOT_DIR}/scripts/plot_poincare_disk.py"
OUT_DIR="${ROOT_DIR}/outputs"
DEVICE="${1:---device}"; DEVICE="${2:-cuda:2}"   # default cuda:2
N=300   # samples per domain

plot() {
  local CONFIG="$1" CKPT="$2" OUT="$3"
  if [[ ! -f "${ROOT_DIR}/${CKPT}" ]]; then
    echo "[SKIP] missing ckpt: ${CKPT}"
    return
  fi
  echo "[PLOT] $(basename ${OUT})"
  "${PYBIN}" "${SCRIPT}" \
    --config "${ROOT_DIR}/${CONFIG}" \
    --ckpt   "${ROOT_DIR}/${CKPT}" \
    --out    "${OUT_DIR}/${OUT}" \
    --n_samples "${N}" \
    --device "${DEVICE}" 2>&1 | tail -1
}

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " HyperDG per-domain α — all datasets (split 0)"
echo "============================================================"
for DS in pacs vlcs officehome domainnet; do
  plot "configs/reviewer_hQnG/single_curv/${DS}_split0.yaml" \
       "experiments/reviewer_hQnG/single_curv/${DS}_split0/model_last.pt" \
       "poincare_hyperdg_${DS}_perdomain.pdf"
done
# TerraIncognita
plot "configs/reviewer_hQnG/single_curv/terraincognita_split0.yaml" \
     "experiments/reviewer_hQnG/single_curv/terraincognita_split0/model_last.pt" \
     "poincare_hyperdg_terra_perdomain.pdf"

echo ""
echo "============================================================"
echo " HyperDG single α — all datasets (split 0)"
echo "============================================================"
for DS in pacs vlcs officehome domainnet; do
  plot "configs/reviewer_hQnG/single_curv/${DS}_split0.yaml" \
       "experiments/reviewer_hQnG/single_curv/${DS}_split0/model_last.pt" \
       "poincare_hyperdg_${DS}_singlecurv.pdf"
done
plot "configs/reviewer_hQnG/single_curv/terraincognita_split0.yaml" \
     "experiments/reviewer_hQnG/single_curv/terraincognita_split0/model_last.pt" \
     "poincare_hyperdg_terra_singlecurv.pdf"

echo ""
echo "============================================================"
echo " MERU per-domain α — all datasets (split 0)"
echo "============================================================"
for DS in pacs vlcs officehome; do
  plot "configs/reviewer_hQnG/meru/${DS}_split0.yaml" \
       "experiments/reviewer_hQnG/meru/${DS}_split0/model_last.pt" \
       "poincare_meru_${DS}_perdomain.pdf"
done
plot "configs/reviewer_hQnG/meru/terraincognita_split0.yaml" \
     "experiments/reviewer_hQnG/meru/terraincognita_split0/model_last.pt" \
     "poincare_meru_terra_perdomain.pdf"
plot "configs/reviewer_hQnG/meru/domainnet_split0.yaml" \
     "experiments/reviewer_hQnG/meru/domainnet_split0/model_last.pt" \
     "poincare_meru_domainnet_perdomain.pdf"

echo ""
echo "============================================================"
echo " MERU single α — all datasets (split 0)"
echo "============================================================"
for DS in pacs vlcs officehome; do
  plot "configs/reviewer_hQnG/meru_single/${DS}_split0.yaml" \
       "experiments/reviewer_hQnG/meru_single/${DS}_split0/model_last.pt" \
       "poincare_meru_${DS}_singlecurv.pdf"
done
plot "configs/reviewer_hQnG/meru_single/terraincognita_split0.yaml" \
     "experiments/reviewer_hQnG/meru_single/terraincognita_split0/model_last.pt" \
     "poincare_meru_terra_singlecurv.pdf"
plot "configs/reviewer_hQnG/meru_single/domainnet_split0.yaml" \
     "experiments/reviewer_hQnG/meru_single/domainnet_split0/model_last.pt" \
     "poincare_meru_domainnet_singlecurv.pdf"

echo ""
echo "============================================================"
echo " Summary of outputs"
echo "============================================================"
ls -1 "${OUT_DIR}"/poincare_*.pdf 2>/dev/null | while read f; do
  echo "  $(basename $f)"
done
echo "Done at $(date)"
