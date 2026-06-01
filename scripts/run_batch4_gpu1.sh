#!/usr/bin/env bash
# ============================================================
# GPU 1 — Batch-4 multi-seed (auto-generated)
#   64 runs (every 3-th config starting at index 1).
#   Each run: ~25 min  →  total ~1600 min  ~26.7 h.
# ============================================================
set -uo pipefail

PYBIN="${HOME}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/domainbed:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

LOG_ROOT="${ROOT_DIR}/logs/batch4/gpu1"
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

run_exp "${ROOT_DIR}/configs/batch4/seed1/baseline/officehome_split1.yaml" \
        "seed1_baseline_officehome_split1" \
        "experiments/batch4/seed1/baseline/officehome_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed1/baseline/pacs_split0.yaml" \
        "seed1_baseline_pacs_split0" \
        "experiments/batch4/seed1/baseline/pacs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed1/baseline/pacs_split3.yaml" \
        "seed1_baseline_pacs_split3" \
        "experiments/batch4/seed1/baseline/pacs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed1/baseline/terraincognita_split2.yaml" \
        "seed1_baseline_terraincognita_split2" \
        "experiments/batch4/seed1/baseline/terraincognita_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed1/baseline/vlcs_split1.yaml" \
        "seed1_baseline_vlcs_split1" \
        "experiments/batch4/seed1/baseline/vlcs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed1/erm/officehome_split0.yaml" \
        "seed1_erm_officehome_split0" \
        "experiments/batch4/seed1/erm/officehome_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed1/erm/officehome_split3.yaml" \
        "seed1_erm_officehome_split3" \
        "experiments/batch4/seed1/erm/officehome_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed1/erm/pacs_split2.yaml" \
        "seed1_erm_pacs_split2" \
        "experiments/batch4/seed1/erm/pacs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed1/erm/terraincognita_split1.yaml" \
        "seed1_erm_terraincognita_split1" \
        "experiments/batch4/seed1/erm/terraincognita_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed1/erm/vlcs_split0.yaml" \
        "seed1_erm_vlcs_split0" \
        "experiments/batch4/seed1/erm/vlcs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed1/erm/vlcs_split3.yaml" \
        "seed1_erm_vlcs_split3" \
        "experiments/batch4/seed1/erm/vlcs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed1/fixed/officehome_split2.yaml" \
        "seed1_fixed_officehome_split2" \
        "experiments/batch4/seed1/fixed/officehome_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed1/fixed/pacs_split1.yaml" \
        "seed1_fixed_pacs_split1" \
        "experiments/batch4/seed1/fixed/pacs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed1/fixed/terraincognita_split0.yaml" \
        "seed1_fixed_terraincognita_split0" \
        "experiments/batch4/seed1/fixed/terraincognita_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed1/fixed/terraincognita_split3.yaml" \
        "seed1_fixed_terraincognita_split3" \
        "experiments/batch4/seed1/fixed/terraincognita_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed1/fixed/vlcs_split2.yaml" \
        "seed1_fixed_vlcs_split2" \
        "experiments/batch4/seed1/fixed/vlcs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed1/novar/officehome_split1.yaml" \
        "seed1_novar_officehome_split1" \
        "experiments/batch4/seed1/novar/officehome_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed1/novar/pacs_split0.yaml" \
        "seed1_novar_pacs_split0" \
        "experiments/batch4/seed1/novar/pacs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed1/novar/pacs_split3.yaml" \
        "seed1_novar_pacs_split3" \
        "experiments/batch4/seed1/novar/pacs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed1/novar/terraincognita_split2.yaml" \
        "seed1_novar_terraincognita_split2" \
        "experiments/batch4/seed1/novar/terraincognita_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed1/novar/vlcs_split1.yaml" \
        "seed1_novar_vlcs_split1" \
        "experiments/batch4/seed1/novar/vlcs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed2/baseline/officehome_split0.yaml" \
        "seed2_baseline_officehome_split0" \
        "experiments/batch4/seed2/baseline/officehome_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed2/baseline/officehome_split3.yaml" \
        "seed2_baseline_officehome_split3" \
        "experiments/batch4/seed2/baseline/officehome_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed2/baseline/pacs_split2.yaml" \
        "seed2_baseline_pacs_split2" \
        "experiments/batch4/seed2/baseline/pacs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed2/baseline/terraincognita_split1.yaml" \
        "seed2_baseline_terraincognita_split1" \
        "experiments/batch4/seed2/baseline/terraincognita_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed2/baseline/vlcs_split0.yaml" \
        "seed2_baseline_vlcs_split0" \
        "experiments/batch4/seed2/baseline/vlcs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed2/baseline/vlcs_split3.yaml" \
        "seed2_baseline_vlcs_split3" \
        "experiments/batch4/seed2/baseline/vlcs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed2/erm/officehome_split2.yaml" \
        "seed2_erm_officehome_split2" \
        "experiments/batch4/seed2/erm/officehome_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed2/erm/pacs_split1.yaml" \
        "seed2_erm_pacs_split1" \
        "experiments/batch4/seed2/erm/pacs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed2/erm/terraincognita_split0.yaml" \
        "seed2_erm_terraincognita_split0" \
        "experiments/batch4/seed2/erm/terraincognita_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed2/erm/terraincognita_split3.yaml" \
        "seed2_erm_terraincognita_split3" \
        "experiments/batch4/seed2/erm/terraincognita_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed2/erm/vlcs_split2.yaml" \
        "seed2_erm_vlcs_split2" \
        "experiments/batch4/seed2/erm/vlcs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed2/fixed/officehome_split1.yaml" \
        "seed2_fixed_officehome_split1" \
        "experiments/batch4/seed2/fixed/officehome_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed2/fixed/pacs_split0.yaml" \
        "seed2_fixed_pacs_split0" \
        "experiments/batch4/seed2/fixed/pacs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed2/fixed/pacs_split3.yaml" \
        "seed2_fixed_pacs_split3" \
        "experiments/batch4/seed2/fixed/pacs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed2/fixed/terraincognita_split2.yaml" \
        "seed2_fixed_terraincognita_split2" \
        "experiments/batch4/seed2/fixed/terraincognita_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed2/fixed/vlcs_split1.yaml" \
        "seed2_fixed_vlcs_split1" \
        "experiments/batch4/seed2/fixed/vlcs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed2/novar/officehome_split0.yaml" \
        "seed2_novar_officehome_split0" \
        "experiments/batch4/seed2/novar/officehome_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed2/novar/officehome_split3.yaml" \
        "seed2_novar_officehome_split3" \
        "experiments/batch4/seed2/novar/officehome_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed2/novar/pacs_split2.yaml" \
        "seed2_novar_pacs_split2" \
        "experiments/batch4/seed2/novar/pacs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed2/novar/terraincognita_split1.yaml" \
        "seed2_novar_terraincognita_split1" \
        "experiments/batch4/seed2/novar/terraincognita_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed2/novar/vlcs_split0.yaml" \
        "seed2_novar_vlcs_split0" \
        "experiments/batch4/seed2/novar/vlcs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed2/novar/vlcs_split3.yaml" \
        "seed2_novar_vlcs_split3" \
        "experiments/batch4/seed2/novar/vlcs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed3/baseline/officehome_split2.yaml" \
        "seed3_baseline_officehome_split2" \
        "experiments/batch4/seed3/baseline/officehome_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed3/baseline/pacs_split1.yaml" \
        "seed3_baseline_pacs_split1" \
        "experiments/batch4/seed3/baseline/pacs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed3/baseline/terraincognita_split0.yaml" \
        "seed3_baseline_terraincognita_split0" \
        "experiments/batch4/seed3/baseline/terraincognita_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed3/baseline/terraincognita_split3.yaml" \
        "seed3_baseline_terraincognita_split3" \
        "experiments/batch4/seed3/baseline/terraincognita_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed3/baseline/vlcs_split2.yaml" \
        "seed3_baseline_vlcs_split2" \
        "experiments/batch4/seed3/baseline/vlcs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed3/erm/officehome_split1.yaml" \
        "seed3_erm_officehome_split1" \
        "experiments/batch4/seed3/erm/officehome_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed3/erm/pacs_split0.yaml" \
        "seed3_erm_pacs_split0" \
        "experiments/batch4/seed3/erm/pacs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed3/erm/pacs_split3.yaml" \
        "seed3_erm_pacs_split3" \
        "experiments/batch4/seed3/erm/pacs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed3/erm/terraincognita_split2.yaml" \
        "seed3_erm_terraincognita_split2" \
        "experiments/batch4/seed3/erm/terraincognita_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed3/erm/vlcs_split1.yaml" \
        "seed3_erm_vlcs_split1" \
        "experiments/batch4/seed3/erm/vlcs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed3/fixed/officehome_split0.yaml" \
        "seed3_fixed_officehome_split0" \
        "experiments/batch4/seed3/fixed/officehome_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed3/fixed/officehome_split3.yaml" \
        "seed3_fixed_officehome_split3" \
        "experiments/batch4/seed3/fixed/officehome_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed3/fixed/pacs_split2.yaml" \
        "seed3_fixed_pacs_split2" \
        "experiments/batch4/seed3/fixed/pacs_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed3/fixed/terraincognita_split1.yaml" \
        "seed3_fixed_terraincognita_split1" \
        "experiments/batch4/seed3/fixed/terraincognita_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed3/fixed/vlcs_split0.yaml" \
        "seed3_fixed_vlcs_split0" \
        "experiments/batch4/seed3/fixed/vlcs_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed3/fixed/vlcs_split3.yaml" \
        "seed3_fixed_vlcs_split3" \
        "experiments/batch4/seed3/fixed/vlcs_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed3/novar/officehome_split2.yaml" \
        "seed3_novar_officehome_split2" \
        "experiments/batch4/seed3/novar/officehome_split2"
run_exp "${ROOT_DIR}/configs/batch4/seed3/novar/pacs_split1.yaml" \
        "seed3_novar_pacs_split1" \
        "experiments/batch4/seed3/novar/pacs_split1"
run_exp "${ROOT_DIR}/configs/batch4/seed3/novar/terraincognita_split0.yaml" \
        "seed3_novar_terraincognita_split0" \
        "experiments/batch4/seed3/novar/terraincognita_split0"
run_exp "${ROOT_DIR}/configs/batch4/seed3/novar/terraincognita_split3.yaml" \
        "seed3_novar_terraincognita_split3" \
        "experiments/batch4/seed3/novar/terraincognita_split3"
run_exp "${ROOT_DIR}/configs/batch4/seed3/novar/vlcs_split2.yaml" \
        "seed3_novar_vlcs_split2" \
        "experiments/batch4/seed3/novar/vlcs_split2"
echo "[GPU1] All assigned runs finished at $(date)"
