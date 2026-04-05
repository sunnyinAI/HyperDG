#!/usr/bin/env python3
"""Generate experiment configs for Reviewer hQnG's requested experiments.

Creates YAML configs under configs/reviewer_hQnG/ for:
  1. single_curv/  -- per-domain α → single shared α ablation (all datasets × splits)
  2. meru/         -- MERU ViT-B/16 backbone (per-domain α and single α)
  3. meru_single/  -- MERU + single α (answers hQnG question 2b)

Usage:
  python scripts/generate_reviewer_configs.py
"""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "configs" / "reviewer_hQnG"
DATASET_CONFIG_DIR = ROOT / "configs" / "datasets"

# ── Dataset specs ─────────────────────────────────────────────────────────────
DATASETS = {
    "pacs":           {"num_splits": 4, "data_dir": "data",
                       "n_steps": 5000, "batch_size": 32},
    "vlcs":           {"num_splits": 4, "data_dir": "data",
                       "n_steps": 5000, "batch_size": 32},
    "officehome":     {"num_splits": 4, "data_dir": "data/officehome_clean",
                       "n_steps": 5000, "batch_size": 32},
    "terraincognita": {"num_splits": 4, "data_dir": "data",
                       "n_steps": 5000, "batch_size": 32},
    "domainnet":      {"num_splits": 6, "data_dir": "data",
                       "n_steps": 10000, "batch_size": 32},
}

# Datasets for MERU experiments (computationally lighter subset)
MERU_DATASETS = ["pacs", "vlcs", "officehome"]

# ── Base HyperDG hparams ──────────────────────────────────────────────────────
BASE_HPARAMS = {
    "lr": 5.0e-5,
    "curvature_lr": 1.0e-4,
    "weight_decay": 0.0,
    "grad_clip_norm": 5.0,
    "nonlinear_classifier": False,
    "checkpoint_freq": 200,
    "alpha_curv": 0.2,
    "beta_align": 0.5,
    "gamma_sharp": 0.05,
    "delta_local_flat": 0.1,
    "delta_global_flat": 0.1,
    "curvature_epsilon": 1.0e-4,
}

# MERU-specific additions
MERU_HPARAMS = {
    "backbone": "meru_vit_b16",
    "meru_variant": "vit_b16",
    "meru_pretrained": True,
    "meru_freeze_backbone": True,
    "meru_output": "euclidean",
    "lr": 1.0e-4,          # higher LR since backbone is frozen
    "curvature_lr": 1.0e-4,
    "batch_size": 32,
    "lambda_contrast": 0.1,
    "contrast_temp": 0.07,
}


def dataset_cfg_name(dataset_key: str) -> str:
    """Map dataset key to its YAML config filename."""
    mapping = {
        "pacs": "pacs",
        "vlcs": "vlcs",
        "officehome": "officehome",
        "terraincognita": "terraincognita",
        "domainnet": "domainnet",
    }
    return mapping[dataset_key]


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote {path.relative_to(ROOT)}")


# ── 1. Single curvature ablation configs ─────────────────────────────────────

def gen_single_curv_configs():
    out_dir = CONFIG_ROOT / "single_curv"
    for ds_key, ds_spec in DATASETS.items():
        for split in range(ds_spec["num_splits"]):
            cfg_name = dataset_cfg_name(ds_key)
            exp_name = f"hyperdg_single_curv_{ds_key}_split{split}"

            hparams = dict(BASE_HPARAMS)
            hparams["n_steps"] = ds_spec["n_steps"]
            hparams["batch_size"] = ds_spec["batch_size"]
            hparams["single_curvature"] = True   # KEY flag

            config = {
                "experiment_name": exp_name,
                "seed": 0,
                "output_dir": f"experiments/reviewer_hQnG/single_curv/{ds_key}_split{split}",
                "device": "cuda",
                "eval_on_start": False,
                "dataset": {
                    "config": f"configs/datasets/{cfg_name}.yaml",
                    "data_dir": ds_spec["data_dir"],
                    "test_envs": [split],
                },
                "algorithm": "hyperdg",
                "hparams": hparams,
            }
            write_yaml(out_dir / f"{ds_key}_split{split}.yaml", config)


# ── 2. MERU per-domain-curvature configs ─────────────────────────────────────

def gen_meru_configs():
    out_dir = CONFIG_ROOT / "meru"
    for ds_key in MERU_DATASETS:
        ds_spec = DATASETS[ds_key]
        for split in range(ds_spec["num_splits"]):
            cfg_name = dataset_cfg_name(ds_key)
            exp_name = f"hyperdg_meru_{ds_key}_split{split}"

            hparams = dict(BASE_HPARAMS)
            hparams.update(MERU_HPARAMS)
            hparams["n_steps"] = ds_spec["n_steps"]
            hparams["single_curvature"] = False

            config = {
                "experiment_name": exp_name,
                "seed": 0,
                "output_dir": f"experiments/reviewer_hQnG/meru/{ds_key}_split{split}",
                "device": "cuda",
                "eval_on_start": False,
                "dataset": {
                    "config": f"configs/datasets/{cfg_name}.yaml",
                    "data_dir": ds_spec["data_dir"],
                    "test_envs": [split],
                },
                "algorithm": "hyperdg_meru",
                "hparams": hparams,
            }
            write_yaml(out_dir / f"{ds_key}_split{split}.yaml", config)


# ── 3. MERU + single curvature configs ───────────────────────────────────────

def gen_meru_single_configs():
    out_dir = CONFIG_ROOT / "meru_single"
    for ds_key in MERU_DATASETS:
        ds_spec = DATASETS[ds_key]
        for split in range(ds_spec["num_splits"]):
            cfg_name = dataset_cfg_name(ds_key)
            exp_name = f"hyperdg_meru_single_{ds_key}_split{split}"

            hparams = dict(BASE_HPARAMS)
            hparams.update(MERU_HPARAMS)
            hparams["n_steps"] = ds_spec["n_steps"]
            hparams["single_curvature"] = True

            config = {
                "experiment_name": exp_name,
                "seed": 0,
                "output_dir": f"experiments/reviewer_hQnG/meru_single/{ds_key}_split{split}",
                "device": "cuda",
                "eval_on_start": False,
                "dataset": {
                    "config": f"configs/datasets/{cfg_name}.yaml",
                    "data_dir": ds_spec["data_dir"],
                    "test_envs": [split],
                },
                "algorithm": "hyperdg_meru",
                "hparams": hparams,
            }
            write_yaml(out_dir / f"{ds_key}_split{split}.yaml", config)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    total_single = sum(s["num_splits"] for s in DATASETS.values())
    total_meru = sum(DATASETS[d]["num_splits"] for d in MERU_DATASETS) * 2
    print(f"\nSummary:")
    print(f"  Single-curv ablation configs : {total_single}  (all 5 datasets × splits)")
    print(f"  MERU configs (per-domain + single): {total_meru}  ({', '.join(MERU_DATASETS)})")
    print(f"  Total configs: {total_single + total_meru}")
    print(f"  Output dir: {CONFIG_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    print("Generating single_curvature ablation configs …")
    gen_single_curv_configs()

    print("\nGenerating MERU (per-domain curvature) configs …")
    gen_meru_configs()

    print("\nGenerating MERU (single curvature) configs …")
    gen_meru_single_configs()

    print_summary()
