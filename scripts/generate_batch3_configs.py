#!/usr/bin/env python3
"""Generate Batch-3 configs for the curvature-trajectory figure.

Goal: pair the existing Batch-1 ``novar`` runs (alpha_curv=0) with
matching runs at alpha_curv=0.2 (default variance reg), with full
per-domain curvature logging, so we can plot trajectories side-by-side
across PACS / VLCS / OfficeHome / TerraIncognita and confirm that
"curvatures cluster even without variance reg" is not just a PACS
artifact.

This is identical to the standard HyperDG baseline but lives in a
batch3/ namespace so its logging is preserved alongside Batch 1's.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    ("pacs",            "configs/datasets/pacs.yaml",            "data"),
    ("vlcs",            "configs/datasets/vlcs.yaml",            "data"),
    ("officehome",      "configs/datasets/officehome.yaml",      "data/officehome_clean"),
    ("terraincognita",  "configs/datasets/terraincognita.yaml",  "data"),
]

SPLITS = [0, 1, 2, 3]

BASELINE_HPARAMS = {
    "lr": 5.0e-5,
    "curvature_lr": 1.0e-4,
    "weight_decay": 0.0,
    "grad_clip_norm": 5.0,
    "nonlinear_classifier": False,
    "checkpoint_freq": 200,
    "alpha_curv": 0.2,                # variance reg ON (matches paper baseline)
    "beta_align": 0.5,
    "gamma_sharp": 0.05,
    "delta_local_flat": 0.1,
    "delta_global_flat": 0.1,
    "curvature_epsilon": 1.0e-4,
    "n_steps": 5000,
    "batch_size": 32,
    "single_curvature": False,
}


def make_cfg(ds: str, ds_cfg: str, data_dir: str, split: int) -> dict:
    return {
        "experiment_name": f"batch3_baseline_{ds}_split{split}",
        "seed": 0,
        "output_dir": f"experiments/batch3/baseline/{ds}_split{split}",
        "device": "cuda",
        "eval_on_start": False,
        "dataset": {
            "config":    ds_cfg,
            "data_dir":  data_dir,
            "test_envs": [split],
        },
        "algorithm": "hyperdg",
        "hparams": dict(BASELINE_HPARAMS),
    }


def dump(cfg: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    n = 0
    for ds, ds_cfg, data_dir in DATASETS:
        for split in SPLITS:
            cfg = make_cfg(ds, ds_cfg, data_dir, split)
            dump(cfg, ROOT / f"configs/batch3/baseline/{ds}_split{split}.yaml")
            n += 1
    print(f"Wrote {n} configs under configs/batch3/baseline/")


if __name__ == "__main__":
    main()
