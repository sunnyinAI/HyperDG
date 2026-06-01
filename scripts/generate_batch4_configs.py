#!/usr/bin/env python3
"""Generate Batch-4 multi-seed configs for variance bars.

Seeds 1, 2, 3 (seed 0 already covered by Batch 1/2/3) for the four
critical conditions that together let us claim:

    "ERM through the HyperDG codepath statistically matches the full
     method on standard DG benchmarks."

Conditions
----------
  * erm        : all auxiliary losses off, curvature frozen at init
                 (mirrors Batch 2 erm)
  * baseline   : alpha_curv=0.2 + all other defaults (full HyperDG;
                 mirrors Batch 3 baseline)
  * fixed      : curvature frozen, all aux losses on
                 (mirrors Batch 1 fixed)
  * novar      : per-domain curvature, alpha_curv=0
                 (mirrors Batch 1 novar)

Datasets: PACS, VLCS, OfficeHome, TerraIncognita.
Splits  : 0, 1, 2, 3.
Seeds   : 1, 2, 3.

Total   : 4 conditions x 4 datasets x 4 splits x 3 seeds = 192 configs.
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
SEEDS  = [1, 2, 3]

BASE = {
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
    "n_steps": 5000,
    "batch_size": 32,
    "single_curvature": False,
}

CONDITIONS = {
    "erm":      dict(alpha_curv=0.0, beta_align=0.0, gamma_sharp=0.0,
                     delta_local_flat=0.0, delta_global_flat=0.0,
                     curvature_lr=0.0),
    "baseline": {},                              # all defaults
    "fixed":    dict(curvature_lr=0.0, alpha_curv=0.0),
    "novar":    dict(alpha_curv=0.0),
}


def make_cfg(cond: str, seed: int, ds: str, ds_cfg: str, data_dir: str,
             split: int) -> dict:
    hparams = dict(BASE)
    hparams.update(CONDITIONS[cond])
    return {
        "experiment_name": f"batch4_seed{seed}_{cond}_{ds}_split{split}",
        "seed": seed,
        "output_dir": f"experiments/batch4/seed{seed}/{cond}/{ds}_split{split}",
        "device": "cuda",
        "eval_on_start": False,
        "dataset": {
            "config":    ds_cfg,
            "data_dir":  data_dir,
            "test_envs": [split],
        },
        "algorithm": "hyperdg",
        "hparams": hparams,
    }


def dump(cfg: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    n = 0
    for seed in SEEDS:
        for cond in CONDITIONS:
            for ds, ds_cfg, data_dir in DATASETS:
                for split in SPLITS:
                    cfg = make_cfg(cond, seed, ds, ds_cfg, data_dir, split)
                    dump(cfg, ROOT / f"configs/batch4/seed{seed}/{cond}/"
                                     f"{ds}_split{split}.yaml")
                    n += 1
    print(f"Wrote {n} configs under configs/batch4/")


if __name__ == "__main__":
    main()
