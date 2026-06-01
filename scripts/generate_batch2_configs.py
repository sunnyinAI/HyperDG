#!/usr/bin/env python3
"""Generate Batch-2 leave-one-out (LOO) component-ablation configs.

Each ablation turns off ONE component of the HyperDG total loss, keeping
everything else at the reference setting.  Combined with Batch 1's
``novar`` (no variance reg) and ``fixed`` (no curvature learning), this
gives a clean leave-one-out across all components, plus a true ERM
floor.

Reference HyperDG components:
  * alpha_curv     = 0.2   (curvature variance reg)     <- killed in Batch 1 (novar)
  * beta_align     = 0.5   (cross-domain prototype alignment)
  * gamma_sharp    = 0.05  (tangent-norm sharpness, K-scaled)
  * delta_local    = 0.1   (curvature-gradient sharpness)
  * delta_global   = 0.1   (per-domain vs global loss disagreement)
  * curvature_lr   = 1e-4  (curvature learning)         <- killed in Batch 1 (fixed)

Batch-2 ablations (new):
  * erm         : all coefficients = 0 and curvature_lr = 0.  Pure ERM
                  through the HyperDG codepath (sanity floor).
  * no_align    : beta_align    = 0.
  * no_sharp    : gamma_sharp   = 0.
  * no_flat     : delta_local = delta_global = 0.

Datasets / splits:
  * erm       : PACS, VLCS, OfficeHome, TerraIncognita      x 4 splits
  * others    : PACS, OfficeHome                            x 4 splits
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DATASETS_ALL = [
    ("pacs",            "configs/datasets/pacs.yaml",            "data"),
    ("vlcs",            "configs/datasets/vlcs.yaml",            "data"),
    ("officehome",      "configs/datasets/officehome.yaml",      "data/officehome_clean"),
    ("terraincognita",  "configs/datasets/terraincognita.yaml",  "data"),
]
DATASETS_PAIR = DATASETS_ALL[:1] + [DATASETS_ALL[2]]   # PACS + OfficeHome

SPLITS = [0, 1, 2, 3]

REFERENCE_HPARAMS = {
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

ABLATIONS = {
    "erm":      dict(alpha_curv=0.0, beta_align=0.0, gamma_sharp=0.0,
                     delta_local_flat=0.0, delta_global_flat=0.0,
                     curvature_lr=0.0),
    "no_align": dict(beta_align=0.0),
    "no_sharp": dict(gamma_sharp=0.0),
    "no_flat":  dict(delta_local_flat=0.0, delta_global_flat=0.0),
}


def make_cfg(family: str, ds: str, ds_cfg: str, data_dir: str,
             split: int, overrides: dict) -> dict:
    hparams = dict(REFERENCE_HPARAMS)
    hparams.update(overrides)
    return {
        "experiment_name": f"batch2_{family}_{ds}_split{split}",
        "seed": 0,
        "output_dir": f"experiments/batch2/{family}/{ds}_split{split}",
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

    # ── ERM floor on all 4 datasets ────────────────────────────────────────
    for ds, ds_cfg, data_dir in DATASETS_ALL:
        for split in SPLITS:
            cfg = make_cfg("erm", ds, ds_cfg, data_dir, split, ABLATIONS["erm"])
            dump(cfg, ROOT / f"configs/batch2/erm/{ds}_split{split}.yaml")
            n += 1

    # ── Leave-one-out on PACS + OfficeHome ────────────────────────────────
    for family in ("no_align", "no_sharp", "no_flat"):
        for ds, ds_cfg, data_dir in DATASETS_PAIR:
            for split in SPLITS:
                cfg = make_cfg(family, ds, ds_cfg, data_dir, split,
                               ABLATIONS[family])
                dump(cfg, ROOT / f"configs/batch2/{family}/{ds}_split{split}.yaml")
                n += 1

    print(f"Wrote {n} configs under configs/batch2/")


if __name__ == "__main__":
    main()
