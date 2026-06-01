#!/usr/bin/env python3
"""Generate Batch-1 reviewer-response configs.

Three families, all using algorithm=hyperdg (per-domain curvature
parameterization unless stated otherwise):

  * novar/    : alpha_curv=0.0   -> no variance regularization. Lets
                per-domain curvatures diverge freely. Answers the AE's
                explicit question.
  * fixed/    : curvature_lr=0.0 -> curvature frozen at init
                (softplus(0)+eps ~= 0.6932). Isolates "value of having
                hyperbolic geometry" from "value of learning curvature".
  * lvar/     : alpha_curv sweep {0.05, 0.5, 2.0} on PACS only. Traces
                the curvature-spread vs accuracy curve.

Datasets: PACS, VLCS, OfficeHome, TerraIncognita.  (DomainNet skipped
for cost; can be added later if signal warrants.)
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
    "n_steps": 5000,
    "batch_size": 32,
    "single_curvature": False,
}


def make_cfg(tag: str, family: str, ds: str, ds_cfg: str, data_dir: str,
             split: int, hparam_overrides: dict) -> dict:
    hparams = dict(BASE_HPARAMS)
    hparams.update(hparam_overrides)
    return {
        "experiment_name": f"batch1_{family}_{tag}_{ds}_split{split}",
        "seed": 0,
        "output_dir": f"experiments/batch1/{family}/{tag}/{ds}_split{split}",
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

    # ── Family 1: novar (per-domain, alpha_curv=0) ────────────────────────
    for ds, ds_cfg, data_dir in DATASETS:
        for split in SPLITS:
            cfg = make_cfg("a0", "novar", ds, ds_cfg, data_dir, split,
                           {"alpha_curv": 0.0})
            dump(cfg, ROOT / f"configs/batch1/novar/{ds}_split{split}.yaml")
            n += 1

    # ── Family 2: fixed curvature (curvature_lr=0) ────────────────────────
    for ds, ds_cfg, data_dir in DATASETS:
        for split in SPLITS:
            cfg = make_cfg("frozen", "fixed", ds, ds_cfg, data_dir, split,
                           {"curvature_lr": 0.0, "alpha_curv": 0.0})
            dump(cfg, ROOT / f"configs/batch1/fixed/{ds}_split{split}.yaml")
            n += 1

    # ── Family 3: lambda_var sweep on PACS only ───────────────────────────
    ds, ds_cfg, data_dir = DATASETS[0]
    for a in (0.05, 0.5, 2.0):
        tag = f"a{a}".replace(".", "p")
        for split in SPLITS:
            cfg = make_cfg(tag, "lvar", ds, ds_cfg, data_dir, split,
                           {"alpha_curv": float(a)})
            dump(cfg, ROOT / f"configs/batch1/lvar/{tag}_{ds}_split{split}.yaml")
            n += 1

    print(f"Wrote {n} configs under configs/batch1/")


if __name__ == "__main__":
    main()
