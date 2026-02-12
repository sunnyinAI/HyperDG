#!/usr/bin/env python3
"""Generate ViT HyperDG configs and split variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DATASETS = {
    "vlcs": {
        "config": "configs/datasets/vlcs.yaml",
        "splits": 4,
    },
    "pacs": {
        "config": "configs/datasets/pacs.yaml",
        "splits": 4,
    },
    "officehome": {
        "config": "configs/datasets/officehome.yaml",
        "splits": 4,
    },
    "terraincognita": {
        "config": "configs/datasets/terraincognita.yaml",
        "splits": 4,
    },
    "domainnet": {
        "config": "configs/datasets/domainnet.yaml",
        "splits": 6,
    },
}

BACKBONES = {
    "vitb16": {
        "vit_model_name": "vit_base_patch16_224",
        "batch_size": 16,
        "prompt_length": 10,
    },
    "vitl14": {
        "vit_model_name": "vit_large_patch16_224",
        "batch_size": 8,
        "prompt_length": 20,
    },
}


def build_base_config(dataset: str, backbone: str) -> dict:
    ds_cfg = DATASETS[dataset]
    bb_cfg = BACKBONES[backbone]
    exp_name = f"{backbone}_hyperdg_{dataset}"
    cfg = {
        "experiment_name": exp_name,
        "seed": 0,
        "output_dir": f"experiments/{exp_name}",
        "device": "cuda",
        "dataset": {
            "config": ds_cfg["config"],
        },
        "algorithm": "hyperdg",
        "hparams": {
            "backbone": backbone,
            "lr": 5.0e-5,
            "curvature_lr": 1.0e-4,
            "weight_decay": 0.0,
            "batch_size": bb_cfg["batch_size"],
            "grad_clip_norm": 5.0,
            "nonlinear_classifier": False,
            "n_steps": 5000,
            "checkpoint_freq": 200,
            "alpha_curv": 0.2,
            "beta_align": 0.5,
            "gamma_sharp": 0.05,
            "delta_local_flat": 0.1,
            "delta_global_flat": 0.1,
            "curvature_epsilon": 1.0e-4,
            "vit_model_name": bb_cfg["vit_model_name"],
            "vit_prompt_length": bb_cfg["prompt_length"],
            "vit_prompt_dropout": 0.1,
            "vit_freeze_backbone": True,
        },
    }
    return cfg


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ViT configs + splits.")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(BACKBONES.keys()),
        choices=BACKBONES.keys(),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=DATASETS.keys(),
    )
    args = parser.parse_args()

    for backbone in args.backbones:
        for dataset in args.datasets:
            cfg = build_base_config(dataset, backbone)
            base_path = Path("configs/training") / f"{backbone}_hyperdg_{dataset}.yaml"
            write_yaml(base_path, cfg)

            splits = DATASETS[dataset]["splits"]
            for split in range(splits):
                split_cfg = cfg.copy()
                split_cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
                split_cfg["experiment_name"] = f"{cfg['experiment_name']}_split{split}"
                split_cfg["output_dir"] = f"{cfg['output_dir']}/split{split}"
                split_cfg["seed"] = split
                split_cfg["dataset"]["test_envs"] = [split]
                split_path = (
                    Path("experiments/configs")
                    / f"{backbone}_hyperdg_{dataset}_split{split}.yaml"
                )
                write_yaml(split_path, split_cfg)
            print(f"[OK] Generated configs for {backbone} / {dataset}")


if __name__ == "__main__":
    main()
