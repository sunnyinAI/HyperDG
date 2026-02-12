#!/usr/bin/env python3
"""Generate per-split configs for SFT to enable parallel training."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

BASE_CONFIGS = {
    "vlcs": Path("configs/training/resnet50_sft_vlcs.yaml"),
    "pacs": Path("configs/training/resnet50_sft_pacs.yaml"),
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def configure_split(cfg: dict, dataset: str, split: int) -> dict:
    cfg = deepcopy(cfg)
    cfg["seed"] = split
    base_name = cfg.get("experiment_name", f"sft_{dataset}")
    cfg["experiment_name"] = f"{base_name}_split{split}"
    base_out = Path(cfg.get("output_dir", f"experiments/{dataset}"))
    cfg["output_dir"] = str(base_out / f"split{split}")

    dataset_cfg = deepcopy(cfg.get("dataset", {}))
    dataset_cfg["test_envs"] = [split]
    cfg["dataset"] = dataset_cfg
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize per-split SFT configs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(BASE_CONFIGS.keys()),
        choices=BASE_CONFIGS.keys(),
        help="Datasets to generate configs for.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=4,
        help="Number of leave-one-domain-out splits (default: 4).",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        base_path = BASE_CONFIGS[dataset]
        if not base_path.exists():
            raise FileNotFoundError(f"Missing base config {base_path}")
        base_cfg = load_yaml(base_path)
        for split in range(args.splits):
            split_cfg = configure_split(base_cfg, dataset, split)
            out_path = Path("experiments/configs") / f"sft_{dataset}_split{split}.yaml"
            write_yaml(out_path, split_cfg)
        print(f"[OK] Generated SFT configs for {dataset} ({args.splits} splits)")


if __name__ == "__main__":
    main()
