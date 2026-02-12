#!/usr/bin/env python3
"""Compute 2D loss surfaces around a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from domainbed import datasets as db_datasets
from domainbed import hparams_registry

from sftdg.algorithms import build_algorithm
from sftdg.utils import load_yaml_config, set_seed


def resolve_dataset_config(root_cfg):
    dataset_cfg = root_cfg.get("dataset", {})
    nested = dataset_cfg.get("config")
    if nested:
        nested_cfg = load_yaml_config(nested)
        merged = {**nested_cfg, **{k: v for k, v in dataset_cfg.items() if k != "config"}}
        return merged
    return dataset_cfg


def default_hparams(algorithm: str, dataset: str, overrides: dict) -> dict:
    try:
        hparams = hparams_registry.default_hparams(algorithm, dataset)
    except Exception:
        hparams = hparams_registry.default_hparams("ERM", dataset)
    hparams.update(overrides or {})
    hparams["batch_size"] = int(hparams.get("batch_size", 32))
    return hparams


def choose_env_indices(total_envs: int, test_envs: Iterable[int], mode: str) -> List[int]:
    all_envs = list(range(total_envs))
    test_set = set(test_envs)
    if mode == "train":
        return [idx for idx in all_envs if idx not in test_set]
    if mode == "test":
        return [idx for idx in all_envs if idx in test_set]
    return all_envs


def build_loader(dataset, env_idx: int, max_samples: int, batch_size: int) -> torch.utils.data.DataLoader:
    env = dataset[env_idx]
    if max_samples > 0 and len(env) > max_samples:
        subset = torch.utils.data.Subset(env, list(range(max_samples)))
    else:
        subset = env
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=dataset.N_WORKERS)


def evaluate_loss(algorithm, loaders, device: torch.device) -> float:
    algorithm.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for loader in loaders:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = algorithm.predict(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
                total_loss += loss.item()
                total_count += y.size(0)
    return total_loss / max(total_count, 1)


def gram_schmidt(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    proj = torch.dot(v2, v1) * v1
    return v2 - proj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute 2D loss surface around a checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Training config used for the run.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint with state_dict (defaults to <output_dir>/model_last.pt).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to save the numpy surface array.")
    parser.add_argument("--grid-size", type=int, default=41, help="Number of samples per axis.")
    parser.add_argument("--delta", type=float, default=2.0, help="Range half-width for parameter perturbations.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help="Max samples per environment when estimating loss (0 keeps all).",
    )
    parser.add_argument(
        "--env-mode",
        choices=["train", "test", "all"],
        default="test",
        help="Which environments to average over when computing the loss.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Eval batch size.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    dataset_cfg = resolve_dataset_config(cfg)

    set_seed(args.seed)

    dataset_class = db_datasets.get_dataset_class(dataset_cfg["dataset"])
    hparams = default_hparams(cfg.get("algorithm", "sft"), dataset_cfg["dataset"], cfg.get("hparams", {}))
    dataset = dataset_class(dataset_cfg["data_dir"], dataset_cfg.get("test_envs", []), hparams)

    algorithm = build_algorithm(
        cfg.get("algorithm", "sft"),
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(dataset_cfg.get("test_envs", [])),
        hparams,
    )
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    algorithm.to(device)

    ckpt_path = args.checkpoint or Path(cfg.get("output_dir", "experiments/unnamed")) / "model_last.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    algorithm.load_state_dict(state_dict)

    env_indices = choose_env_indices(len(dataset), dataset_cfg.get("test_envs", []), args.env_mode)
    loaders = [
        build_loader(dataset, idx, args.max_samples, args.batch_size)
        for idx in env_indices
    ]

    base_vector = parameters_to_vector(algorithm.parameters()).detach()
    dir1 = torch.randn_like(base_vector)
    dir1 = dir1 / (dir1.norm() + 1e-12)
    dir2 = torch.randn_like(base_vector)
    dir2 = gram_schmidt(dir1, dir2)
    dir2 = dir2 / (dir2.norm() + 1e-12)

    grid = np.linspace(-args.delta, args.delta, args.grid_size)
    surface = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)

    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            vec = base_vector + a * dir1 + b * dir2
            vector_to_parameters(vec, algorithm.parameters())
            loss = evaluate_loss(algorithm, loaders, device)
            surface[i, j] = float(loss)

    vector_to_parameters(base_vector, algorithm.parameters())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, surface)
    print(f"[OK] Saved loss surface to {args.output}")


if __name__ == "__main__":
    main()
