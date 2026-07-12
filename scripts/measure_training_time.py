#!/usr/bin/env python3
"""Measure and compare wall-clock training time across algorithms.

Addresses the reviewer request to quantify training cost against a genuine ERM
baseline. The benchmark also records peak PyTorch-allocated device memory.

The script runs a fixed number of update steps for each specified algorithm
config, records wall-clock seconds per step, and prints a comparison table.

Usage:
    python scripts/measure_training_time.py \
        --configs \
            configs/training/resnet50_erm_pacs.yaml \
            configs/training/resnet50_hyperdg_pacs.yaml \
            configs/training/resnet50_sft_pacs.yaml \
        --labels ERM HyperDG SFT \
        --baseline-label ERM \
        --warmup 20 \
        --steps 200 \
        --device cuda

Output (example):
    Algorithm    Steps    Time/step (ms)    Peak MiB    Relative
    ---------    -----    --------------    --------    --------
    ERM          200      42.1              8200        1.00x
    HyperDG      200      50.0              8300        1.19x
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from domainbed import datasets as db_datasets
from domainbed import hparams_registry
from domainbed.algorithms import ALGORITHMS as DB_ALGORITHMS
from domainbed.lib.fast_data_loader import InfiniteDataLoader
from domainbed.lib import misc as db_misc

from sftdg.algorithms import build_algorithm
from sftdg.utils import load_yaml_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark training wall-clock time across algorithms."
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="YAML config paths, one per algorithm to benchmark.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Human-readable labels matching --configs order.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup steps before timing begins.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of timed update steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda / cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write a CSV summary (e.g. outputs/timing.csv).",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default=None,
        help="Label used to normalize relative cost (defaults to the first result).",
    )
    return parser.parse_args()


def resolve_dataset_cfg(root_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = root_cfg.get("dataset", {})
    nested = dataset_cfg.get("config")
    if nested:
        merged = {**load_yaml_config(nested), **{k: v for k, v in dataset_cfg.items() if k != "config"}}
        return merged
    return dataset_cfg


def build_loaders(dataset, test_envs: Sequence[int], hparams: Dict[str, Any], seed: int):
    in_splits = []
    for env_i, env in enumerate(dataset):
        env_split = int(len(env) * hparams.get("holdout_fraction", 0.2))
        in_split, _ = db_misc.split_dataset(env, len(env) - env_split, db_misc.seed_hash(seed, env_i))
        in_splits.append(in_split)

    loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=hparams["batch_size"],
            num_workers=0,  # keep timing clean; avoid loader overhead
        )
        for env_i, env in enumerate(in_splits)
        if env_i not in test_envs
    ]
    return loaders


def time_algorithm(
    cfg_path: str,
    label: str,
    warmup: int,
    steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    cfg = load_yaml_config(cfg_path)
    dataset_cfg = resolve_dataset_cfg(cfg)

    base_algo = cfg.get("algorithm", "sft")
    base_key = base_algo if base_algo in DB_ALGORITHMS else "ERM"
    hparams = hparams_registry.default_hparams(base_key, dataset_cfg["dataset"])
    hparams.update(cfg.get("hparams", {}))
    hparams["batch_size"] = int(hparams.get("batch_size", 32))

    test_envs = dataset_cfg.get("test_envs", [0])
    dataset_class = db_datasets.get_dataset_class(dataset_cfg["dataset"])
    dataset = dataset_class(dataset_cfg["data_dir"], test_envs, hparams)

    loaders = build_loaders(dataset, test_envs, hparams, seed)
    algorithm = build_algorithm(
        base_algo,
        dataset.input_shape,
        dataset.num_classes,
        len(loaders),
        hparams,
    )
    algorithm.to(device)
    algorithm.train()

    iterator = zip(*loaders)

    # Warmup
    for _ in range(warmup):
        minibatches = [(x.to(device), y.to(device)) for x, y in next(iterator)]
        algorithm.update(minibatches)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        minibatches = [(x.to(device), y.to(device)) for x, y in next(iterator)]
        algorithm.update(minibatches)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = elapsed / steps * 1000.0
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "label": label,
        "config": cfg_path,
        "steps": steps,
        "total_s": elapsed,
        "ms_per_step": ms_per_step,
        "peak_memory_mb": peak_memory_mb,
    }


def baseline_time(results: List[Dict[str, Any]], baseline_label: str | None) -> float:
    if baseline_label is not None:
        for result in results:
            if result["label"] == baseline_label:
                return float(result["ms_per_step"])
        raise ValueError(f"Baseline label '{baseline_label}' was not measured.")
    return float(results[0]["ms_per_step"])


def print_table(results: List[Dict[str, Any]], baseline_label: str | None) -> None:
    if not results:
        return
    baseline_ms = baseline_time(results, baseline_label)
    col_w = [max(len(r["label"]) for r in results) + 2, 8, 18, 14, 12]
    header = (
        f"{'Algorithm':<{col_w[0]}}"
        f"{'Steps':>{col_w[1]}}"
        f"{'ms / step':>{col_w[2]}}"
        f"{'Peak MiB':>{col_w[3]}}"
        f"{'Relative':>{col_w[4]}}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        rel = r["ms_per_step"] / baseline_ms
        print(
            f"{r['label']:<{col_w[0]}}"
            f"{r['steps']:>{col_w[1]}}"
            f"{r['ms_per_step']:>{col_w[2]}.2f}"
            f"{r['peak_memory_mb']:>{col_w[3]}.1f}"
            f"{rel:>{col_w[4]}.2f}x"
        )
    print(sep + "\n")


def main() -> None:
    args = parse_args()
    labels = args.labels or [Path(c).stem for c in args.configs]
    if len(labels) < len(args.configs):
        labels += [Path(c).stem for c in args.configs[len(labels):]]

    raw_device = args.device
    if raw_device is None:
        raw_device = "cuda" if torch.cuda.is_available() else "cpu"
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raw_device = "cpu"
    device = torch.device(raw_device)

    print(f"Device: {device}  |  Warmup: {args.warmup}  |  Timed steps: {args.steps}\n")
    results: List[Dict[str, Any]] = []
    for cfg_path, label in zip(args.configs, labels):
        print(f"Benchmarking [{label}] from {cfg_path} ...")
        try:
            result = time_algorithm(cfg_path, label, args.warmup, args.steps, device, args.seed)
            results.append(result)
            print(f"  -> {result['ms_per_step']:.2f} ms/step  ({result['total_s']:.1f}s total)")
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print_table(results, args.baseline_label)

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_ms = baseline_time(results, args.baseline_label)
        lines = ["algorithm,config,steps,ms_per_step,peak_memory_mb,relative"]
        for r in results:
            rel = r["ms_per_step"] / baseline_ms
            lines.append(
                f"{r['label']},{r['config']},{r['steps']},{r['ms_per_step']:.4f},"
                f"{r['peak_memory_mb']:.2f},{rel:.4f}"
            )
        out_path.write_text("\n".join(lines) + "\n")
        print(f"CSV saved to {out_path}")


if __name__ == "__main__":
    main()
