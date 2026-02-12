#!/usr/bin/env python3
"""Plot per-dataset curvature variance vs accuracy (two-axis combo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curvature variance + accuracy summary.")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metrics.jsonl files (one per dataset)")
    parser.add_argument("--labels", nargs="+", required=True, help="Dataset labels matching the metrics order")
    parser.add_argument("--num-envs", nargs="+", type=int, required=True, help="Number of eval domains, one per dataset")
    parser.add_argument("--output", type=str, default="outputs/fig_curvature_summary.pdf")
    parser.add_argument("--title", type=str, default="Curvature variance vs DG accuracy")
    return parser.parse_args()


def load_final_stats(path: Path, num_envs: int) -> tuple[float, float]:
    lines = path.read_text().strip().splitlines()
    if not lines:
        raise ValueError(f"Empty metrics file: {path}")
    record = json.loads(lines[-1])
    var = record.get("curvature_var")
    if var is None:
        raise ValueError(f"No curvature_var in {path}")
    accs = [record.get(f"env{i}_out_acc") for i in range(num_envs)]
    if any(a is None for a in accs):
        raise ValueError(f"Missing env accuracies in {path}")
    acc = float(np.mean(accs)) * 100.0
    return var, acc


def main() -> None:
    args = parse_args()
    if not (len(args.metrics) == len(args.labels) == len(args.num_envs)):
        raise SystemExit("metrics, labels, and num_envs must have same length")

    stats: List[tuple[float, float]] = []
    for metric_path, n in zip(args.metrics, args.num_envs):
        stats.append(load_final_stats(Path(metric_path), n))

    vars_, accs = zip(*stats)

    x = np.arange(len(args.labels))
    fig, ax1 = plt.subplots(figsize=(1.8 * len(x), 3.5))
    ax1.bar(x, vars_, color="#4C72B0", alpha=0.85, label="Var(K)")
    ax1.set_ylabel("Curvature Variance Var(K)")
    ax1.set_xlabel("Dataset")
    ax1.set_xticks(x)
    ax1.set_xticklabels(args.labels)

    ax2 = ax1.twinx()
    ax2.plot(x, accs, color="orange", marker="o", linewidth=2, label="Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")

    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle(args.title)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved summary plot to {out_path}")


if __name__ == "__main__":
    main()
