"""Plot per-domain accuracy bars for VLCS/PACS splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-domain accuracy bars.")
    parser.add_argument(
        "--vlcs",
        type=str,
        nargs="+",
        required=True,
        help="Metrics files for VLCS splits.",
    )
    parser.add_argument(
        "--pacs",
        type=str,
        nargs="+",
        required=True,
        help="Metrics files for PACS splits.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/accuracy_variance.png",
    )
    parser.add_argument("--title", type=str, default="Per-domain Accuracy Comparison")
    parser.add_argument("--label", type=str, default="FedGeoRefine")
    return parser.parse_args()


def load_best_accuracy(metrics_path: Path, domain_index: int) -> float:
    key = f"env{domain_index}_out_acc"
    records: List[Dict] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    if not records or key not in records[0]:
        raise ValueError(f"Accuracy key {key} not found in {metrics_path}")
    best = max(records, key=lambda r: r.get(key, float("-inf")))
    return best[key] * 100.0


def plot_group(ax, accuracies: List[float], label: str, color: str) -> None:
    x = np.arange(len(accuracies))
    ax.bar(x, accuracies, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(label)
    for idx, acc in enumerate(accuracies):
        ax.text(idx, acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    args = parse_args()

    vlcs_acc = []
    for idx, path in enumerate(sorted(args.vlcs)):
        vlcs_acc.append(load_best_accuracy(Path(path), domain_index=idx))

    pacs_acc = []
    for idx, path in enumerate(sorted(args.pacs)):
        pacs_acc.append(load_best_accuracy(Path(path), domain_index=idx))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    plot_group(axes[0], vlcs_acc, "VLCS splits", color="#4C72B0")
    plot_group(axes[1], pacs_acc, "PACS splits", color="#55A868")
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved accuracy variance plot to {output_path}")


if __name__ == "__main__":
    main()
