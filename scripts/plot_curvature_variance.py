"""Plot curvature variance trajectory from training metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curvature variance over rounds.")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="List of metrics.jsonl files (e.g., baseline then ours).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each curve (defaults to filenames).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/curvature_variance.png",
        help="Path to save the figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Curvature Variance Across Rounds",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    variances = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "curvature_var" in record:
                steps.append(record["step"])
                variances.append(record["curvature_var"])
    if not steps:
        raise ValueError(f"No curvature statistics found in {path}")
    return np.array(steps), np.array(variances)


def main() -> None:
    args = parse_args()
    metrics_paths = [Path(p) for p in args.metrics]
    labels = args.labels if args.labels else [p.stem for p in metrics_paths]

    plt.figure(figsize=(6.5, 4.0))
    for path, label in zip(metrics_paths, labels):
        steps, variances = load_metrics(path)
        plt.plot(steps, variances, label=label, linewidth=2)

    plt.xlabel("Training step")
    plt.ylabel("Curvature variance")
    plt.title(args.title)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved curvature variance plot to {output_path}")


if __name__ == "__main__":
    main()
