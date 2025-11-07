#!/usr/bin/env python3
"""
Figure 5: Curvature distribution histogram.

CSV schema:
method,curvature
FlatLand,0.32
FlatLand,0.40
FedGeoRefine,0.18
...

Usage:
    python figures/plot_curvature_histogram.py \
        --csv data/curvature_distribution.csv \
        --output figures/fig5_curvature_histogram.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE = {
    "FlatLand": "#B3B3B3",
    "FedGeoRefine": "#1B9E77",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot curvature distribution histograms.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with columns [method, curvature].")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig5_curvature_histogram.png"),
        help="Output image path.",
    )
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins.")
    parser.add_argument("--kde", action="store_true", help="Overlay kernel density estimate curves.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "method" not in df.columns or "curvature" not in df.columns:
        raise ValueError("CSV must contain 'method' and 'curvature' columns.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    methods = df["method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 3.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        subset = df[df["method"] == method]["curvature"]
        sns.histplot(
            subset,
            bins=args.bins,
            kde=args.kde,
            ax=ax,
            color=PALETTE.get(method, "#377EB8"),
            edgecolor="white",
        )
        ax.axvline(subset.mean(), linestyle="--", color="black", linewidth=1.2)
        ax.set_title(method)
        ax.set_xlabel("Curvature $K_i$")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
