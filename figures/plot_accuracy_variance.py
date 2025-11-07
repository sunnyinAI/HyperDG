#!/usr/bin/env python3
"""
Figure 4: Accuracy variance across domains.

CSV schema:
dataset,domain,flatland_mean,flatland_std,fedgeorefine_mean,fedgeorefine_std
VLCS,A,64.3,1.2,70.4,0.8
VLCS,C,59.8,1.5,66.7,0.9
PACS,Art,78.0,0.7,82.5,0.6
...

Usage:
    python figures/plot_accuracy_variance.py \
        --csv data/domain_accuracy.csv \
        --output figures/fig4_accuracy_variance.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_COLOR = "#B0B0B0"
OURS_COLOR = "#1F77B4"


def _plot_dataset(ax, df_dataset, title):
    domains = df_dataset["domain"].tolist()
    x = np.arange(len(domains))
    width = 0.35

    ax.bar(
        x - width / 2,
        df_dataset["flatland_mean"],
        width,
        yerr=df_dataset["flatland_std"],
        color=BASELINE_COLOR,
        label="FlatLand",
        capsize=3,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        df_dataset["fedgeorefine_mean"],
        width,
        yerr=df_dataset["fedgeorefine_std"],
        color=OURS_COLOR,
        label="FedGeoRefine",
        capsize=3,
        alpha=0.95,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=20, ha="right")
    ax.set_ylim(0, max(df_dataset[["flatland_mean", "fedgeorefine_mean"]].values.max() + 5, 100))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-domain accuracy variance.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with per-domain accuracies.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig4_accuracy_variance.pdf"),
        help="Output figure path.",
    )
    parser.add_argument("--legend-outside", action="store_true", help="Place legend outside the axes.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [
        "dataset",
        "domain",
        "flatland_mean",
        "flatland_std",
        "fedgeorefine_mean",
        "fedgeorefine_std",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    datasets = df["dataset"].unique().tolist()
    n_cols = min(len(datasets), 2)
    n_rows = int(np.ceil(len(datasets) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)

    for ax, dataset in zip(axes.flatten(), datasets):
        subset = df[df["dataset"] == dataset].copy()
        _plot_dataset(ax, subset, dataset)

    # Remove unused axes
    for ax in axes.flatten()[len(datasets) :]:
        ax.axis("off")

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if args.legend_outside:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    else:
        axes.flatten()[0].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
