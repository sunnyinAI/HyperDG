#!/usr/bin/env python3
"""Plot curvature variance bars per domain for multiple methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bar chart of curvature variance per domain.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV with columns dataset,domain,method,variance")
    parser.add_argument("--dataset", type=str, help="Filter to a single dataset")
    parser.add_argument("--output", type=str, default="outputs/fig_curvature_bars.pdf")
    parser.add_argument("--title", type=str, default="Curvature variance per domain")
    parser.add_argument("--ylabel", type=str, default="Var(K)")
    parser.add_argument("--palette", type=str, nargs="*", default=None, help="Custom colors per method")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.dataset:
        df = df[df["dataset"].str.lower() == args.dataset.lower()]
    if df.empty:
        raise SystemExit("No data after filtering; check dataset name.")

    methods = df["method"].unique()
    domains = df["domain"].unique()
    methods_sorted = sorted(methods)
    domains_sorted = sorted(domains)

    palette = args.palette or ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]
    colors = {m: palette[i % len(palette)] for i, m in enumerate(methods_sorted)}

    width = 0.8 / len(methods_sorted)
    x = range(len(domains_sorted))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(1.8 * len(domains_sorted), 3.5))

    for idx, method in enumerate(methods_sorted):
        subset = df[df["method"] == method].set_index("domain").reindex(domains_sorted)
        ax.bar(
            [i + idx * width for i in x],
            subset["variance"],
            width=width,
            label=method,
            color=colors[method],
            edgecolor="white",
        )
        for i, value in enumerate(subset["variance"].values):
            if pd.isna(value):
                continue
            ax.text(i + idx * width, value + 0.01 * max(df["variance"]), f"{value:.2e}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks([i + width * (len(methods_sorted) - 1) / 2 for i in x])
    ax.set_xticklabels(domains_sorted, rotation=0)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved bar chart to {out_path}")


if __name__ == "__main__":
    main()
