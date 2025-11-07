#!/usr/bin/env python3
"""
Figure 2: Curvature variance trajectory.

Expected CSV schema (header row required):
round,flatland_mean,flatland_std,fedgeorefine_mean,fedgeorefine_std
0,0.45,0.05,0.45,0.05
1,0.43,0.06,0.40,0.05
...

Usage:
    python figures/plot_curvature_variance.py \
        --csv data/curvature_variance.csv \
        --output figures/fig2_curvature_variance.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASELINE_COLOR = "#D55E00"
OURS_COLOR = "#0072B2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot curvature variance trajectory.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV file containing curvature statistics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig2_curvature_variance.pdf"),
        help="Path to save the figure.",
    )
    parser.add_argument("--title", type=str, default="", help="Optional title override.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in [
        "round",
        "flatland_mean",
        "flatland_std",
        "fedgeorefine_mean",
        "fedgeorefine_std",
    ]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    x = df["round"].values
    ax.plot(x, df["flatland_mean"], color=BASELINE_COLOR, linestyle="--", linewidth=2.0, label="FlatLand")
    ax.fill_between(
        x,
        df["flatland_mean"] - df["flatland_std"],
        df["flatland_mean"] + df["flatland_std"],
        color=BASELINE_COLOR,
        alpha=0.18,
    )

    ax.plot(x, df["fedgeorefine_mean"], color=OURS_COLOR, linestyle="-", linewidth=2.2, label="FedGeoRefine")
    ax.fill_between(
        x,
        df["fedgeorefine_mean"] - df["fedgeorefine_std"],
        df["fedgeorefine_mean"] + df["fedgeorefine_std"],
        color=OURS_COLOR,
        alpha=0.2,
    )

    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel(r"Curvature Variance $\mathrm{Var}(K_i)$")
    if args.title:
        ax.set_title(args.title, fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
