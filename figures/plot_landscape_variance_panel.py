#!/usr/bin/env python3
"""
Compose the mixed landscape consistency + curvature variance figure (Option C).

Top row: curvature-perturbed loss contours before/after HyperDG refinement.
Bottom row: curvature variance trajectory and Ki histograms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed landscape + curvature variance panel.")
    parser.add_argument("--baseline-npy", type=Path, required=True, help="Baseline loss surface (np.load-able array).")
    parser.add_argument("--ours-npy", type=Path, required=True, help="Refined loss surface array.")
    parser.add_argument(
        "--curvature-csv",
        type=Path,
        required=True,
        help="CSV with columns [round, flatland_mean, flatland_std, fedgeorefine_mean, fedgeorefine_std].",
    )
    parser.add_argument(
        "--distribution-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns [method, curvature] to render histograms.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Half-width for the perturbation grid assumed by the surface arrays.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=30,
        help="Number of contour levels in the top-row plots.",
    )
    parser.add_argument(
        "--basin-threshold",
        type=float,
        default=0.1,
        help="Add a dashed contour for {min + threshold} low-loss basin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig_mixed_landscape.png"),
        help="Destination path for the composed figure.",
    )
    parser.add_argument("--baseline-label", type=str, default="FlatLand", help="Label for the baseline method.")
    parser.add_argument("--ours-label", type=str, default="FedGeoRefine", help="Label for the refined method.")
    parser.add_argument("--hist-bins", type=int, default=30, help="Number of bins for the curvature histogram.")
    return parser.parse_args()


def _build_grid(shape: tuple[int, int], delta: float) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    x = np.linspace(-delta, delta, cols)
    y = np.linspace(-delta, delta, rows)
    return np.meshgrid(x, y)


def _plot_surface(ax, X, Y, Z, cmap, title, basin_threshold, levels):
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$\beta_1$  (curvature perturbation)")
    ax.set_ylabel(r"$\beta_2$  (embedding projection)")
    if basin_threshold is not None:
        level = Z.min() + basin_threshold
        ax.contour(X, Y, Z, levels=[level], colors="white", linestyles="--", linewidths=1.3)
    return contour


def _maybe_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy()
    return np.zeros(len(df), dtype=float)


def _plot_variance(ax, df: pd.DataFrame, baseline_label: str, ours_label: str) -> None:
    rounds = df["round"].to_numpy()
    flat_mean = df["flatland_mean"].to_numpy()
    flat_std = _maybe_column(df, "flatland_std")
    ours_mean = df["fedgeorefine_mean"].to_numpy()
    ours_std = _maybe_column(df, "fedgeorefine_std")

    ax.plot(rounds, flat_mean, color="#C44E52", linewidth=2.0, label=f"{baseline_label} Var(K)")
    ax.fill_between(rounds, flat_mean - flat_std, flat_mean + flat_std, color="#C44E52", alpha=0.2)

    ax.plot(rounds, ours_mean, color="#4C72B0", linewidth=2.2, label=f"{ours_label} Var(K)")
    ax.fill_between(rounds, ours_mean - ours_std, ours_mean + ours_std, color="#4C72B0", alpha=0.2)

    ax.set_xlabel("Training round")
    ax.set_ylabel(r"Curvature variance  $\mathrm{Var}(K)$")
    ax.set_title("Curvature variance collapses after refinement", fontsize=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right")


def _plot_hist(ax, df: pd.DataFrame, bins: int, baseline_label: str, ours_label: str) -> None:
    methods = df["method"].unique()
    colors = {
        baseline_label: "#C44E52",
        ours_label: "#4C72B0",
        "FlatLand": "#C44E52",
        "FedGeoRefine": "#4C72B0",
    }

    for method in methods:
        subset = df[df["method"] == method]["curvature"].to_numpy()
        label = method
        color = colors.get(method, "#55A868")
        ax.hist(
            subset,
            bins=bins,
            alpha=0.55,
            density=True,
            label=label,
            color=color,
            edgecolor="white",
        )
    ax.set_xlabel("Curvature $K_i$")
    ax.set_ylabel("Density")
    ax.set_title("Curvature distribution across domains", fontsize=12)
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend()


if __name__ == "__main__":
    args = parse_args()

    Z_base = np.load(args.baseline_npy)
    Z_ours = np.load(args.ours_npy)
    if Z_base.shape != Z_ours.shape:
        raise ValueError(f"Surface shape mismatch: {Z_base.shape} vs {Z_ours.shape}")

    X, Y = _build_grid(Z_base.shape, args.delta)

    df_var = pd.read_csv(args.curvature_csv)
    df_hist = pd.read_csv(args.distribution_csv) if args.distribution_csv else None

    fig = plt.figure(figsize=(11.2, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.18)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    contour0 = _plot_surface(
        ax00,
        X,
        Y,
        Z_base,
        cmap="magma_r",
        title=f"{args.baseline_label}: Euclidean loss surface",
        basin_threshold=args.basin_threshold,
        levels=args.levels,
    )
    contour1 = _plot_surface(
        ax01,
        X,
        Y,
        Z_ours,
        cmap="viridis",
        title=f"{args.ours_label}: curvature-aligned surface",
        basin_threshold=args.basin_threshold,
        levels=args.levels,
    )
    fig.colorbar(contour0, ax=ax00, fraction=0.046, pad=0.03)
    fig.colorbar(contour1, ax=ax01, fraction=0.046, pad=0.03)

    ax10 = fig.add_subplot(gs[1, 0])
    _plot_variance(ax10, df_var, args.baseline_label, args.ours_label)

    ax11 = fig.add_subplot(gs[1, 1])
    if df_hist is not None:
        _plot_hist(ax11, df_hist, args.hist_bins, args.baseline_label, args.ours_label)
    else:
        ax11.axis("off")
        ax11.text(
            0.5,
            0.5,
            "No curvature distribution CSV provided",
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.suptitle("Landscape consistency + curvature variance", fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"[OK] Saved mixed figure to {args.output}")
