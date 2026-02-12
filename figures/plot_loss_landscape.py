#!/usr/bin/env python3
"""
Figure 3: Loss landscape visualization.

Inputs:
  --baseline-npy  Numpy file containing a 2D array Z_baseline (loss values).
  --ours-npy      Numpy file containing a 2D array Z_ours with the same shape.

Both surfaces are plotted over a symmetric perturbation grid defined by:
  ΔK1, ΔK2 ∈ [-delta, delta], where delta defaults to 1.0.
You may change the extent via --delta or provide explicit ticks.

Usage:
    python figures/plot_loss_landscape.py \
        --baseline-npy data/loss_flatland.npy \
        --ours-npy data/loss_fedgeorefine.npy \
        --output figures/fig3_loss_landscape.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator

CMAP_BASELINE = "magma_r"
CMAP_OURS = "viridis"


def _build_grid(shape, delta):
    n_rows, n_cols = shape
    x = np.linspace(-delta, delta, n_cols)
    y = np.linspace(-delta, delta, n_rows)
    return np.meshgrid(x, y)


def _plot_surface(ax, X, Y, Z, title, cmap):
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_xlabel(r"$\Delta K_1$")
    ax.set_ylabel(r"$\Delta K_2$")
    ax.set_zlabel("Validation Loss")
    ax.set_title(title, pad=12)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.view_init(elev=35, azim=-55)
    return surf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss landscape comparison.")
    parser.add_argument("--baseline-npy", type=Path, required=True, help="2D numpy array for FlatLand loss surface.")
    parser.add_argument("--ours-npy", type=Path, required=True, help="2D numpy array for FedGeoRefine loss surface.")
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Half-width of the curvature perturbation range (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig3_loss_landscape.png"),
        help="Output path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (dots per inch) used when saving the figure.",
    )
    parser.add_argument("--contour", action="store_true", help="Render 2D contour plots instead of 3D surfaces.")
    parser.add_argument(
        "--axis-labels",
        nargs=2,
        default=[r"$\Delta K_1$", r"$\Delta K_2$"],
        metavar=("X_LABEL", "Y_LABEL"),
        help="Axis labels for the perturbation grid.",
    )
    parser.add_argument(
        "--title-baseline",
        type=str,
        default="FlatLand",
        help="Panel title for the baseline subplot.",
    )
    parser.add_argument(
        "--title-ours",
        type=str,
        default="FedGeoRefine",
        help="Panel title for the refined subplot.",
    )
    parser.add_argument(
        "--basin-threshold",
        type=float,
        default=None,
        help="If set, overlay the {min + threshold} low-loss basin boundary.",
    )
    parser.add_argument(
        "--annotate-min",
        action="store_true",
        help="Annotate the minimum loss point in each subplot.",
    )
    args = parser.parse_args()

    Z_base = np.load(args.baseline_npy)
    Z_ours = np.load(args.ours_npy)
    if Z_base.shape != Z_ours.shape:
        raise ValueError(f"Shape mismatch: baseline {Z_base.shape} vs ours {Z_ours.shape}")

    X, Y = _build_grid(Z_base.shape, args.delta)
    x_label, y_label = args.axis_labels

    def _annotate_min(ax, Z, title_color="white", is3d=False):
        if not args.annotate_min:
            return
        idx = np.unravel_index(np.argmin(Z), Z.shape)
        if is3d:
            ax.scatter(
                X[idx],
                Y[idx],
                Z[idx],
                marker="*",
                s=45,
                color=title_color,
            )
        else:
            ax.plot(
                X[idx],
                Y[idx],
                marker="+",
                markersize=9,
                markeredgewidth=1.8,
                color=title_color,
            )

    def _draw_basin(ax, Z, color):
        if args.basin_threshold is None:
            return
        level = Z.min() + args.basin_threshold
        ax.contour(
            X,
            Y,
            Z,
            levels=[level],
            colors=[color],
            linewidths=1.2,
            linestyles="--",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.contour:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        cs0 = axes[0].contourf(X, Y, Z_base, cmap=CMAP_BASELINE, levels=30)
        cs1 = axes[1].contourf(X, Y, Z_ours, cmap=CMAP_OURS, levels=30)
        axes[0].set_title(args.title_baseline)
        axes[1].set_title(args.title_ours)
        for ax in axes:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        _draw_basin(axes[0], Z_base, "white")
        _draw_basin(axes[1], Z_ours, "white")
        _annotate_min(axes[0], Z_base)
        _annotate_min(axes[1], Z_ours)
        fig.colorbar(cs0, ax=axes[0], shrink=0.8)
        fig.colorbar(cs1, ax=axes[1], shrink=0.8)
    else:
        fig = plt.figure(figsize=(10, 4.2))
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_surface(ax0, X, Y, Z_base, args.title_baseline, CMAP_BASELINE)
        _plot_surface(ax1, X, Y, Z_ours, args.title_ours, CMAP_OURS)
        ax0.set_xlabel(x_label)
        ax0.set_ylabel(y_label)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        _annotate_min(ax0, Z_base, title_color="black", is3d=True)
        _annotate_min(ax1, Z_ours, title_color="black", is3d=True)
        fig.tight_layout()

    fig.savefig(args.output, dpi=args.dpi)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
