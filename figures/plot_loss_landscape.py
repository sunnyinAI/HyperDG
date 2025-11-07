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
    parser.add_argument("--contour", action="store_true", help="Render 2D contour plots instead of 3D surfaces.")
    args = parser.parse_args()

    Z_base = np.load(args.baseline_npy)
    Z_ours = np.load(args.ours_npy)
    if Z_base.shape != Z_ours.shape:
        raise ValueError(f"Shape mismatch: baseline {Z_base.shape} vs ours {Z_ours.shape}")

    X, Y = _build_grid(Z_base.shape, args.delta)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.contour:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        cs0 = axes[0].contourf(X, Y, Z_base, cmap=CMAP_BASELINE, levels=30)
        cs1 = axes[1].contourf(X, Y, Z_ours, cmap=CMAP_OURS, levels=30)
        axes[0].set_title("FlatLand")
        axes[1].set_title("FedGeoRefine")
        for ax in axes:
            ax.set_xlabel(r"$\Delta K_1$")
            ax.set_ylabel(r"$\Delta K_2$")
        fig.colorbar(cs0, ax=axes[0], shrink=0.8)
        fig.colorbar(cs1, ax=axes[1], shrink=0.8)
    else:
        fig = plt.figure(figsize=(10, 4.2))
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_surface(ax0, X, Y, Z_base, "FlatLand", CMAP_BASELINE)
        _plot_surface(ax1, X, Y, Z_ours, "FedGeoRefine", CMAP_OURS)
        fig.tight_layout()

    fig.savefig(args.output, dpi=300)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
