#!/usr/bin/env python3
"""
Figure 1: Overview diagram for FedGeoRefine.

The script builds a conceptual illustration with three panels:
  • Left  – heterogeneous client manifolds.
  • Middle – Log/Exp mappings through the shared tangent plane.
  • Right – curvature-refined global manifold.

Usage
-----
python figures/plot_framework_overview.py --output figures/fig1_framework.png

The drawing is intentionally lightweight so it can serve as a
foundation for further polishing in Illustrator/Figma if desired.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

LEFT_COLORSET = ["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4"]
TANGENT_COLOR = "#E6F5D0"
GLOBAL_COLOR = "#4B9CD3"


def _draw_manifold(ax, center, width, height, color, label):
    ellipse = patches.Ellipse(
        center,
        width,
        height,
        angle=0,
        linewidth=2.0,
        edgecolor=color,
        facecolor="none",
        linestyle="-",
    )
    ax.add_patch(ellipse)
    ax.text(center[0], center[1] + height * 0.35, label, ha="center", va="bottom", fontsize=10, color=color)


def _draw_client_panel(ax):
    ax.set_title("Client Domains", fontsize=12, pad=10)
    x = 0.3
    y_positions = [0.25, 0.45, 0.65, 0.85]
    curvatures = [0.10, 0.20, 0.05, 0.15]
    for idx, (y, curv) in enumerate(zip(y_positions, curvatures)):
        width = 0.28
        height = 0.18 + curv
        _draw_manifold(
            ax,
            (x, y),
            width,
            height,
            LEFT_COLORSET[idx],
            f"Domain {chr(ord('A') + idx)}",
        )
        ax.text(
            x - width * 0.45,
            y - height * 0.5,
            "$K_{%d}$" % (idx + 1),
            fontsize=9,
            color=LEFT_COLORSET[idx],
        )
        ax.annotate(
            "",
            xy=(x + width * 0.5, y + 0.05),
            xytext=(x + width * 0.15, y + 0.15),
            arrowprops=dict(arrowstyle="->", color=LEFT_COLORSET[idx], lw=1.2),
        )

    ax.text(
        0.12,
        0.05,
        "Curvature mismatch\nacross clients",
        fontsize=9,
        ha="left",
        va="bottom",
    )


def _draw_tangent_panel(ax):
    ax.set_title("Tangent-Space Aggregation", fontsize=12, pad=10)

    # Tangent plane rectangle
    plane = patches.Rectangle(
        (0.2, 0.25),
        0.6,
        0.45,
        linewidth=1.5,
        edgecolor="#5C7A29",
        facecolor=TANGENT_COLOR,
        zorder=1,
    )
    ax.add_patch(plane)
    ax.text(0.5, 0.26, "Shared Tangent Plane", fontsize=10, ha="center", va="top", color="#5C7A29")

    # Log map arrows from clients to plane
    log_sources = [(0.1, 0.2), (0.1, 0.4), (0.1, 0.6), (0.1, 0.8)]
    for idx, (sx, sy) in enumerate(log_sources):
        ax.annotate(
            "",
            xy=(0.35, 0.45),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color=LEFT_COLORSET[idx], lw=1.5),
        )

    # Aggregated embeddings blob
    blob = patches.Ellipse((0.5, 0.47), 0.18, 0.10, facecolor="#377EB8", alpha=0.35, edgecolor="none")
    ax.add_patch(blob)
    ax.text(0.5, 0.47, "Aggregated\nEmbeddings", fontsize=9, ha="center", va="center", color="#17476B")

    ax.annotate(
        "Log map",
        xy=(0.22, 0.55),
        xytext=(0.03, 0.9),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#17476B", lw=1.0),
    )
    ax.annotate(
        "Exp map",
        xy=(0.66, 0.47),
        xytext=(0.95, 0.67),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#17476B", lw=1.0),
    )

    # Exp map to the right panel
    ax.annotate(
        "",
        xy=(1.0, 0.47),
        xytext=(0.7, 0.47),
        arrowprops=dict(arrowstyle="->", color="#17476B", lw=2.0),
    )


def _draw_global_panel(ax):
    ax.set_title("Curvature-Refined Global Model", fontsize=12, pad=10)
    global_curve = patches.Ellipse(
        (0.55, 0.5),
        0.65,
        0.35,
        angle=0,
        linewidth=2.2,
        edgecolor=GLOBAL_COLOR,
        facecolor="#E3F2FD",
        alpha=0.6,
    )
    ax.add_patch(global_curve)

    ax.text(0.55, 0.52, "Aligned\nCurvature", fontsize=11, ha="center", va="center", color=GLOBAL_COLOR)
    ax.text(
        0.55,
        0.12,
        "FedGeoRefine Global Model",
        fontsize=10,
        ha="center",
        va="center",
        color=GLOBAL_COLOR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FedGeoRefine framework overview figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig1_framework.png"),
        help="Output image path (PNG/PDF/SVG).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _draw_client_panel(axes[0])
    _draw_tangent_panel(axes[1])
    _draw_global_panel(axes[2])

    fig.savefig(args.output, dpi=300)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
