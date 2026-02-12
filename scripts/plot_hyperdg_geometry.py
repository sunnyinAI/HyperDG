#!/usr/bin/env python3
"""Conceptual figure: Euclidean vs. Hyperbolic domain alignment."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def draw_plane(ax, origin, color, label):
    x = np.linspace(-1.0, 1.0, 30)
    y = np.linspace(-0.3, 0.3, 30)
    X, Y = np.meshgrid(x, y)
    Z = 0.0 * X
    ax.plot_surface(
        X + origin[0],
        Y + origin[1],
        Z + origin[2],
        color=color,
        alpha=0.25,
        linewidth=0,
        shade=False,
    )
    ax.text(origin[0], origin[1], origin[2] + 0.4, label, color=color, ha="center")
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        0.0,
        0.4,
        0.0,
        color=color,
        linewidth=2,
        arrow_length_ratio=0.2,
    )


def draw_hyperboloid(ax, origin, color, label):
    theta = np.linspace(0, 2 * np.pi, 60)
    r = np.linspace(0.0, 0.7, 40)
    THETA, R = np.meshgrid(theta, r)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    Z = np.sqrt(1.0 + R**2) - 1.0
    ax.plot_surface(
        origin[0] + X,
        origin[1] + Y,
        origin[2] + Z,
        color=color,
        alpha=0.35,
        linewidth=0,
        shade=False,
    )
    ax.text(origin[0], origin[1], origin[2] + 1.0, label, color=color, ha="center")
    ax.quiver(
        origin[0],
        origin[1],
        origin[2] + 0.2,
        0.0,
        0.4,
        0.4,
        color=color,
        linewidth=2,
        arrow_length_ratio=0.2,
    )


def main():
    fig = plt.figure(figsize=(11, 4))
    # Panel A: Euclidean space
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_title("(a) Euclidean latent space ($K_i = 0$)")
    draw_plane(ax1, (-1.2, -0.4, -0.2), "#4C72B0", "Domain A")
    draw_plane(ax1, (0.0, 0.4, 0.1), "#C44E52", "Domain B")
    draw_plane(ax1, (1.2, -0.2, 0.3), "#55A868", "Domain C")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-0.5, 1.0)
    ax1.set_axis_off()
    ax1.text2D(0.5, -0.08, "Disjoint tangent directions", transform=ax1.transAxes, ha="center")

    # Panel B: Hyperbolic space
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("(b) Hyperbolic latent space ($K_i < 0$)")
    draw_hyperboloid(ax2, (-1.0, -0.2, -0.5), "#4C72B0", "Domain A")
    draw_hyperboloid(ax2, (0.0, 0.2, -0.3), "#C44E52", "Domain B")
    draw_hyperboloid(ax2, (1.0, -0.1, -0.6), "#55A868", "Domain C")
    # shared geodesic arrow
    ax2.quiver(
        -1.2,
        -0.2,
        -0.1,
        2.4,
        0.1,
        0.3,
        color="#8172B3",
        linewidth=2.5,
        arrow_length_ratio=0.1,
    )
    ax2.text2D(0.5, -0.08, "Shared Lorentzian meta-manifold", transform=ax2.transAxes, ha="center")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_axis_off()

    fig.suptitle("HyperDG vs. Euclidean latent spaces")
    fig.tight_layout()
    fig.savefig("outputs/fig1_hyperdg_geometry.png", dpi=300)
    print("Saved to outputs/fig1_hyperdg_geometry.png")


if __name__ == "__main__":
    main()
