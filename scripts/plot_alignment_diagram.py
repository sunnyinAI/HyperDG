"""Generate conceptual diagram for FedGeoRefine aggregation."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle


def draw_domain(ax, center, color, label):
    circle = Circle(center, radius=0.65, edgecolor=color, facecolor="none", linewidth=2)
    ax.add_patch(circle)
    ax.text(center[0], center[1] + 0.85, label, ha="center", va="bottom", fontsize=10, color=color)
    ax.scatter(center[0], center[1], color=color, s=50)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Left panel: heterogeneous manifolds
    ax = axes[0]
    ax.set_title("Heterogeneous manifolds")
    draw_domain(ax, (0.0, 0.0), "#4C72B0", "Domain A")
    draw_domain(ax, (1.4, -0.2), "#55A868", "Domain B")
    draw_domain(ax, (0.7, 1.4), "#C44E52", "Domain C")
    ax.text(0.7, -1.2, "Curvature mismatch", ha="center", va="center", fontsize=9)
    ax.axis("off")

    # Middle panel: tangent projection
    ax = axes[1]
    ax.set_title("Tangent-space aggregation")
    ax.plot([-0.5, 1.8], [0, 0], color="#333333", linewidth=1.5)
    ax.text(0.65, 0.2, "shared tangent plane $T_o$", ha="center", fontsize=9)
    proj_colors = ["#4C72B0", "#55A868", "#C44E52"]
    proj_positions = [-0.2, 0.8, 1.5]
    for pos, color in zip(proj_positions, proj_colors):
        ax.scatter(pos, 0.0, color=color, s=50)
        arrow = FancyArrowPatch(
            posA=(pos, 0.9),
            posB=(pos, 0.05),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=1.5,
            color=color,
        )
        ax.add_patch(arrow)
    ax.scatter(0.7, 0.0, color="#8172B3", s=100, marker="X", zorder=5)
    ax.text(0.7, -0.25, "Aggregated embedding", ha="center", fontsize=9, color="#8172B3")
    ax.axis("off")

    # Right panel: curvature-refined manifold
    ax = axes[2]
    ax.set_title("Curvature-aligned manifold")
    circle = Circle((0.7, 0.2), radius=0.85, edgecolor="#8172B3", facecolor="none", linewidth=2.5)
    ax.add_patch(circle)
    ax.scatter([0.1, 0.7, 1.3], [0.4, 0.8, 0.1], color="#8172B3", s=40)
    ax.text(0.7, -0.9, "Consistent curvature $\\bar{K}$", ha="center", fontsize=9)
    ax.axis("off")

    fig.suptitle("FedGeoRefine: Tangent-space alignment and curvature refinement", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/fedgeorefine_overview.png", dpi=300)
    print("Saved overview diagram to plots/fedgeorefine_overview.png")


if __name__ == "__main__":
    main()
