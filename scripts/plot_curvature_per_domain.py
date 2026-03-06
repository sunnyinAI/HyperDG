#!/usr/bin/env python3
"""Plot per-domain curvature K_i evolution during training.

Addresses reviewer request: "Show how curvatures evolve during training and
across domains. Do they converge to similar values? Do they correlate with
dataset properties (e.g., class hierarchy depth)?"

Usage (single run):
    python scripts/plot_curvature_per_domain.py \
        --metrics experiments/resnet50_hyperdg_pacs/split0/metrics.jsonl \
        --domain-names Photo Art Cartoon Sketch \
        --output outputs/curvature_per_domain_pacs.pdf

Usage (multi-split average with shaded std):
    python scripts/plot_curvature_per_domain.py \
        --metrics experiments/resnet50_hyperdg_pacs/split*/metrics.jsonl \
        --domain-names Photo Art Cartoon Sketch \
        --output outputs/curvature_per_domain_pacs.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-domain curvature K_i evolution over training."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="One or more metrics.jsonl files (multiple = averaged with shaded std).",
    )
    parser.add_argument(
        "--domain-names",
        type=str,
        nargs="*",
        default=None,
        help="Human-readable domain labels (e.g. Photo Art Cartoon Sketch). "
        "Defaults to Domain 0, Domain 1, ...",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/curvature_per_domain.pdf",
        help="Output path for the figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Per-domain curvature K_i over training",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[7.0, 4.0],
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=1,
        help="Moving average window for smoothing curves (1 = no smoothing).",
    )
    return parser.parse_args()


def load_per_domain_curves(path: Path) -> Dict[int, Tuple[List[int], List[float]]]:
    """Return {domain_idx: (steps, K_values)} from a metrics.jsonl file."""
    domain_steps: Dict[int, List[int]] = defaultdict(list)
    domain_vals: Dict[int, List[float]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record.get("step", 0))
            d = 0
            while f"curvature_domain_{d}" in record:
                domain_steps[d].append(step)
                domain_vals[d].append(record[f"curvature_domain_{d}"])
                d += 1

    if not domain_steps:
        raise ValueError(
            f"No 'curvature_domain_*' keys found in {path}. "
            "Make sure you trained with the updated train.py that logs per-domain curvatures."
        )
    return {d: (domain_steps[d], domain_vals[d]) for d in sorted(domain_steps)}


def _moving_avg(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def _align_to_common_steps(
    all_curves: List[Dict[int, Tuple[List[int], List[float]]]]
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Given multiple runs, find common step indices per domain and compute
    mean ± std arrays.

    Returns {domain_idx: (steps, mean_K, std_K)}.
    """
    # Collect all domain indices
    all_domains = set()
    for curves in all_curves:
        all_domains.update(curves.keys())

    result: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for d in sorted(all_domains):
        run_steps = [curves[d][0] for curves in all_curves if d in curves]
        run_vals = [curves[d][1] for curves in all_curves if d in curves]
        if not run_steps:
            continue

        # Use the step sequence of the first run as reference; interpolate others
        ref_steps = np.array(run_steps[0])
        interp_vals = []
        for steps, vals in zip(run_steps, run_vals):
            s = np.array(steps, dtype=float)
            v = np.array(vals, dtype=float)
            if len(s) < 2:
                interp_vals.append(np.full_like(ref_steps, v[0], dtype=float))
            else:
                interp_vals.append(np.interp(ref_steps, s, v))

        stacked = np.stack(interp_vals, axis=0)
        result[d] = (ref_steps, stacked.mean(axis=0), stacked.std(axis=0))

    return result


def main() -> None:
    args = parse_args()

    mpl.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    paths = [Path(p) for p in args.metrics]
    all_curves = []
    for p in paths:
        try:
            all_curves.append(load_per_domain_curves(p))
        except ValueError as exc:
            print(f"[WARN] Skipping {p}: {exc}")

    if not all_curves:
        raise SystemExit("No valid metrics files with per-domain curvature data.")

    # Align and average across runs
    aligned = _align_to_common_steps(all_curves)
    num_domains = len(aligned)

    domain_labels: List[str] = args.domain_names or [
        f"Domain {d}" for d in range(num_domains)
    ]
    if len(domain_labels) < num_domains:
        domain_labels += [f"Domain {d}" for d in range(len(domain_labels), num_domains)]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for d, (steps, mean_k, std_k) in aligned.items():
        label = domain_labels[d] if d < len(domain_labels) else f"Domain {d}"
        color = colors[d % len(colors)]

        sm = _moving_avg(mean_k, args.smoothing)
        sd = _moving_avg(std_k, args.smoothing)
        xs = steps[len(steps) - len(sm):]  # trim steps to match smoothed length

        ax.plot(xs, sm, label=label, color=color, linewidth=1.8)
        if len(paths) > 1:
            ax.fill_between(xs, sm - sd, sm + sd, color=color, alpha=0.18)

    # Annotate final values
    for d, (steps, mean_k, std_k) in aligned.items():
        final_k = mean_k[-1]
        color = colors[d % len(colors)]
        label = domain_labels[d] if d < len(domain_labels) else f"Domain {d}"
        ax.annotate(
            f"K={final_k:.3f}",
            xy=(steps[-1], final_k),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="center",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Curvature $K_i$")
    ax.set_title(args.title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", frameon=False)

    # Inset: final curvature bar chart for easy reading
    inset = ax.inset_axes([0.65, 0.05, 0.32, 0.40])
    final_ks = [aligned[d][1][-1] for d in sorted(aligned)]
    final_stds = [aligned[d][2][-1] for d in sorted(aligned)] if len(paths) > 1 else None
    x_pos = np.arange(num_domains)
    bar_colors = [colors[d % len(colors)] for d in sorted(aligned)]
    inset.bar(
        x_pos,
        final_ks,
        yerr=final_stds,
        color=bar_colors,
        edgecolor="white",
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    inset.set_xticks(x_pos)
    inset.set_xticklabels(
        [domain_labels[d] if d < len(domain_labels) else f"D{d}" for d in sorted(aligned)],
        fontsize=7,
        rotation=30,
        ha="right",
    )
    inset.set_ylabel(r"Final $K_i$", fontsize=7)
    inset.tick_params(axis="y", labelsize=7)
    inset.set_title("Final values", fontsize=8)
    inset.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved per-domain curvature plot to {out_path}")


if __name__ == "__main__":
    main()
