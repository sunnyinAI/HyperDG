#!/usr/bin/env python3
"""Plot per-domain curvature trajectories during training.

Produces the central figure for the rewritten paper:

    For each dataset (PACS, VLCS, OfficeHome, TerraIncognita), show
    side-by-side curvature trajectories under

        (left)  alpha_curv = 0.0   (no variance reg)      -- Batch 1 novar
        (right) alpha_curv = 0.2   (default variance reg) -- Batch 3 baseline

    Each line is one (split, domain) pair.  If both panels show the
    curvatures collapsing into a tight band, the story is:
        "Per-domain curvature parameterization is unnecessary -- gradient
         dynamics drive them together regardless of the variance penalty."

Inputs:
  experiments/batch1/novar/a0/<ds>_split<n>/metrics.jsonl     (novar)
  experiments/batch3/baseline/<ds>_split<n>/metrics.jsonl     (baseline)

Output:
  outputs/curvature_trajectories.pdf       (one figure, 4x2 grid)
  outputs/curvature_trajectories_<ds>.pdf  (one figure per dataset)
  outputs/curvature_summary.csv            (final values + spread stats)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["pacs", "vlcs", "officehome", "terraincognita"]
SPLITS = [0, 1, 2, 3]

NOVAR_PATH    = ROOT / "experiments/batch1/novar/a0"
BASELINE_PATH = ROOT / "experiments/batch3/baseline"

DOMAIN_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3",
                 "#CCB974", "#64B5CD", "#E07B39", "#6BAED6"]


def load_trajectory(metrics_path: Path) -> dict:
    """Return {'step': [...], 'domain_<i>': [...], 'mean': [...], ...}."""
    if not metrics_path.exists():
        return {}
    records = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {}
    steps = [r.get("step", i * 200) for i, r in enumerate(records)]
    domain_keys = sorted(k for k in records[0] if k.startswith("curvature_domain_"))
    out = {"step": steps,
           "mean": [r.get("curvature_mean", r.get("mean_curvature", np.nan))
                    for r in records],
           "std":  [r.get("curvature_std", np.nan) for r in records]}
    for k in domain_keys:
        out[k] = [r.get(k, np.nan) for r in records]
    return out


def collect_runs(base_path: Path, dataset: str) -> list[dict]:
    """Load every (split) trajectory for one (condition, dataset)."""
    runs = []
    for s in SPLITS:
        p = base_path / f"{dataset}_split{s}" / "metrics.jsonl"
        traj = load_trajectory(p)
        if traj:
            traj["split"] = s
            runs.append(traj)
    return runs


def plot_one_panel(ax, runs: list[dict], title: str,
                   show_legend: bool = False) -> tuple[float, float]:
    """Plot all (split, domain) curvature lines on one axis.

    Returns (final_spread_mean, final_spread_max) — width of the
    domain-curvature band across splits at the final step.
    """
    domain_keys_seen: set[str] = set()
    final_spreads = []

    for run in runs:
        steps = run["step"]
        dks = [k for k in run if k.startswith("curvature_domain_")]
        domain_keys_seen.update(dks)
        for i, dk in enumerate(sorted(dks)):
            color = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]
            vals = run[dk]
            ax.plot(steps, vals, color=color, lw=0.9, alpha=0.55)
        finals = [run[dk][-1] for dk in dks if not np.isnan(run[dk][-1])]
        if finals:
            final_spreads.append(max(finals) - min(finals))

    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("learnt curvature K", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    # Subtle init reference (softplus(0)+1e-4 ≈ 0.6932)
    ax.axhline(0.6932, color="gray", lw=0.6, linestyle=":",
               alpha=0.6, label="init (≈0.6932)" if show_legend else None)

    if show_legend:
        # One legend entry per domain index, not per split
        n_doms = len(domain_keys_seen)
        handles = [plt.Line2D([], [], color=DOMAIN_COLORS[i % len(DOMAIN_COLORS)],
                              lw=1.5, label=f"domain {i}")
                   for i in range(n_doms)]
        handles.append(plt.Line2D([], [], color="gray", lw=0.8, linestyle=":",
                                   label="init"))
        ax.legend(handles=handles, fontsize=6.5, loc="best", frameon=False)

    if final_spreads:
        return float(np.mean(final_spreads)), float(np.max(final_spreads))
    return float("nan"), float("nan")


def plot_dataset_pair(dataset: str,
                      novar_runs: list[dict],
                      baseline_runs: list[dict],
                      out_path: Path) -> dict:
    """Two-panel figure (novar | baseline) for one dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)

    s_novar = plot_one_panel(
        axes[0], novar_runs,
        title=f"{dataset.upper()}   α_curv = 0.0  (no variance reg)",
        show_legend=True,
    )
    s_base = plot_one_panel(
        axes[1], baseline_runs,
        title=f"{dataset.upper()}   α_curv = 0.2  (default)",
        show_legend=False,
    )

    fig.suptitle(
        f"Per-domain curvature trajectories — {dataset.upper()}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[traj] {dataset}: novar spread={s_novar[0]:.4f} (max {s_novar[1]:.4f}),"
          f"  baseline spread={s_base[0]:.4f} (max {s_base[1]:.4f})")
    return {"dataset": dataset,
            "novar_spread_mean":    s_novar[0], "novar_spread_max":    s_novar[1],
            "baseline_spread_mean": s_base[0],  "baseline_spread_max": s_base[1]}


def plot_grid_figure(all_data: list[tuple], out_path: Path) -> None:
    """One mega-figure: 4 datasets × 2 conditions."""
    fig, axes = plt.subplots(len(all_data), 2, figsize=(8.4, 2.6 * len(all_data)),
                              sharey=False)
    if len(all_data) == 1:
        axes = axes.reshape(1, 2)
    for row, (dataset, novar, base) in enumerate(all_data):
        plot_one_panel(axes[row, 0], novar,
                       title=f"{dataset.upper()}   α_curv = 0.0",
                       show_legend=(row == 0))
        plot_one_panel(axes[row, 1], base,
                       title=f"{dataset.upper()}   α_curv = 0.2",
                       show_legend=False)
    fig.suptitle("Per-domain curvature trajectories: variance reg vs. none",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[traj] grid → {out_path}")


def write_summary_csv(rows: Iterable[dict], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[traj] summary → {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(ROOT / "outputs"))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    summary = []
    grid_data = []
    for ds in args.datasets:
        novar = collect_runs(NOVAR_PATH, ds)
        base  = collect_runs(BASELINE_PATH, ds)
        if not novar and not base:
            print(f"[traj] {ds}: NO data on either side — skipping")
            continue
        if not novar:
            print(f"[traj] {ds}: novar missing (have baseline only)")
        if not base:
            print(f"[traj] {ds}: baseline missing (have novar only) — "
                  "(Batch 3 not finished yet?)")
        # Even if one side is empty, still emit the partial figure.
        per_path = out_dir / f"curvature_trajectories_{ds}.pdf"
        row = plot_dataset_pair(ds, novar, base, per_path)
        summary.append(row)
        grid_data.append((ds, novar, base))

    if grid_data:
        plot_grid_figure(grid_data, out_dir / "curvature_trajectories.pdf")
        write_summary_csv(summary, out_dir / "curvature_summary.csv")

    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
