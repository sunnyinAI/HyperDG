#!/usr/bin/env python3
"""Legacy exploratory visualization of the HyperDG penalty grid.

The built-in grid changes several coefficients between configurations and is
not a one-factor sensitivity study. It also does not establish stability across
datasets. The revised paper therefore excludes this figure from its evidence.

The script reads metrics.jsonl files produced by run_ablation_grid.py
(hyperdg_penalty_sweep schedule) and generates a sensitivity figure showing
accuracy vs. each hyperparameter, one subplot per hparam.

Usage:
    # 1. Run the penalty sweep first:
    python scripts/run_ablation_grid.py \
        --base-config configs/training/resnet50_hyperdg_officehome.yaml \
        --schedule hyperdg_penalty_sweep \
        --gpus 0,1,2,3

    # 2. Plot sensitivity from the resulting experiments:
    python scripts/plot_hparam_sensitivity.py \
        --sweep-root experiments/ablations/hyperdg_penalty_sweep \
        --num-envs 4 \
        --output outputs/hparam_sensitivity.pdf

Alternatively, pass a hand-crafted CSV:
    python scripts/plot_hparam_sensitivity.py \
        --csv sensitivity_data.csv \
        --output outputs/hparam_sensitivity.pdf

CSV format:
    hparam,value,dataset,accuracy
    lambda_align,0.2,OfficeHome,68.1
    lambda_align,0.5,OfficeHome,70.4
    ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Mapping of experiment name substrings -> hparam values for the built-in sweep.
# When sweeping the penalty grid (run_ablation_grid.py --schedule hyperdg_penalty_sweep),
# the experiment names encode the hyperparameter values in their suffixes, e.g.
# "exp10_full_alpha03_beta03_gamma002_deltas005".  We parse them here.
_SUFFIX_TO_HPARAMS = {
    "alpha03_beta03": {"alpha_curv": 0.3, "beta_align": 0.3, "gamma_sharp": 0.0},
    "alpha04_beta02": {"alpha_curv": 0.4, "beta_align": 0.2, "gamma_sharp": 0.0},
    "alpha03_beta03_gamma002": {"alpha_curv": 0.3, "beta_align": 0.3, "gamma_sharp": 0.02},
    "alpha04_beta02_gamma002": {"alpha_curv": 0.4, "beta_align": 0.2, "gamma_sharp": 0.02},
    "alpha03_beta03_gamma002_delta005": {"alpha_curv": 0.3, "beta_align": 0.3, "gamma_sharp": 0.02, "delta_local_flat": 0.05},
    "alpha04_beta02_gamma002_delta005": {"alpha_curv": 0.4, "beta_align": 0.2, "gamma_sharp": 0.02, "delta_local_flat": 0.05},
    "alpha03_beta03_gamma002_deltas005": {"alpha_curv": 0.3, "beta_align": 0.3, "gamma_sharp": 0.02, "delta_local_flat": 0.05, "delta_global_flat": 0.05},
    "alpha035_beta02_gamma002_deltas005": {"alpha_curv": 0.35, "beta_align": 0.2, "gamma_sharp": 0.02, "delta_local_flat": 0.05, "delta_global_flat": 0.05},
    "alpha04_beta02_gamma001_deltas002": {"alpha_curv": 0.4, "beta_align": 0.2, "gamma_sharp": 0.01, "delta_local_flat": 0.05, "delta_global_flat": 0.02},
}

# Friendly display names for the hyperparameters
_DISPLAY_NAMES = {
    "alpha_curv": r"$\lambda_{\mathrm{var}}$ (α_curv)",
    "beta_align": r"$\lambda_{\mathrm{align}}$ (β_align)",
    "gamma_sharp": r"$\lambda_{\mathrm{fb}}$ (γ_sharp)",
    "delta_local_flat": r"$\delta_{\mathrm{local}}$",
    "delta_global_flat": r"$\delta_{\mathrm{global}}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter sensitivity for HyperDG penalty weights."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sweep-root",
        type=str,
        help="Root directory produced by run_ablation_grid.py --schedule hyperdg_penalty_sweep.",
    )
    source.add_argument(
        "--csv",
        type=str,
        help="Pre-built CSV with columns: hparam,value,dataset,accuracy",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Legacy option retained for CLI compatibility.",
    )
    parser.add_argument(
        "--test-env",
        type=int,
        default=0,
        help="Held-out environment whose out accuracy is plotted (default: 0).",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default="OfficeHome",
        help="Dataset name label for the figure title.",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        nargs="*",
        default=None,
        help="Subset of hparams to plot (defaults to all found).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/hparam_sensitivity.pdf",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Figure size in inches (auto if omitted).",
    )
    return parser.parse_args()


def _load_latest(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _heldout_acc(record: Dict, test_env: int) -> Optional[float]:
    value = record.get(f"env{test_env}_out_acc")
    return float(value) * 100.0 if value is not None else None


def load_from_sweep(sweep_root: Path, test_env: int) -> List[Dict]:
    """
    Walk sweep_root, match experiment name suffixes to known hparam values,
    and return a list of {hparam, value, accuracy} records.
    """
    rows = []
    for exp_dir in sorted(sweep_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics = _load_latest(exp_dir / "metrics.jsonl")
        if metrics is None:
            continue
        acc = _heldout_acc(metrics, test_env)
        if acc is None:
            continue
        name = exp_dir.name
        matched = None
        for suffix, hparams in _SUFFIX_TO_HPARAMS.items():
            if suffix in name:
                matched = hparams
                break
        if matched is None:
            continue
        for hp, val in matched.items():
            rows.append({"hparam": hp, "value": val, "accuracy": acc})
    return rows


def load_from_csv(csv_path: Path) -> List[Dict]:
    rows = []
    lines = csv_path.read_text().strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        record = dict(zip(header, parts))
        rows.append({
            "hparam": record["hparam"],
            "value": float(record["value"]),
            "accuracy": float(record["accuracy"]),
        })
    return rows


def group_by_hparam(rows: List[Dict]) -> Dict[str, Tuple[List[float], List[float]]]:
    """Return {hparam: (values, accuracies)} sorted by value."""
    grouped: Dict[str, Dict[float, List[float]]] = {}
    for r in rows:
        hp = r["hparam"]
        v = float(r["value"])
        grouped.setdefault(hp, {}).setdefault(v, []).append(float(r["accuracy"]))
    result = {}
    for hp, val_map in grouped.items():
        vals = sorted(val_map.keys())
        accs = [float(np.mean(val_map[v])) for v in vals]
        stds = [float(np.std(val_map[v])) if len(val_map[v]) > 1 else 0.0 for v in vals]
        result[hp] = (vals, accs, stds)
    return result


def main() -> None:
    args = parse_args()

    mpl.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    if args.sweep_root:
        print(
            "WARNING: the built-in penalty grid is confounded because multiple "
            "coefficients change between configurations; use this plot for "
            "exploration only."
        )
        rows = load_from_sweep(Path(args.sweep_root), args.test_env)
    else:
        rows = load_from_csv(Path(args.csv))

    if not rows:
        raise SystemExit("No data found. Check --sweep-root or --csv path.")

    grouped = group_by_hparam(rows)

    # Filter to requested hparams
    if args.hparams:
        grouped = {k: v for k, v in grouped.items() if k in args.hparams}

    if not grouped:
        raise SystemExit("No matching hyperparameters found.")

    hparam_keys = sorted(grouped.keys())
    n = len(hparam_keys)
    figsize = tuple(args.figsize) if args.figsize else (4.0 * n, 3.8)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, hp in zip(axes, hparam_keys):
        vals, accs, stds = grouped[hp]
        x = np.array(vals)
        y = np.array(accs)
        err = np.array(stds)

        color = colors[hparam_keys.index(hp) % len(colors)]
        ax.errorbar(
            x, y,
            yerr=err if any(e > 0 for e in err) else None,
            marker="o",
            linewidth=2.0,
            markersize=6,
            capsize=4,
            color=color,
            elinewidth=1.2,
        )
        ax.set_xlabel(_DISPLAY_NAMES.get(hp, hp))
        ax.set_ylabel("Accuracy (%)" if ax is axes[0] else "")
        ax.grid(alpha=0.25, linestyle="--")

        # Shade region within 0.5% of best
        best = max(accs)
        ax.axhline(best, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axhspan(best - 0.5, best + 0.5, color="gray", alpha=0.08)
        ax.set_title(_DISPLAY_NAMES.get(hp, hp))

        # Annotate each point
        for xi, yi, si in zip(x, y, err):
            offset = 3 if yi >= best - 0.5 else -12
            ax.annotate(
                f"{yi:.1f}",
                xy=(xi, yi),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle(
        f"Hyperparameter sensitivity — {args.dataset_label}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved sensitivity figure to {out_path}")

    # Print summary table
    print(f"\n{'Hparam':<35} {'Value':>8} {'Acc (%)':>10} {'± Std':>8}")
    print("-" * 65)
    for hp in hparam_keys:
        vals, accs, stds = grouped[hp]
        for v, a, s in zip(vals, accs, stds):
            std_str = f"{s:.2f}" if s > 0 else "—"
            print(f"{_DISPLAY_NAMES.get(hp, hp):<35} {v:>8.3f} {a:>10.2f} {std_str:>8}")
    print()


if __name__ == "__main__":
    main()
