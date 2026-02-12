#!/usr/bin/env python3
"""Plot curvature variance vs accuracy for multiple datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curvature vs accuracy plot")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="List of metrics.jsonl files (one per dataset).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Dataset labels in the same order as metrics files.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        nargs="+",
        required=True,
        help="Number of held-out domains per dataset (one per metrics).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/curvature_vs_accuracy.png",
        help="Destination for the plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Curvature variance vs. DG accuracy",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Add inset scatter showing correlation.",
    )
    return parser.parse_args()


def load_summary(path: Path, num_envs: int) -> Dict[str, float]:
    steps: List[int] = []
    variances: List[float] = []
    acc_steps: List[int] = []
    accuracies: List[float] = []
    last_record: Dict[str, float] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step_idx = int(record.get("step", len(steps)))
            if "curvature_var" in record:
                steps.append(step_idx)
                variances.append(record["curvature_var"])
            env_accs = [record.get(f"env{i}_out_acc") for i in range(num_envs)]
            if all(acc is not None for acc in env_accs):
                mean_acc = float(sum(env_accs) / len(env_accs))
                acc_steps.append(step_idx)
                accuracies.append(mean_acc)
            last_record = record
    if last_record is None:
        raise ValueError(f"No metrics in {path}")
    if not accuracies:
        raise ValueError(f"No env accuracy entries found in {path}")
    final_var = variances[-1] if variances else float("nan")
    final_acc = accuracies[-1]
    return {
        "steps": steps,
        "variances": variances,
        "acc_steps": acc_steps,
        "accuracies": accuracies,
        "final_var": final_var,
        "accuracy": final_acc,
    }


def _add_trendline(ax, xs: np.ndarray, ys: np.ndarray, color: str, order: int = 2, linestyle: str = ":") -> None:
    if len(xs) < order + 1:
        return
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = min(order, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, deg=order)
    dense_x = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(
        dense_x,
        np.polyval(coeffs, dense_x),
        color=color,
        linestyle=linestyle,
        linewidth=1.2,
        alpha=0.6,
    )


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-darkgrid")
    mpl.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    paths = [Path(p) for p in args.metrics]
    summaries = [load_summary(p, n) for p, n in zip(paths, args.num_envs)]

    fig, axes = plt.subplots(1, len(paths), figsize=(4.7 * len(paths), 3.8), sharex=False)
    if len(paths) == 1:
        axes = [axes]

    VAR_COLOR = "#4C72B0"
    ACC_COLOR = "#C44E52"

    for idx, (ax, summary, label) in enumerate(zip(axes, summaries, args.labels)):
        step_arr = np.array(summary["steps"])
        var_arr = np.array(summary["variances"])
        acc_steps = np.array(summary["acc_steps"])
        acc_vals = 100.0 * np.array(summary["accuracies"])

        var_line = ax.plot(step_arr, var_arr, color=VAR_COLOR, linewidth=2.0, label="Var(K)")
        _add_trendline(ax, step_arr, var_arr, VAR_COLOR, order=3)
        ax.set_xlabel("Training iteration")
        ax.grid(alpha=0.25, linestyle="--")
        if idx == 0:
            ax.set_ylabel(r"Curvature variance  $\mathrm{Var}(K)$")

        acc_ax = ax.twinx()
        acc_line = acc_ax.plot(
            acc_steps,
            acc_vals,
            color=ACC_COLOR,
            linewidth=2.0,
            linestyle="--",
            label="DG accuracy",
        )
        _add_trendline(acc_ax, acc_steps, acc_vals, ACC_COLOR, order=2)
        acc_ax.set_ylabel("DG accuracy (%)")

        ax.set_title(
            f"{label} | Var(K)={summary['final_var']:.2e}, Acc={summary['accuracy']*100:.1f}%",
            fontsize=12,
        )

        lines = var_line + acc_line
        labels_combined = ["Var(K)", "Accuracy"]
        ax.legend(lines, labels_combined, loc="upper right")

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if args.scatter:
        inset_width = 0.18 * len(paths)
        inset_ax = fig.add_axes([0.72, 0.58, inset_width, 0.28])
        vars_arr = np.array([s["final_var"] for s in summaries])
        acc_arr = 100.0 * np.array([s["accuracy"] for s in summaries])
        inset_ax.scatter(vars_arr, acc_arr, c="#C44E52", s=45, edgecolors="white", linewidths=0.5)
        slope = np.polyfit(vars_arr, acc_arr, deg=1)
        xs = np.linspace(vars_arr.min(), vars_arr.max(), 100)
        inset_ax.plot(xs, slope[0] * xs + slope[1], color="#55A868", linewidth=1.5)
        inset_ax.set_xlabel("Var(K)", fontsize=8)
        inset_ax.set_ylabel("Accuracy", fontsize=8)
        inset_ax.tick_params(labelsize=7)
        r = np.corrcoef(vars_arr, acc_arr)[0, 1]
        inset_ax.set_title(f"Correlation   r={r:.2f}", fontsize=9)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
