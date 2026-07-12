#!/usr/bin/env python3
"""Plot paired effects for the controlled HyperDG study."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["pacs", "vlcs", "officehome", "terraincognita"]
DATASET_LABELS = {
    "pacs": "PACS",
    "vlcs": "VLCS",
    "officehome": "OfficeHome",
    "terraincognita": "TerraIncognita",
}


def paths_for(condition: str, seed: int) -> Path:
    if seed == 0:
        return {
            "erm": ROOT / "experiments" / "batch2" / "erm",
            "baseline": ROOT / "experiments" / "batch3" / "baseline",
            "fixed": ROOT / "experiments" / "batch1" / "fixed" / "frozen",
            "novar": ROOT / "experiments" / "batch1" / "novar" / "a0",
        }[condition]
    return ROOT / "experiments" / "batch4" / f"seed{seed}" / condition


def collect(condition: str) -> dict[str, dict[tuple[int, int], float]]:
    out: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for seed in range(4):
        base = paths_for(condition, seed)
        if not base.exists():
            continue
        for run_dir in base.iterdir():
            if "_split" not in run_dir.name:
                continue
            dataset, split_text = run_dir.name.rsplit("_split", 1)
            if dataset not in DATASETS:
                continue
            try:
                split = int(split_text)
                records = [
                    line
                    for line in (run_dir / "metrics.jsonl").read_text().splitlines()
                    if line
                ]
                value = json.loads(records[-1])[f"env{split}_out_acc"] * 100
            except (ValueError, OSError, IndexError, KeyError, json.JSONDecodeError):
                continue
            out[dataset][(seed, split)] = value
    return out


def paired_effect(xs: list[float], ys: list[float]) -> tuple[float, float, float, int]:
    diffs = [x - y for x, y in zip(xs, ys)]
    n = len(diffs)
    mean = statistics.mean(diffs)
    sem = float(stats.sem(diffs))
    half_width = float(stats.t.ppf(0.975, n - 1) * sem) if sem else 0.0
    p_raw = float(stats.ttest_rel(xs, ys).pvalue)
    return mean, half_width, p_raw, n


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(
            running_max,
            min(1.0, (len(p_values) - rank) * p_values[idx]),
        )
        adjusted[idx] = running_max
    return adjusted


def main() -> None:
    control = collect("erm")
    conditions = [
        ("Full objective", collect("baseline")),
        ("Curvature frozen", collect("fixed")),
        ("No variance penalty", collect("novar")),
    ]
    colors = {
        "Full objective": "#C44E52",
        "Curvature frozen": "#4C72B0",
        "No variance penalty": "#55A868",
    }

    rows: list[list[object]] = []
    for dataset in DATASETS:
        for label, runs in conditions:
            common = sorted(set(runs[dataset]) & set(control[dataset]))
            mean, ci, p_raw, n = paired_effect(
                [runs[dataset][pair] for pair in common],
                [control[dataset][pair] for pair in common],
            )
            rows.append([dataset, label, mean, ci, p_raw, n])
    for row, p_holm in zip(rows, holm_adjust([float(row[4]) for row in rows])):
        row.append(p_holm)

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    y = 0.0
    ticks, tick_labels = [], []
    plotted = []
    for dataset_index, dataset in enumerate(reversed(DATASETS)):
        dataset_rows = [row for row in rows if row[0] == dataset]
        if dataset_index % 2 == 0:
            ax.axhspan(y - 0.5, y + 2.5, color="0.97", zorder=0)
        ticks.append(y + 1)
        tick_labels.append(DATASET_LABELS[dataset])
        for row in dataset_rows:
            _, label, mean, ci, p_raw, n, p_holm = row
            significant = float(p_holm) <= 0.05
            ax.errorbar(
                float(mean),
                y,
                xerr=float(ci),
                fmt="o",
                color=colors[str(label)],
                capsize=4,
                markersize=7,
                markeredgecolor="black" if significant else "none",
                zorder=3,
            )
            ax.text(
                float(mean) + float(ci) + 0.15,
                y,
                f"{float(mean):+.2f} [{float(mean)-float(ci):+.2f}, "
                f"{float(mean)+float(ci):+.2f}]  "
                f"p={float(p_raw):.3f}, Holm={float(p_holm):.3f}",
                va="center",
                fontsize=7.2,
            )
            plotted.append(row)
            y += 1
        y += 0.55

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy difference relative to matched zero-penalty control (pp)")
    ax.set_title(
        "Controlled paired effects (4 seeds x 4 held-out domains)\n"
        "bars: unadjusted 95% paired t intervals; Holm correction across 12 tests"
    )
    ax.grid(axis="x", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D([], [], color=color, marker="o", linestyle="", label=label)
        for label, color in colors.items()
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=8,
    )
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 1.2)
    fig.tight_layout()

    output = ROOT / "outputs" / "forest_erm_vs_hyperdg.pdf"
    fig.savefig(output, bbox_inches="tight", dpi=200)
    fig.savefig(ROOT / "HyperDG_TMLR" / "fig_forest_erm_vs_hyperdg.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {output}")
    print("\nDataset           Condition             Delta      CI half    raw p    Holm p")
    for dataset, label, mean, ci, p_raw, n, p_holm in plotted:
        print(
            f"{str(dataset):<18}{str(label):<22}{float(mean):+7.3f}  "
            f"+/-{float(ci):6.3f}  {float(p_raw):7.4f}  {float(p_holm):7.4f}"
        )


if __name__ == "__main__":
    main()
