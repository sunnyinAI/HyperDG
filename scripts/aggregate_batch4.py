#!/usr/bin/env python3
"""Aggregate the controlled HyperDG study into publication-ready outputs.

Each condition contains four seeds by four leave-one-domain-out splits. The
analysis pairs conditions on (seed, split), reports effect sizes and 95% paired
confidence intervals, and applies Holm correction to the complete family of
12 variant-by-dataset comparisons.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DATASETS = ["pacs", "vlcs", "officehome", "terraincognita"]
CONDITIONS = {
    "Zero-penalty control": "erm",
    "Full HyperDG (alpha_var=0.2)": "baseline",
    "Curvature frozen (no learning)": "fixed",
    "No variance reg (alpha_var=0)": "novar",
}


def paths_for(condition: str, seed: int) -> Path:
    if seed == 0:
        return {
            "erm": EXPERIMENTS / "batch2" / "erm",
            "baseline": EXPERIMENTS / "batch3" / "baseline",
            "fixed": EXPERIMENTS / "batch1" / "fixed" / "frozen",
            "novar": EXPERIMENTS / "batch1" / "novar" / "a0",
        }[condition]
    return EXPERIMENTS / "batch4" / f"seed{seed}" / condition


def final_accuracy(metrics_path: Path, test_env: int) -> float | None:
    try:
        records = [line for line in metrics_path.read_text().splitlines() if line]
        return json.loads(records[-1]).get(f"env{test_env}_out_acc")
    except (OSError, IndexError, json.JSONDecodeError):
        return None


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
            except ValueError:
                continue
            value = final_accuracy(run_dir / "metrics.jsonl", split)
            if value is not None:
                out[dataset][(seed, split)] = value * 100
    return out


def paired_summary(xs: list[float], ys: list[float]) -> dict[str, float | int]:
    if len(xs) != len(ys):
        raise ValueError("Paired samples must have equal length.")
    n = len(xs)
    if n < 2:
        return {
            "n": n,
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_raw": float("nan"),
        }

    diffs = [x - y for x, y in zip(xs, ys)]
    mean_diff = statistics.mean(diffs)
    sem = float(stats.sem(diffs))
    half_width = 0.0 if sem == 0 else float(stats.t.ppf(0.975, n - 1) * sem)
    test = stats.ttest_rel(xs, ys)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_low": mean_diff - half_width,
        "ci_high": mean_diff + half_width,
        "p_raw": float(test.pvalue),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def format_cell(values: list[float]) -> str:
    return f"{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}"


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    cells = {label: collect(key) for label, key in CONDITIONS.items()}
    control_label = "Zero-penalty control"
    control = cells[control_label]

    effect_keys: list[tuple[str, str]] = []
    effects: dict[tuple[str, str], dict[str, float | int]] = {}
    for label, runs_by_dataset in cells.items():
        if label == control_label:
            continue
        for dataset in DATASETS:
            ours = runs_by_dataset.get(dataset, {})
            baseline = control.get(dataset, {})
            common = sorted(set(ours) & set(baseline))
            key = (label, dataset)
            effect_keys.append(key)
            effects[key] = paired_summary(
                [ours[pair] for pair in common],
                [baseline[pair] for pair in common],
            )

    adjusted = holm_adjust([float(effects[key]["p_raw"]) for key in effect_keys])
    for key, p_holm in zip(effect_keys, adjusted):
        effects[key]["p_holm"] = p_holm

    outputs = ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    per_run_path = outputs / "batch4_per_run.csv"
    with per_run_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["condition", "dataset", "seed", "split", "acc"])
        for label, runs_by_dataset in cells.items():
            for dataset, runs in runs_by_dataset.items():
                for (seed, split), accuracy in sorted(runs.items()):
                    writer.writerow([label, dataset, seed, split, f"{accuracy:.4f}"])

    summary_path = outputs / "batch4_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "condition",
                "dataset",
                "n_runs",
                "mean_acc",
                "std_acc",
                "mean_diff_vs_control",
                "ci95_low",
                "ci95_high",
                "p_vs_control_raw",
                "p_vs_control_holm_12",
            ]
        )
        for label, runs_by_dataset in cells.items():
            for dataset in DATASETS:
                values = list(runs_by_dataset.get(dataset, {}).values())
                if not values:
                    continue
                row = [
                    label,
                    dataset,
                    len(values),
                    f"{statistics.mean(values):.4f}",
                    f"{statistics.stdev(values):.4f}",
                ]
                if label == control_label:
                    row.extend([""] * 5)
                else:
                    effect = effects[(label, dataset)]
                    row.extend(
                        [
                            f"{float(effect['mean_diff']):.4f}",
                            f"{float(effect['ci_low']):.4f}",
                            f"{float(effect['ci_high']):.4f}",
                            f"{float(effect['p_raw']):.6f}",
                            f"{float(effect['p_holm']):.6f}",
                        ]
                    )
                writer.writerow(row)

    lines = [
        "# Controlled multi-seed results",
        "",
        "Mean +/- sample SD over 4 seeds x 4 held-out domains. Parentheses "
        "show raw/Holm-adjusted paired p-values versus the matched zero-penalty control.",
        "",
        "| Condition | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join(["---:"] * len(DATASETS)) + "|",
    ]
    for label, runs_by_dataset in cells.items():
        row = [label]
        for dataset in DATASETS:
            cell = format_cell(list(runs_by_dataset[dataset].values()))
            if label != control_label:
                effect = effects[(label, dataset)]
                cell += (
                    f" (p={format_p(float(effect['p_raw']))}/"
                    f"{format_p(float(effect['p_holm']))})"
                )
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    docs_path = ROOT / "docs" / "results_tables.md"
    docs_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote:\n  {per_run_path}\n  {summary_path}\n  {docs_path}")


if __name__ == "__main__":
    main()
