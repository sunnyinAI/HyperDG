#!/usr/bin/env python3
"""Aggregate repeated timing CSVs produced by measure_training_time.py."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline", default="ERM")
    parser.add_argument("--per-run-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for repeat, path in enumerate(sorted(args.inputs)):
        with path.open(newline="") as handle:
            current = list(csv.DictReader(handle))
        baseline = next(
            float(row["ms_per_step"])
            for row in current
            if row["algorithm"] == args.baseline
        )
        for row in current:
            rows.append(
                {
                    "repeat": repeat,
                    "algorithm": row["algorithm"],
                    "ms_per_step": float(row["ms_per_step"]),
                    "peak_memory_mb": float(row["peak_memory_mb"]),
                    "relative_to_erm": float(row["ms_per_step"]) / baseline,
                }
            )

    args.per_run_output.parent.mkdir(parents=True, exist_ok=True)
    with args.per_run_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    algorithms = sorted({row["algorithm"] for row in rows})
    summary = []
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        times = [float(row["ms_per_step"]) for row in selected]
        memories = [float(row["peak_memory_mb"]) for row in selected]
        relatives = [float(row["relative_to_erm"]) for row in selected]
        summary.append(
            {
                "algorithm": algorithm,
                "n_repeats": len(selected),
                "ms_per_step_mean": statistics.mean(times),
                "ms_per_step_std": statistics.stdev(times),
                "peak_memory_mb_mean": statistics.mean(memories),
                "peak_memory_mb_std": statistics.stdev(memories),
                "relative_to_erm_mean": statistics.mean(relatives),
                "relative_to_erm_std": statistics.stdev(relatives),
            }
        )

    with args.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    print(
        "Algorithm   repeats    ms/step (mean +/- SD)    "
        "peak MiB (mean +/- SD)    relative to ERM"
    )
    for row in summary:
        print(
            f"{row['algorithm']:<12}{row['n_repeats']:<11}"
            f"{row['ms_per_step_mean']:7.2f} +/- {row['ms_per_step_std']:6.2f}    "
            f"{row['peak_memory_mb_mean']:7.1f} +/- {row['peak_memory_mb_std']:6.1f}    "
            f"{row['relative_to_erm_mean']:.3f} +/- "
            f"{row['relative_to_erm_std']:.3f}"
        )


if __name__ == "__main__":
    main()
