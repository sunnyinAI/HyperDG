#!/usr/bin/env python3
"""Aggregate split metrics into mean/std for table reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HyperDG results.")
    parser.add_argument("--root", type=str, required=True, help="Directory containing split*/metrics.jsonl")
    parser.add_argument("--num-envs", type=int, required=True, help="Number of evaluation domains.")
    parser.add_argument("--verbose", action="store_true", help="Print per-split accuracies.")
    return parser.parse_args()


def load_latest_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    splits = sorted(root.glob("split*/metrics.jsonl"))
    if not splits:
        raise SystemExit(f"No split metrics found under {root}")

    avgs: List[float] = []
    for metrics_path in splits:
        data = load_latest_metrics(metrics_path)
        if data is None:
            continue
        env_accs = [data.get(f"env{i}_out_acc") for i in range(args.num_envs)]
        if any(acc is None for acc in env_accs):
            continue
        avg = sum(env_accs) / len(env_accs)
        avgs.append(avg)
        if args.verbose:
            pretty = " ".join(f"env{i+1}={acc*100:.2f}%" for i, acc in enumerate(env_accs))
            print(f"{metrics_path.parent.name}: {pretty} avg={avg*100:.2f}%")

    if not avgs:
        raise SystemExit("No complete splits to summarize.")

    mean = sum(avgs) / len(avgs)
    std = statistics.stdev(avgs) if len(avgs) > 1 else 0.0
    print(f"Splits: {len(avgs)}")
    print(f"Mean accuracy: {mean*100:.2f}%")
    print(f"Std: {std*100:.2f}%")


if __name__ == "__main__":
    main()
