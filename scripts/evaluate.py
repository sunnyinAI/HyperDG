"""Utility script to summarize metrics.jsonl outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics.")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics.jsonl produced during training.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=None,
        help="Metric keys to track (defaults to *_out_acc).",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    records = load_metrics(metrics_path)

    if not records:
        print("No records found.")
        return

    keys = args.keys or [k for k in records[0].keys() if k.endswith("_out_acc")]
    summary = {}
    for key in keys:
        best = max(records, key=lambda r: r.get(key, float("-inf")))
        summary[key] = {"best": best.get(key), "step": best.get("step")}

    print(f"Loaded {len(records)} records from {metrics_path}")
    for key, value in summary.items():
        print(f"{key}: {value['best']:.4f} @ step {value['step']}")


if __name__ == "__main__":
    main()
