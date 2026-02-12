#!/usr/bin/env python3
"""Plot validation accuracies over training for ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot env accuracies vs. steps.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of experiment directories (each containing metrics.jsonl).",
    )
    parser.add_argument(
        "--env-names",
        nargs="*",
        default=None,
        help="Optional labels for env*_out_acc curves (e.g. Art Clipart ...).",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics.jsonl",
        help="Metrics filename inside each run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/ablations"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--moving-average",
        type=int,
        default=1,
        help="Window size for simple moving average smoothing (default 1 = no smoothing).",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.5,
        help="Lower y-axis bound (default 0.5).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Upper y-axis bound (default 1.0).",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    data = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    if not data:
        raise ValueError(f"No metrics lines found in {path}")
    return data


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    smoothed = []
    acc = 0.0
    buf: List[float] = []
    for val in values:
        buf.append(val)
        acc += val
        if len(buf) > window:
            acc -= buf.pop(0)
        smoothed.append(acc / len(buf))
    return smoothed


def plot_run(run_dir: Path, args: argparse.Namespace) -> Path:
    metrics_path = run_dir / args.metrics_file
    data = load_metrics(metrics_path)
    env_keys = [k for k in data[-1].keys() if k.startswith("env") and k.endswith("_out_acc")]
    env_keys.sort()

    if args.env_names and len(args.env_names) != len(env_keys):
        raise SystemExit(
            f"env-names length ({len(args.env_names)}) does not match detected envs ({len(env_keys)})."
        )
    env_labels = args.env_names or env_keys

    steps = [entry["step"] for entry in data]
    curves = {label: [] for label in env_labels}
    for entry in data:
        for label, key in zip(env_labels, env_keys):
            curves[label].append(entry.get(key, float("nan")))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in env_labels:
        values = moving_average(curves[label], args.moving_average)
        ax.plot(steps, values, label=label)

    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Out-of-domain accuracy")
    ax.set_title(run_dir.name)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = output_dir / f"{run_dir.name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    run_dirs = [Path(run) for run in args.runs]
    for run_dir in run_dirs:
        path = plot_run(run_dir, args)
        print(f"[OK] Saved {path}")


if __name__ == "__main__":
    main()
