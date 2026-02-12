#!/usr/bin/env python3
"""Summarize ablation experiment metrics into a concise table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing exp*/metrics.jsonl files "
        "(e.g. experiments/ablations/hyperdg_components).",
    )
    parser.add_argument(
        "--env-names",
        nargs="*",
        default=None,
        help="Optional list of environment labels (e.g. Art Clipart Product RealWorld). "
        "Defaults to env0, env1, ... order.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=None,
        help="Optional directory with exp*.yaml configs to read 'notes'/descriptions from.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics.jsonl",
        help="Metrics filename inside each exp directory (default: metrics.jsonl).",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "avg"],
        default="name",
        help="Order rows alphabetically (name) or by decreasing avg accuracy.",
    )
    return parser.parse_args()


def load_latest_metrics(path: Path) -> Dict | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def load_description(config_dir: Path | None, exp_name: str) -> str:
    if not config_dir:
        return ""
    cfg_path = config_dir / f"{exp_name}.yaml"
    if not cfg_path.exists():
        return ""
    try:
        data = yaml.safe_load(cfg_path.read_text())
    except Exception:
        return ""
    return data.get("notes", "") or data.get("description", "")


def format_table(rows: Sequence[Dict], env_names: Sequence[str]) -> str:
    headers = ["Exp"] + list(env_names) + ["Avg", "Notes"]
    col_widths = [max(len(h), 8) for h in headers]

    def fmt(value: float) -> str:
        return f"{value * 100:.1f}%"

    table_lines = []
    header_line = " | ".join(h.ljust(col_widths[idx]) for idx, h in enumerate(headers))
    table_lines.append(header_line)
    table_lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        cells = [
            row["name"].ljust(col_widths[0]),
        ]
        for idx, env in enumerate(env_names, start=1):
            cells.append(fmt(row["env_accs"][idx - 1]).ljust(col_widths[idx]))
        cells.append(fmt(row["avg"]).ljust(col_widths[-2]))
        notes = row["notes"] or ""
        cells.append(notes.ljust(col_widths[-1]))
        table_lines.append(" | ".join(cells))
    return "\n".join(table_lines)


def main() -> None:
    args = parse_args()
    runs_root: Path = args.runs_root
    if not runs_root.exists():
        raise SystemExit(f"Runs directory not found: {runs_root}")

    exp_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    if not exp_dirs:
        raise SystemExit(f"No exp*/ directories under {runs_root}")

    rows: List[Dict] = []
    env_names: List[str] = []

    for exp_dir in exp_dirs:
        metrics_path = exp_dir / args.metrics_file
        metrics = load_latest_metrics(metrics_path)
        if metrics is None:
            continue
        exp_name = exp_dir.name
        env_accs: List[float] = []
        env_idx = 0
        while True:
            key = f"env{env_idx}_out_acc"
            if key not in metrics:
                break
            env_accs.append(metrics[key])
            env_idx += 1
        if not env_accs:
            continue
        if not env_names:
            if args.env_names:
                if len(args.env_names) != len(env_accs):
                    raise SystemExit(
                        "env-names length does not match metrics environments "
                        f"({len(args.env_names)} vs {len(env_accs)})."
                    )
                env_names = list(args.env_names)
            else:
                env_names = [f"env{i}" for i in range(len(env_accs))]
        avg = sum(env_accs) / len(env_accs)
        notes = load_description(args.config_root, exp_name) if args.config_root else ""
        rows.append(
            {
                "name": exp_name,
                "env_accs": env_accs,
                "avg": avg,
                "notes": notes,
            }
        )

    if not rows:
        raise SystemExit("No metrics loaded.")

    if args.sort == "avg":
        rows.sort(key=lambda r: r["avg"], reverse=True)
    else:
        rows.sort(key=lambda r: r["name"])

    table = format_table(rows, env_names)
    print(table)


if __name__ == "__main__":
    main()
