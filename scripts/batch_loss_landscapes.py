#!/usr/bin/env python3
"""Batch driver that materializes loss surfaces and figures for multiple datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_AXIS_LABELS = ["Curvature perturbation ΔK", "Embedding projection β"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch loss-surface + figure generator.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/figures/loss_landscape_jobs.yaml"),
        help="YAML manifest describing the jobs to run.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of job names to run.",
    )
    parser.add_argument(
        "--force-surfaces",
        action="store_true",
        help="Recompute .npy surfaces even if they already exist.",
    )
    parser.add_argument(
        "--force-figures",
        action="store_true",
        help="Regenerate figures even if the PNG/PDF files already exist.",
    )
    parser.add_argument("--grid-size", type=int, default=41, help="Default grid size for parameter sweeps.")
    parser.add_argument("--delta", type=float, default=1.5, help="Default perturbation half-width.")
    parser.add_argument("--env-mode", choices=["train", "test", "all"], default="test", help="Default env mode.")
    parser.add_argument("--max-samples", type=int, default=2048, help="Default max samples per environment.")
    parser.add_argument("--batch-size", type=int, default=128, help="Default batch size for evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Default RNG seed for direction sampling.")
    parser.add_argument("--dpi-png", type=int, default=300, help="Default DPI for PNG outputs.")
    parser.add_argument("--dpi-pdf", type=int, default=600, help="Default DPI for PDF outputs.")
    return parser.parse_args()


def run_command(cmd: List[str]) -> None:
    print(f"[batch] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_jobs(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = yaml.safe_load(path.read_text())
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Manifest must define a top-level 'jobs' list.")
    return jobs


def ensure_surface(job_name: str, role: str, spec: Dict, args: argparse.Namespace) -> bool:
    required_keys = {"config", "checkpoint", "output"}
    if not required_keys.issubset(spec):
        missing = required_keys - set(spec)
        raise ValueError(f"[{job_name}/{role}] Missing required keys: {missing}")

    ckpt = Path(spec["checkpoint"])
    if not ckpt.exists():
        print(f"[batch] Skipping {job_name}/{role}: checkpoint not found at {ckpt}")
        return False

    surface_path = Path(spec["output"])
    if surface_path.exists() and not args.force_surfaces:
        print(f"[batch] [{job_name}] Reusing cached surface for {role}: {surface_path}")
        return True

    cmd = [
        sys.executable,
        "scripts/compute_loss_surface.py",
        "--config",
        spec["config"],
        "--checkpoint",
        str(ckpt),
        "--output",
        str(surface_path),
        "--grid-size",
        str(spec.get("grid_size", args.grid_size)),
        "--delta",
        str(spec.get("delta", args.delta)),
        "--max-samples",
        str(spec.get("max_samples", args.max_samples)),
        "--env-mode",
        spec.get("env_mode", args.env_mode),
        "--batch-size",
        str(spec.get("batch_size", args.batch_size)),
        "--seed",
        str(spec.get("seed", args.seed)),
    ]
    run_command(cmd)
    return True


def render_figures(job: Dict, args: argparse.Namespace) -> None:
    outputs = job.get("outputs")
    if not outputs:
        return

    baseline_path = Path(job["baseline"]["output"])
    ours_path = Path(job["ours"]["output"])
    if not baseline_path.exists() or not ours_path.exists():
        print(f"[batch] Skipping figure for {job['name']}: missing surfaces.")
        return

    axis_labels = job.get("axis_labels", DEFAULT_AXIS_LABELS)
    if len(axis_labels) != 2:
        raise ValueError(f"[{job['name']}] axis_labels must have exactly two entries.")

    base_label = job["baseline"].get("label", "Baseline")
    ours_label = job["ours"].get("label", "Ours")
    contour = job.get("contour", True)
    annotate_min = job.get("annotate_min", True)
    delta = outputs.get("delta", job["baseline"].get("delta", args.delta))

    base_cmd = [
        sys.executable,
        "figures/plot_loss_landscape.py",
        "--baseline-npy",
        str(baseline_path),
        "--ours-npy",
        str(ours_path),
        "--axis-labels",
        axis_labels[0],
        axis_labels[1],
        "--title-baseline",
        base_label,
        "--title-ours",
        ours_label,
        "--delta",
        str(delta),
    ]
    if contour:
        base_cmd.append("--contour")
    if annotate_min:
        base_cmd.append("--annotate-min")

    for key, dpi_default in (("png", args.dpi_png), ("pdf", args.dpi_pdf)):
        out_path = outputs.get(key)
        if not out_path:
            continue
        out_file = Path(out_path)
        if out_file.exists() and not args.force_figures:
            print(f"[batch] [{job['name']}] Reusing figure {out_file}")
            continue
        cmd = list(base_cmd)
        cmd.extend(["--output", str(out_file), "--dpi", str(outputs.get(f"dpi_{key}", dpi_default))])
        run_command(cmd)


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args.manifest)
    requested = set(args.only) if args.only else None

    completed = []
    skipped = []

    for job in jobs:
        name = job.get("name")
        if not name:
            print("[batch] Skipping unnamed job entry.")
            continue
        if requested and name not in requested:
            continue

        print(f"[batch] === {name} ===")
        baseline_ok = ensure_surface(name, "baseline", job["baseline"], args)
        ours_ok = ensure_surface(name, "ours", job["ours"], args)
        if baseline_ok and ours_ok:
            render_figures(job, args)
            completed.append(name)
        else:
            skipped.append(name)

    print(f"[batch] Completed jobs: {completed}")
    if skipped:
        print(f"[batch] Skipped jobs (missing checkpoints or errors): {skipped}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
