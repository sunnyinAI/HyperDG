#!/usr/bin/env python3
"""Generate placeholder CSV/NPY data for figure scripts.

This utility produces minimal, self-consistent datasets so the plotting
pipelines can run end-to-end when real experiment outputs are not yet ready.

Usage:
    python scripts/generate_dummy_figure_data.py --outdir data

Outputs:
  curvature_variance.csv
  loss_flatland.npy, loss_fedgeorefine.npy
  domain_accuracy.csv
  curvature_distribution.csv
  baseline_train.npy, baseline_domains.npy
  ours_train.npy, ours_domains.npy
  (optional test embeddings if --with-test is set)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gen_curvature_variance(path: Path, rounds: int = 50) -> None:
    x = np.arange(rounds)
    flat_mean = 0.45 * np.exp(-0.01 * x) + 0.05 * np.sin(0.3 * x) + 0.05
    ours_mean = 0.45 * np.exp(-0.05 * x) + 0.02
    df = pd.DataFrame(
        {
            "round": x,
            "flatland_mean": flat_mean,
            "flatland_std": 0.05 + 0.01 * np.random.rand(rounds),
            "fedgeorefine_mean": ours_mean,
            "fedgeorefine_std": 0.03 + 0.005 * np.random.rand(rounds),
        }
    )
    df.to_csv(path, index=False)


def gen_loss_surfaces(base_path: Path, ours_path: Path, grid: int = 30) -> None:
    x = np.linspace(-1.0, 1.0, grid)
    y = np.linspace(-1.0, 1.0, grid)
    X, Y = np.meshgrid(x, y)
    Z_base = 0.8 + 3 * (X**2 + 1.2 * Y**2) + 0.3 * np.sin(3 * X) * np.cos(2 * Y)
    Z_ours = 0.6 + 1.2 * (0.7 * X**2 + 0.5 * Y**2) + 0.15 * np.sin(2 * X) * np.cos(2 * Y)
    np.save(base_path, Z_base.astype(np.float32))
    np.save(ours_path, Z_ours.astype(np.float32))


def gen_domain_accuracy(path: Path) -> None:
    rows = [
        ("VLCS", "Caltech", 62.1, 1.6, 70.2, 1.0),
        ("VLCS", "LabelMe", 58.4, 1.9, 68.5, 1.2),
        ("VLCS", "SUN", 55.9, 1.8, 66.8, 1.3),
        ("VLCS", "VOC", 64.5, 1.4, 72.4, 0.9),
        ("PACS", "Art", 75.2, 1.0, 81.6, 0.8),
        ("PACS", "Cartoon", 78.6, 0.9, 84.1, 0.7),
        ("PACS", "Photo", 83.4, 0.7, 87.9, 0.5),
        ("PACS", "Sketch", 70.5, 1.2, 79.8, 1.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "domain",
            "flatland_mean",
            "flatland_std",
            "fedgeorefine_mean",
            "fedgeorefine_std",
        ],
    )
    df.to_csv(path, index=False)


def gen_curvature_distribution(path: Path, n: int = 400) -> None:
    flat = np.random.normal(loc=0.35, scale=0.08, size=n)
    ours = np.random.normal(loc=0.22, scale=0.04, size=n)
    df = pd.DataFrame(
        {
            "method": ["FlatLand"] * n + ["FedGeoRefine"] * n,
            "curvature": np.concatenate([flat, ours]),
        }
    )
    df.to_csv(path, index=False)


def gen_embeddings(train_path: Path, domain_path: Path, n_samples: int = 800, n_domains: int = 4) -> None:
    rng = np.random.default_rng(42)
    domains = np.arange(n_domains)
    labels = rng.choice(domains, size=n_samples)
    centers = rng.standard_normal((n_domains, 16)) * 0.5
    embeddings = centers[labels] + 0.4 * rng.standard_normal((n_samples, 16))
    np.save(train_path, embeddings.astype(np.float32))
    np.save(domain_path, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate placeholder data for figure plots.")
    parser.add_argument("--outdir", type=Path, default=Path("data"), help="Directory to store generated files.")
    parser.add_argument("--with-test", action="store_true", help="Also generate test embeddings.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    gen_curvature_variance(args.outdir / "curvature_variance.csv")
    gen_loss_surfaces(args.outdir / "loss_flatland.npy", args.outdir / "loss_fedgeorefine.npy")
    gen_domain_accuracy(args.outdir / "domain_accuracy.csv")
    gen_curvature_distribution(args.outdir / "curvature_distribution.csv")

    gen_embeddings(args.outdir / "baseline_train.npy", args.outdir / "baseline_domains.npy")
    gen_embeddings(args.outdir / "ours_train.npy", args.outdir / "ours_domains.npy")

    if args.with_test:
        gen_embeddings(args.outdir / "baseline_test.npy", args.outdir / "baseline_domains_test.npy")
        gen_embeddings(args.outdir / "ours_test.npy", args.outdir / "ours_domains_test.npy")

    print(f"[OK] Dummy data written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
