#!/usr/bin/env python3
"""
Convert raw experiment outputs into the standardized CSV/NPY files
used by the figure plotting scripts.

Expected directory layout
-------------------------
<run_dir>/
    curvature_rounds.npy        # shape: [num_rounds, num_clients]
    curvature_final.npy         # shape: [num_clients]
    domain_accuracy.csv         # columns: dataset,domain,accuracy,std
    embeddings_train.npy        # shape: [N_train, D]
    embeddings_domains.npy      # shape: [N_train] (ints or strings)
    embeddings_test.npy         # optional
    embeddings_domains_test.npy # optional
    loss_surface.npy            # optional 2D array (ΔK1, ΔK2 grid)

Usage
-----
python scripts/prepare_real_figure_data.py \
    --flatland-dir runs/flatland \
    --fedgeorefine-dir runs/fedgeorefine \
    --output-dir data

The script writes the following files into --output-dir:
    curvature_variance.csv
    loss_flatland.npy / loss_fedgeorefine.npy (if surfaces provided)
    domain_accuracy.csv
    curvature_distribution.csv
    baseline_train.npy / baseline_domains.npy
    baseline_test.npy / baseline_domains_test.npy (if provided)
    ours_train.npy / ours_domains.npy
    ours_test.npy / ours_domains_test.npy (if provided)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_required(path: Path, description: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return np.load(path, allow_pickle=True)


def curvature_variance(flat_curv: np.ndarray, ours_curv: np.ndarray) -> pd.DataFrame:
    if flat_curv.shape != ours_curv.shape:
        raise ValueError(f"Curvature matrices must match, got {flat_curv.shape} vs {ours_curv.shape}")
    rounds = np.arange(flat_curv.shape[0])

    df = pd.DataFrame(
        {
            "round": rounds,
            "flatland_mean": flat_curv.mean(axis=1),
            "flatland_std": flat_curv.std(axis=1),
            "fedgeorefine_mean": ours_curv.mean(axis=1),
            "fedgeorefine_std": ours_curv.std(axis=1),
        }
    )
    return df


def combine_accuracy(flat_csv: Path, ours_csv: Path) -> pd.DataFrame:
    flat = pd.read_csv(flat_csv)
    ours = pd.read_csv(ours_csv)
    required_cols = {"dataset", "domain", "accuracy"}
    if not required_cols.issubset(flat.columns) or not required_cols.issubset(ours.columns):
        raise ValueError("Domain accuracy CSVs must contain columns: dataset, domain, accuracy[, std]")

    merged = flat.merge(
        ours,
        on=["dataset", "domain"],
        suffixes=("_flatland", "_fedgeorefine"),
        how="outer",
    ).rename(
        columns={
            "accuracy_flatland": "flatland_mean",
            "std_flatland": "flatland_std",
            "accuracy_fedgeorefine": "fedgeorefine_mean",
            "std_fedgeorefine": "fedgeorefine_std",
        }
    )
    for col in ["flatland_std", "fedgeorefine_std"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[["dataset", "domain", "flatland_mean", "flatland_std", "fedgeorefine_mean", "fedgeorefine_std"]]
    merged.sort_values(["dataset", "domain"], inplace=True)
    return merged


def curvature_distribution(flat_final: np.ndarray, ours_final: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["FlatLand"] * len(flat_final) + ["FedGeoRefine"] * len(ours_final),
            "curvature": np.concatenate([flat_final, ours_final]),
        }
    )


def copy_embeddings(run_dir: Path, prefix: str, outdir: Path, with_test: bool = True) -> None:
    train = load_required(run_dir / "embeddings_train.npy", f"{prefix} embeddings_train.npy")
    domains = load_required(run_dir / "embeddings_domains.npy", f"{prefix} embeddings_domains.npy")
    np.save(outdir / f"{prefix}_train.npy", train)
    np.save(outdir / f"{prefix}_domains.npy", domains)

    if with_test:
        test_path = run_dir / "embeddings_test.npy"
        domains_test_path = run_dir / "embeddings_domains_test.npy"
        if test_path.exists() and domains_test_path.exists():
            np.save(outdir / f"{prefix}_test.npy", np.load(test_path, allow_pickle=True))
            np.save(outdir / f"{prefix}_domains_test.npy", np.load(domains_test_path, allow_pickle=True))


def maybe_copy_loss_surface(run_dir: Path, out_path: Path) -> None:
    surface = run_dir / "loss_surface.npy"
    if surface.exists():
        np.save(out_path, np.load(surface, allow_pickle=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real figure data from experiment outputs.")
    parser.add_argument("--flatland-dir", type=Path, required=True, help="Directory containing FlatLand run artifacts.")
    parser.add_argument(
        "--fedgeorefine-dir",
        type=Path,
        required=True,
        help="Directory containing FedGeoRefine run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Destination directory for figure-ready files.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Do not copy embedding files (useful if you generated custom t-SNE inputs already).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Curvature variance
    flat_curv_rounds = load_required(args.flatland_dir / "curvature_rounds.npy", "FlatLand curvature_rounds.npy")
    ours_curv_rounds = load_required(args.fedgeorefine_dir / "curvature_rounds.npy", "FedGeoRefine curvature_rounds.npy")
    curvature_variance(flat_curv_rounds, ours_curv_rounds).to_csv(
        args.output_dir / "curvature_variance.csv", index=False
    )

    # Curvature distribution
    flat_curv_final = load_required(args.flatland_dir / "curvature_final.npy", "FlatLand curvature_final.npy")
    ours_curv_final = load_required(args.fedgeorefine_dir / "curvature_final.npy", "FedGeoRefine curvature_final.npy")
    curvature_distribution(flat_curv_final, ours_curv_final).to_csv(
        args.output_dir / "curvature_distribution.csv", index=False
    )

    # Domain accuracy
    domain_df = combine_accuracy(
        args.flatland_dir / "domain_accuracy.csv",
        args.fedgeorefine_dir / "domain_accuracy.csv",
    )
    domain_df.to_csv(args.output_dir / "domain_accuracy.csv", index=False)

    # Loss surfaces (optional)
    maybe_copy_loss_surface(args.flatland_dir, args.output_dir / "loss_flatland.npy")
    maybe_copy_loss_surface(args.fedgeorefine_dir, args.output_dir / "loss_fedgeorefine.npy")

    # Embeddings (optional)
    if not args.skip_embeddings:
        copy_embeddings(args.flatland_dir, "baseline", args.output_dir)
        copy_embeddings(args.fedgeorefine_dir, "ours", args.output_dir)

    print(f"[OK] Figure data materialized in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
