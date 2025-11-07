#!/usr/bin/env python3
"""
Optional Figure 6: t-SNE embeddings comparison.

Inputs:
  --baseline-emb npy file containing shape (N, D) embeddings.
  --baseline-domain npy file containing corresponding domain labels (strings or ints).
  --ours-emb npy file containing shape (M, D) embeddings.
  --ours-domain npy file containing corresponding domain labels.

The script applies t-SNE separately to each method and renders a 2×2 grid:
  (a) Baseline train embeddings (colored by domain)
  (b) Baseline test embeddings (if provided via --baseline-emb-test / --baseline-domain-test)
  (c) FedGeoRefine train embeddings
  (d) FedGeoRefine test embeddings

If no test embeddings are supplied, the bottom row is omitted.

Usage example:
    python figures/plot_tsne_embeddings.py \
        --baseline-emb data/baseline_train.npy \
        --baseline-domain data/baseline_domains.npy \
        --ours-emb data/ours_train.npy \
        --ours-domain data/ours_domains.npy \
        --perplexity 35 \
        --output figures/fig6_tsne.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

PALETTE = sns.color_palette("Set2", 8)


def _fit_tsne(embeddings, perplexity, random_state):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def _scatter(ax, coords, domains, title):
    unique_domains = np.unique(domains)
    colors = {dom: PALETTE[i % len(PALETTE)] for i, dom in enumerate(unique_domains)}
    for dom in unique_domains:
        mask = domains == dom
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=str(dom),
            s=15,
            alpha=0.8,
            color=colors[dom],
            edgecolor="none",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best", fontsize=8, markerscale=1.3)


def _load(path: Path):
    arr = np.load(path)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported array shape {arr.shape} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot t-SNE embeddings colored by domain.")
    parser.add_argument("--baseline-emb", type=Path, required=True, help="Baseline train embeddings (.npy).")
    parser.add_argument("--baseline-domain", type=Path, required=True, help="Baseline domain labels (.npy).")
    parser.add_argument("--ours-emb", type=Path, required=True, help="FedGeoRefine train embeddings (.npy).")
    parser.add_argument("--ours-domain", type=Path, required=True, help="FedGeoRefine domain labels (.npy).")
    parser.add_argument("--baseline-emb-test", type=Path, help="Baseline test embeddings (.npy).")
    parser.add_argument("--baseline-domain-test", type=Path, help="Baseline test domain labels (.npy).")
    parser.add_argument("--ours-emb-test", type=Path, help="FedGeoRefine test embeddings (.npy).")
    parser.add_argument("--ours-domain-test", type=Path, help="FedGeoRefine test domain labels (.npy).")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for t-SNE.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig6_tsne.pdf"),
        help="Output filename.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    baseline_train = _load(args.baseline_emb)
    baseline_train_dom = _load(args.baseline_domain)
    ours_train = _load(args.ours_emb)
    ours_train_dom = _load(args.ours_domain)

    rows = 2 if args.baseline_emb_test and args.ours_emb_test else 1
    fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows), squeeze=False)

    baseline_coords = _fit_tsne(baseline_train, args.perplexity, args.random_state)
    _scatter(axes[0, 0], baseline_coords, baseline_train_dom, "FlatLand (train)")

    ours_coords = _fit_tsne(ours_train, args.perplexity, args.random_state)
    _scatter(axes[0, 1], ours_coords, ours_train_dom, "FedGeoRefine (train)")

    if rows == 2:
        baseline_test = _load(args.baseline_emb_test)
        baseline_test_dom = _load(args.baseline_domain_test)
        ours_test = _load(args.ours_emb_test)
        ours_test_dom = _load(args.ours_domain_test)

        baseline_test_coords = _fit_tsne(baseline_test, args.perplexity, args.random_state)
        _scatter(axes[1, 0], baseline_test_coords, baseline_test_dom, "FlatLand (test)")

        ours_test_coords = _fit_tsne(ours_test, args.perplexity, args.random_state)
        _scatter(axes[1, 1], ours_test_coords, ours_test_dom, "FedGeoRefine (test)")

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"[OK] Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
