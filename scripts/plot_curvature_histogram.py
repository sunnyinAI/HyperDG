"""Plot histogram of curvature values from checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curvature histograms from checkpoints.")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint paths.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each checkpoint (defaults to filenames).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/curvature_histogram.png",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
    )
    return parser.parse_args()


def extract_curvatures(checkpoint_path: Path) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    tensor = state_dict.get("log_curvatures")
    if tensor is None:
        raise ValueError(f"log_curvatures not found in {checkpoint_path}")
    curvatures = torch.nn.functional.softplus(tensor) + 1e-4
    return curvatures.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    labels = args.labels if args.labels else [Path(p).stem for p in args.checkpoints]

    plt.figure(figsize=(6.0, 4.0))
    for path_str, label in zip(args.checkpoints, labels):
        curvatures = extract_curvatures(Path(path_str))
        plt.hist(
            curvatures,
            bins=args.bins,
            density=True,
            alpha=0.45,
            label=label,
        )

    plt.xlabel("Curvature value $K$")
    plt.ylabel("Density")
    plt.title("Curvature distributions across domains")
    plt.legend()
    plt.grid(alpha=0.2, linestyle="--")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved curvature histogram to {output_path}")


if __name__ == "__main__":
    main()
