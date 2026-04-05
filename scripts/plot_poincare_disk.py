#!/usr/bin/env python3
"""Poincaré disk visualization of per-domain representations.

Approach (matches reviewer hQnG's explicit request):
  1. Extract backbone features for a subset of samples per domain.
  2. Apply the learnt per-domain tangent embedding (feats / sqrt(K_i)).
  3. Reduce high-d tangent vectors to 2D via PCA (linear, geometry-preserving).
  4. L2-normalise to unit norm, scale to 0.85 to stay in stable cosh range.
  5. Map 2D tangent vectors to Lorentz coords via exp_map at fixed K=1 (curvature=-1).
  6. Apply stereographic projection (Lorentz -> Poincaré disk).
  7. Plot each domain as a separate panel on the Poincaré disk.

This gives a *direct* Poincaré disk representation at fixed K=1 (no UMAP),
matching the standard approach in hyperbolic learning papers cited by the reviewer.

Usage:
  python scripts/plot_poincare_disk.py \
      --config  configs/reviewer_hQnG/meru/pacs_split0.yaml \
      --ckpt    experiments/reviewer_hQnG/meru/pacs_split0/model_last.pt \
      --out     outputs/poincare_meru_pacs.pdf \
      --n_samples 300 \
      --device  cuda:2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "domainbed"))

from domainbed import datasets as db_datasets
from domainbed import hparams_registry
from domainbed.lib import misc as db_misc
from domainbed.lib.fast_data_loader import FastDataLoader

from sftdg.algorithms import build_algorithm
from sftdg.utils import load_yaml_config, set_seed


# ── Lorentz / Poincaré math ───────────────────────────────────────────────────

def exp_map_origin_2d(v2: np.ndarray, curv: float = 1.0) -> np.ndarray:
    """Exp map from origin on L^2 at curvature magnitude |K|=curv.

    v2  : (N, 2) 2-D tangent vectors
    Returns (N, 3) Lorentz coordinates [x0, x1, x2]
    satisfying  -x0^2 + x1^2 + x2^2 = -1/curv
    """
    sqrt_c = float(curv) ** 0.5
    norm = np.linalg.norm(v2, axis=1, keepdims=True).clip(min=1e-8)
    x0 = np.cosh(sqrt_c * norm)
    xi = np.sinh(sqrt_c * norm) / norm * v2
    return np.concatenate([x0, xi], axis=1)


def lorentz_to_poincare_2d(lorentz: np.ndarray) -> np.ndarray:
    """Stereographic projection L^2 -> Poincaré disk D^2.

    lorentz : (N, 3) [x0, x1, x2]
    Returns : (N, 2) with ||p|| < 1
    """
    x0 = lorentz[:, :1]
    xi = lorentz[:, 1:]
    return xi / (1.0 + x0)


def tangent_to_poincare_disk(tangent: np.ndarray, curv: float = 1.0) -> np.ndarray:
    """Full pipeline: high-d tangent features -> 2D Poincaré disk at K=1.

    Steps:
      1. PCA to 2D  (linear dimensionality reduction)
      2. L2-normalise each row to unit norm
      3. Scale by 0.85 (keeps cosh(sqrt_c * 0.85) well-defined)
      4. exp_map at K=curv
      5. Stereographic projection -> Poincaré disk

    Returns (N, 2) with ||p|| < 1.
    """
    pca = PCA(n_components=2, random_state=42)
    v2 = pca.fit_transform(tangent.astype(np.float32))

    norms = np.linalg.norm(v2, axis=1, keepdims=True).clip(min=1e-8)
    v2 = v2 / norms * 0.85          # unit-norm then scale to 0.85

    lorentz = exp_map_origin_2d(v2, curv=curv)
    return lorentz_to_poincare_2d(lorentz)


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(algorithm, loaders, curvatures, device, n_samples, num_source):
    algorithm.eval()
    results = {}
    is_meru = "meru" in type(algorithm).__name__.lower()

    for env_idx, (name, loader, _) in enumerate(loaders):
        if env_idx >= num_source:
            break

        single = getattr(algorithm, "single_curvature", False)
        curv_val = (curvatures[0].item() if single
                    else curvatures[min(env_idx, len(curvatures) - 1)].item())

        all_feats, all_labels = [], []
        collected = 0
        for x, y in loader:
            if collected >= n_samples:
                break
            x = x.to(device)
            feats = algorithm.featurizer(x)
            if is_meru:
                feats = F.normalize(feats, dim=-1)
            scale = max(curv_val ** 0.5, 1e-8)
            tangent = feats / scale
            all_feats.append(tangent.cpu())
            all_labels.append(y)
            collected += x.shape[0]

        feats_cat  = torch.cat(all_feats,  0)[:n_samples]
        labels_cat = torch.cat(all_labels, 0)[:n_samples]

        results[name] = {
            "tangent":   feats_cat.numpy(),
            "labels":    labels_cat.numpy(),
            "curvature": curv_val,
        }
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

DOMAIN_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3",
                 "#CCB974", "#64B5CD", "#E07B39", "#6BAED6"]
CLASS_MARKERS  = ["o", "s", "^", "D", "v", "P", "*", "X"]


def plot_poincare_disk(domain_data: dict, out_path: Path, title: str = ""):
    domain_names = list(domain_data.keys())
    n_panels = len(domain_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    all_classes = sorted({int(l) for d in domain_data.values() for l in d["labels"]})

    for ax, dom_name in zip(axes, domain_names):
        data   = domain_data[dom_name]
        coords = data["coords_2d"]
        labels = data["labels"]
        curv   = data["curvature"]

        # Disk boundary
        ax.add_patch(plt.Circle((0, 0), 1.0, color="black",
                                fill=False, linewidth=1.8))
        # Guide circles at 1/3 and 2/3 radius
        for r in [0.33, 0.66]:
            ax.add_patch(plt.Circle((0, 0), r, color="gray", fill=False,
                                    linewidth=0.4, linestyle="--", alpha=0.4))

        for cls_idx, cls in enumerate(all_classes):
            mask = labels == cls
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=DOMAIN_COLORS[cls_idx % len(DOMAIN_COLORS)],
                marker=CLASS_MARKERS[cls_idx % len(CLASS_MARKERS)],
                s=22, alpha=0.75, linewidths=0,
            )

        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{dom_name}\nlearnt K={curv:.4f}", fontsize=9)

    handles = [
        mpatches.Patch(color=DOMAIN_COLORS[i % len(DOMAIN_COLORS)],
                       label=f"class {c}")
        for i, c in enumerate(all_classes[:8])
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(all_classes), 8), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or "Poincaré disk: per-domain feature distributions",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=200)
    print(f"[plot_poincare_disk] Saved → {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--out",       default="outputs/poincare_disk.pdf")
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--device",    default="cuda:2")
    return p.parse_args()


def resolve_dataset_config(root_cfg):
    dataset_cfg = root_cfg.get("dataset", {})
    nested_path = dataset_cfg.get("config")
    if nested_path:
        nested_cfg = load_yaml_config(nested_path)
        return {**nested_cfg, **{k: v for k, v in dataset_cfg.items() if k != "config"}}
    return dataset_cfg


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg         = load_yaml_config(args.config)
    dataset_cfg = resolve_dataset_config(cfg)
    seed        = cfg.get("seed", 0)
    set_seed(seed)

    dataset_name = dataset_cfg["dataset"]
    test_envs    = dataset_cfg.get("test_envs", [0])
    holdout_frac = dataset_cfg.get("holdout_fraction", 0.2)

    hparams = hparams_registry.default_hparams("ERM", dataset_name)
    hparams.update(cfg.get("hparams", {}))
    hparams["batch_size"] = int(hparams.get("batch_size", 32))

    dataset_class = db_datasets.get_dataset_class(dataset_name)
    dataset       = dataset_class(dataset_cfg["data_dir"], test_envs, hparams)
    num_source    = len(dataset) - len(test_envs)

    in_splits = []
    for env_i, env in enumerate(dataset):
        n_out = int(len(env) * holdout_frac)
        in_sp, _ = db_misc.split_dataset(env, len(env) - n_out,
                                          db_misc.seed_hash(seed, env_i))
        in_splits.append(in_sp)

    eval_loaders = [
        (f"env{i}", FastDataLoader(dataset=sp, batch_size=64,
                                   num_workers=dataset.N_WORKERS), None)
        for i, sp in enumerate(in_splits)
    ]

    algo_name = cfg.get("algorithm", "hyperdg")
    algorithm  = build_algorithm(algo_name, dataset.input_shape,
                                 dataset.num_classes, num_source, hparams)
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    algorithm.load_state_dict(state, strict=False)
    algorithm.to(device)

    with torch.no_grad():
        curvatures = F.softplus(algorithm.log_curvatures) + algorithm.curv_eps
    print(f"Curvatures: {[round(c, 4) for c in curvatures.tolist()]}")

    raw = extract_features(algorithm, eval_loaders, curvatures,
                           device, args.n_samples, num_source)

    domain_data = {}
    for dom_name, d in raw.items():
        print(f"  {dom_name}: {len(d['tangent'])} pts, learnt K={d['curvature']:.5f} ...")
        coords = tangent_to_poincare_disk(d["tangent"], curv=1.0)
        domain_data[dom_name] = {
            "coords_2d": coords,
            "labels":    d["labels"][:len(coords)],
            "curvature": d["curvature"],
        }

    single = hparams.get("single_curvature", False)
    title  = (f"Poincaré disk (fixed K=−1) — {dataset_name} "
              f"({'single α' if single else 'per-domain α'})\n"
              f"PCA → exp-map → stereographic projection")
    plot_poincare_disk(domain_data, Path(args.out), title=title)


if __name__ == "__main__":
    main()
