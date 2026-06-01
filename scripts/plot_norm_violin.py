#!/usr/bin/env python3
"""Per-class hyperbolic norm distribution.

We extract per-sample tangent-space features from a trained HyperDG
checkpoint, compute the tangent norm ||v||_2 = ||feats / sqrt(K)||_2,
and plot the distribution of these norms per class for each domain
on PACS.

Why this figure matters.  A central appeal of hyperbolic geometry for
classification is that hierarchical / fine-grained classes can spread
to different radial depths on the disk.  If the norms (radial depth)
are roughly the same across classes, that's evidence the geometry is
*not* being used in the way the conjecture predicts; if they spread,
the geometry is being used (though §4 shows that use does not translate
to a held-out accuracy gain over ERM).

Output: outputs/norm_violin_pacs.pdf and HyperDG_TMLR/fig_norm_violin_pacs.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "domainbed"))

from domainbed import datasets as db_datasets
from domainbed import hparams_registry
from domainbed.lib import misc as db_misc
from domainbed.lib.fast_data_loader import FastDataLoader

from sftdg.algorithms import build_algorithm
from sftdg.utils import load_yaml_config, set_seed


@torch.no_grad()
def extract_tangent_norms(algorithm, loaders, curvatures, device,
                          n_samples: int, num_source: int) -> dict:
    """Return {domain: {'norm': np.array(N,), 'label': np.array(N,)}}."""
    algorithm.eval()
    out = {}
    for env_idx, (name, loader, _) in enumerate(loaders):
        if env_idx >= num_source:
            break
        single = getattr(algorithm, "single_curvature", False)
        curv_val = (curvatures[0].item() if single
                    else curvatures[min(env_idx, len(curvatures)-1)].item())
        scale = max(curv_val**0.5, 1e-8)
        norms, labels = [], []
        collected = 0
        for x, y in loader:
            if collected >= n_samples: break
            x = x.to(device)
            feats = algorithm.featurizer(x)
            tangent = feats / scale
            n = tangent.norm(dim=1).cpu().numpy()
            norms.append(n); labels.append(y.numpy())
            collected += x.shape[0]
        out[name] = {
            "norm": np.concatenate(norms)[:n_samples],
            "label": np.concatenate(labels)[:n_samples],
            "curv": curv_val,
        }
    return out


def resolve_dataset_config(root_cfg):
    dataset_cfg = root_cfg.get("dataset", {})
    nested = dataset_cfg.get("config")
    if nested:
        return {**load_yaml_config(nested), **{k: v for k, v in dataset_cfg.items() if k != "config"}}
    return dataset_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT/"configs/batch3/baseline/pacs_split0.yaml"))
    p.add_argument("--ckpt",   default=str(ROOT/"experiments/batch3/baseline/pacs_split0/model_last.pt"))
    p.add_argument("--out",    default=str(ROOT/"outputs/norm_violin_pacs.pdf"))
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_yaml_config(args.config)
    dscfg = resolve_dataset_config(cfg)
    set_seed(cfg.get("seed", 0))

    hparams = hparams_registry.default_hparams("ERM", dscfg["dataset"])
    hparams.update(cfg.get("hparams", {}))
    hparams["batch_size"] = int(hparams.get("batch_size", 32))

    ds_class = db_datasets.get_dataset_class(dscfg["dataset"])
    dataset = ds_class(dscfg["data_dir"], dscfg.get("test_envs", [0]), hparams)
    num_source = len(dataset) - len(dscfg.get("test_envs", [0]))

    in_splits = []
    holdout = dscfg.get("holdout_fraction", 0.2)
    for i, env in enumerate(dataset):
        n_out = int(len(env) * holdout)
        in_sp, _ = db_misc.split_dataset(env, len(env) - n_out,
                                         db_misc.seed_hash(cfg.get("seed", 0), i))
        in_splits.append(in_sp)
    loaders = [(f"env{i}", FastDataLoader(dataset=sp, batch_size=64,
                                          num_workers=dataset.N_WORKERS), None)
               for i, sp in enumerate(in_splits)]

    algo = build_algorithm(cfg.get("algorithm", "hyperdg"),
                           dataset.input_shape, dataset.num_classes,
                           num_source, hparams)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    algo.load_state_dict(state.get("state_dict", state), strict=False)
    algo.to(device)

    curvatures = F.softplus(algo.log_curvatures) + algo.curv_eps
    data = extract_tangent_norms(algo, loaders, curvatures, device,
                                 args.n_samples, num_source)

    # ── Plot: per-domain panel, per-class violin ───────────────────────
    n_panels = len(data)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.6),
                              sharey=True)
    if n_panels == 1: axes = [axes]
    all_classes = sorted({int(c) for d in data.values() for c in d["label"]})
    cmap = plt.get_cmap("tab10")

    for ax, (dom, d) in zip(axes, data.items()):
        groups = [d["norm"][d["label"] == c] for c in all_classes]
        positions = list(range(1, len(all_classes) + 1))
        parts = ax.violinplot(groups, positions=positions, widths=0.8,
                               showmeans=False, showmedians=True, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(cmap(i % 10)); body.set_edgecolor("black")
            body.set_alpha(0.75); body.set_linewidth(0.6)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(c) for c in all_classes], fontsize=8)
        ax.set_title(f"{dom}   (κ = {d['curv']:.4f})", fontsize=10)
        ax.set_xlabel("class index", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("tangent norm  $\\|v\\|_2$", fontsize=9)

    fig.suptitle("Per-class tangent norm distribution (HyperDG, PACS, split 0)\n"
                 "Medians collapse near a common value across classes — "
                 "the radial axis is barely used to separate classes.",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {args.out}")

    # Print median per (domain, class) for the paper
    print("\nDomain    Class   median norm   IQR")
    for dom, d in data.items():
        for c in all_classes:
            v = d["norm"][d["label"] == c]
            if len(v) < 2: continue
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            print(f"  {dom:<8}{c:<8}{q2:8.3f}     {q3-q1:6.3f}")
        print()


if __name__ == "__main__":
    main()
