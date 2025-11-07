"""Generate loss landscape plots for FedCuri / curvature-aware algorithms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from domainbed import datasets as db_datasets
from domainbed import hparams_registry
from domainbed.algorithms import ALGORITHMS as DB_ALGORITHMS

from sftdg.algorithms import build_algorithm
from sftdg.utils import load_yaml_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curvature-aware loss landscapes.")
    parser.add_argument("--config", type=str, required=True, help="Training config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint saved by scripts/train.py",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--points", type=int, default=41, help="Number of samples along the θ-line")
    parser.add_argument("--radius", type=float, default=0.15, help="Half-length of the θ sweep")
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=-1,
        help="Maximum samples per domain to estimate losses",
    )
    parser.add_argument("--output", type=str, default="fedcuri_landscape.png")
    parser.add_argument("--json", type=str, default=None, help="Optional JSON export")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fit-degree",
        type=int,
        default=4,
        help="Polynomial degree for smoothing (-1 disables smoothing)",
    )
    return parser.parse_args()


def build_dataset(cfg: Dict, hparams: Dict):
    dataset_name = cfg["dataset"]
    dataset_class = db_datasets.get_dataset_class(dataset_name)
    return dataset_class(cfg["data_dir"], cfg.get("test_envs", []), hparams)


def default_hparams(algorithm: str, dataset: str, overrides: Dict[str, float]) -> Dict:
    base_algo = algorithm if algorithm in DB_ALGORITHMS else "ERM"
    hparams = hparams_registry.default_hparams(base_algo, dataset)
    hparams.update(overrides or {})
    return hparams


def truncate_loader(dataset, max_samples: int, batch_size: int, num_workers: int):
    if max_samples <= 0:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    indices = list(range(min(len(dataset), max_samples)))
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def compute_domain_loss(
    algorithm,
    loader,
    env_idx: int,
    device: torch.device,
    mode: str = "curvature",
    env_to_domain: Dict[int, int] | None = None,
    global_curv: torch.Tensor | None = None,
) -> float:
    algorithm.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            domain_idx = env_to_domain.get(env_idx) if env_to_domain else None
            if (
                mode == "curvature"
                and domain_idx is not None
                and hasattr(algorithm, "log_curvatures")
            ):
                curvatures = torch.nn.functional.softplus(algorithm.log_curvatures) + algorithm.curv_eps
                feats = algorithm.featurizer(x)
                tangent = algorithm._tangent_embedding(feats, curvatures[domain_idx])
                logits = algorithm.classifier(tangent)
            elif (
                mode in {"curvature", "global"}
                and global_curv is not None
                and hasattr(algorithm, "log_global_curvature")
            ):
                feats = algorithm.featurizer(x)
                tangent = algorithm._tangent_embedding(feats, global_curv)
                logits = algorithm.classifier(tangent)
            else:
                logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += loss.item()
            total_count += y.size(0)
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_yaml_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    if "config" in dataset_cfg:
        dataset_cfg = {**load_yaml_config(dataset_cfg["config"]), **dataset_cfg}
        dataset_cfg.pop("config", None)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hparams = default_hparams(cfg.get("algorithm", "sft"), dataset_cfg["dataset"], cfg.get("hparams", {}))
    dataset = build_dataset(dataset_cfg, hparams)

    algorithm = build_algorithm(
        cfg["algorithm"],
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(dataset_cfg.get("test_envs", [])),
        hparams,
    )
    algorithm.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    algorithm.load_state_dict(checkpoint["state_dict"])

    param_vector = parameters_to_vector(algorithm.parameters()).detach()
    direction = torch.randn_like(param_vector)
    direction /= direction.norm() + 1e-8

    train_envs = [
        idx for idx in range(len(dataset)) if idx not in dataset_cfg.get("test_envs", [])
    ]
    env_to_domain = {env_idx: domain_idx for domain_idx, env_idx in enumerate(train_envs)}

    if hasattr(algorithm, "log_global_curvature"):
        global_curv = (
            torch.nn.functional.softplus(algorithm.log_global_curvature) + algorithm.curv_eps
        )
    else:
        global_curv = None

    loaders = []
    for env_idx, env in enumerate(dataset):
        loaders.append(
            truncate_loader(env, args.samples_per_domain, batch_size=64, num_workers=dataset.N_WORKERS)
        )

    thetas = torch.linspace(-args.radius, args.radius, steps=args.points)
    losses_raw: Dict[int, List[float]] = {idx: [] for idx in range(len(loaders))}
    losses_curv: Dict[int, List[float]] = {idx: [] for idx in range(len(loaders))}

    with torch.no_grad():
        for theta in thetas:
            vector = param_vector + theta * direction
            vector_to_parameters(vector, algorithm.parameters())

            for env_idx, loader in enumerate(loaders):
                raw_loss = compute_domain_loss(
                    algorithm,
                    loader,
                    env_idx,
                    device,
                    mode="raw",
                    env_to_domain=env_to_domain,
                    global_curv=global_curv,
                )
                curv_loss = compute_domain_loss(
                    algorithm,
                    loader,
                    env_idx,
                    device,
                    mode="curvature",
                    env_to_domain=env_to_domain,
                    global_curv=global_curv,
                )
                losses_raw[env_idx].append(raw_loss)
                losses_curv[env_idx].append(curv_loss)

        vector_to_parameters(param_vector, algorithm.parameters())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    theta_vals = thetas.cpu().numpy()
    theta_smooth = np.linspace(theta_vals.min(), theta_vals.max(), num=400)

    for env_idx in losses_raw:
        raw_np = np.array(losses_raw[env_idx])
        curv_np = np.array(losses_curv[env_idx])

        if args.fit_degree >= 1 and args.fit_degree < len(theta_vals):
            raw_coeff = np.polyfit(theta_vals, raw_np, deg=args.fit_degree)
            curv_coeff = np.polyfit(theta_vals, curv_np, deg=args.fit_degree)
            raw_smooth = np.polyval(raw_coeff, theta_smooth)
            curv_smooth = np.polyval(curv_coeff, theta_smooth)
            axes[0].plot(theta_smooth, raw_smooth, color=colors[env_idx % len(colors)], label=f"Domain {env_idx}")
            axes[1].plot(theta_smooth, curv_smooth, color=colors[env_idx % len(colors)], label=f"Domain {env_idx}")
        else:
            axes[0].plot(theta_vals, raw_np, color=colors[env_idx % len(colors)], label=f"Domain {env_idx}")
            axes[1].plot(theta_vals, curv_np, color=colors[env_idx % len(colors)], label=f"Domain {env_idx}")

    axes[0].set_title("Without curvature alignment")
    axes[1].set_title("FedCuri curvature-aligned")
    for ax in axes:
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

    if args.json:
        payload = {
            "theta": theta_vals.tolist(),
            "losses_raw": {str(k): [float(val) for val in vals] for k, vals in losses_raw.items()},
            "losses_curv": {str(k): [float(val) for val in vals] for k, vals in losses_curv.items()},
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON data to {args.json}")


if __name__ == "__main__":
    main()
