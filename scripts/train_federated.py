"""Federated training entry point for HyperDG-style experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from domainbed import datasets as db_datasets
from domainbed import hparams_registry
from domainbed import algorithms as db_algorithms
from domainbed.lib import misc as db_misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader

from sftdg.federated.hyperdg_client import HyperDGFedClient, _softplus_curvature
from sftdg.utils import load_yaml_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated HyperDG training.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--device", type=str, default=None, help="Override device from config (cuda/cpu)"
    )
    return parser.parse_args()


def resolve_dataset_config(root_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = root_cfg.get("dataset", {})
    nested_path = dataset_cfg.get("config")
    if nested_path:
        nested_cfg = load_yaml_config(nested_path)
        merged = {**nested_cfg, **{k: v for k, v in dataset_cfg.items() if k != "config"}}
        return merged
    return dataset_cfg


def default_hparams(algorithm: str, dataset: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Bootstraps hparams from DomainBed registry and applies overrides."""
    base_algo = algorithm if algorithm in db_algorithms.ALGORITHMS else "ERM"
    hparams = hparams_registry.default_hparams(base_algo, dataset)
    hparams.update(overrides)
    hparams["batch_size"] = int(hparams.get("batch_size", 32))
    return hparams


def prepare_splits(
    dataset, test_envs: Sequence[int], hparams: Dict[str, Any], holdout_fraction: float, seed: int
) -> Tuple[List, List, List]:
    """Prepare DomainBed train/val splits per environment."""
    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        env_split = int(len(env) * holdout_fraction)
        in_split, out_split = db_misc.split_dataset(
            env,
            len(env) - env_split,
            db_misc.seed_hash(seed, env_i),
        )
        if hparams.get("class_balanced", False):
            in_weights = db_misc.make_weights_for_balanced_classes(in_split)
            out_weights = db_misc.make_weights_for_balanced_classes(out_split)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_split, in_weights))
        out_splits.append((out_split, out_weights))

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)
    ]
    eval_weights = [weights for _, weights in (in_splits + out_splits)]
    eval_names = [f"env{idx}_in" for idx in range(len(in_splits))]
    eval_names += [f"env{idx}_out" for idx in range(len(out_splits))]
    return in_splits, list(zip(eval_names, eval_loaders, eval_weights)), out_splits


def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("FedAvg requires positive sample weights.")
    avg_state: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        accum = None
        for state, weight in zip(state_dicts, weights):
            tensor = state[key] * (weight / total)
            accum = tensor if accum is None else accum + tensor
        avg_state[key] = accum
    return avg_state


def evaluate_global(
    global_state: Dict[str, torch.Tensor],
    input_shape,
    num_classes: int,
    hparams: Dict[str, Any],
    eval_loaders,
    device: torch.device,
) -> Dict[str, float]:
    model = HyperDGFedClient(input_shape, num_classes, hparams)
    model.set_global_state(global_state)
    model.to(device)
    model.eval()
    metrics = {}
    for name, loader, weights in eval_loaders:
        metrics[f"{name}_acc"] = db_misc.accuracy(model, loader, weights, device)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    dataset_cfg = resolve_dataset_config(cfg)

    output_dir = Path(cfg.get("output_dir", "experiments/federated/unnamed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.jsonl"

    device_name = args.device or cfg.get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    seed = cfg.get("seed", 0)
    set_seed(seed)
    rng = np.random.RandomState(seed)

    dataset_name = dataset_cfg["dataset"]
    test_envs = dataset_cfg.get("test_envs", [])
    holdout_fraction = dataset_cfg.get("holdout_fraction", 0.2)
    hparams = default_hparams(cfg.get("algorithm", "hyperdg"), dataset_name, cfg.get("hparams", {}))

    federated_cfg = cfg.get("federated", {})
    num_rounds = int(federated_cfg.get("num_rounds", 500))
    local_steps = int(federated_cfg.get("local_steps", 10))
    eval_freq = int(federated_cfg.get("eval_freq", 10))
    clients_per_round = federated_cfg.get("clients_per_round", None)
    if clients_per_round is None:
        clients_per_round = -1

    dataset_class = db_datasets.get_dataset_class(dataset_name)
    dataset = dataset_class(dataset_cfg["data_dir"], test_envs, hparams)
    if "num_workers" in dataset_cfg:
        dataset.N_WORKERS = int(dataset_cfg["num_workers"])

    in_splits, eval_loaders, _ = prepare_splits(
        dataset, test_envs, hparams, holdout_fraction, seed
    )

    train_envs = [i for i in range(len(dataset)) if i not in test_envs]
    if not train_envs:
        raise ValueError("No training environments available after excluding test_envs.")

    client_loaders = []
    for env_i in train_envs:
        env, weights = in_splits[env_i]
        client_loaders.append(
            InfiniteDataLoader(
                dataset=env,
                weights=weights,
                batch_size=hparams["batch_size"],
                num_workers=dataset.N_WORKERS,
            )
        )
    client_iters = [iter(loader) for loader in client_loaders]

    num_clients = len(train_envs)
    if clients_per_round <= 0 or clients_per_round > num_clients:
        clients_per_round = num_clients

    template = HyperDGFedClient(dataset.input_shape, dataset.num_classes, hparams)
    global_state = template.get_global_state()
    feat_dim = template.featurizer.n_outputs
    mu_global = torch.zeros(feat_dim)
    curv_mean = _softplus_curvature(torch.tensor(0.0), template.curv_eps).detach()

    client_log_curvatures = [torch.tensor(0.0) for _ in range(num_clients)]
    curvature_rounds: List[np.ndarray] = []

    with results_path.open("w", encoding="utf-8") as metrics_f:
        for round_idx in range(num_rounds):
            selected = rng.choice(num_clients, size=clients_per_round, replace=False)
            client_states: List[Dict[str, torch.Tensor]] = []
            client_weights: List[int] = []
            client_mus: List[torch.Tensor] = []
            round_metrics_sum: Dict[str, float] = {}
            round_weight = 0

            for client_id in selected:
                client = HyperDGFedClient(dataset.input_shape, dataset.num_classes, hparams)
                client.set_global_state(global_state)
                client.log_curvature.data = client_log_curvatures[client_id].clone()
                client.to(device)

                mu_accum = torch.zeros_like(mu_global)
                sample_accum = 0
                for _ in range(local_steps):
                    batch = next(client_iters[client_id])
                    metrics, mu_batch, num_samples = client.update(
                        batch, mu_global=mu_global, curv_mean=curv_mean
                    )
                    mu_accum += mu_batch.cpu() * num_samples
                    sample_accum += num_samples
                    for key, value in metrics.items():
                        round_metrics_sum[key] = round_metrics_sum.get(key, 0.0) + value * num_samples

                if sample_accum == 0:
                    continue

                client_states.append(client.get_global_state())
                client_weights.append(sample_accum)
                client_mus.append(mu_accum / sample_accum)
                client_log_curvatures[client_id] = client.log_curvature.detach().cpu()
                round_weight += sample_accum

            if not client_states:
                raise RuntimeError("No client updates collected for this round.")

            global_state = fedavg(client_states, client_weights)

            weighted_mu = torch.zeros_like(mu_global)
            weighted_curv = 0.0
            total_weight = float(sum(client_weights))
            for mu, weight, client_id in zip(client_mus, client_weights, selected):
                weighted_mu += mu * (weight / total_weight)
                curv_val = _softplus_curvature(client_log_curvatures[client_id], template.curv_eps).item()
                weighted_curv += curv_val * (weight / total_weight)
            mu_global = weighted_mu
            curv_mean = torch.tensor(weighted_curv)

            all_curvatures = np.array(
                [
                    _softplus_curvature(log_k, template.curv_eps).item()
                    for log_k in client_log_curvatures
                ],
                dtype=np.float32,
            )
            curvature_rounds.append(all_curvatures)

            global_curv = _softplus_curvature(
                global_state["log_global_curvature"], template.curv_eps
            ).item()
            metrics_out = {
                "round": round_idx,
                "loss_total": round_metrics_sum.get("loss_total", 0.0) / max(round_weight, 1),
                "loss_sup": round_metrics_sum.get("loss_sup", 0.0) / max(round_weight, 1),
                "loss_curv": round_metrics_sum.get("loss_curv", 0.0) / max(round_weight, 1),
                "loss_align": round_metrics_sum.get("loss_align", 0.0) / max(round_weight, 1),
                "loss_sharp": round_metrics_sum.get("loss_sharp", 0.0) / max(round_weight, 1),
                "loss_local_flat": round_metrics_sum.get("loss_local_flat", 0.0) / max(round_weight, 1),
                "loss_global_flat": round_metrics_sum.get("loss_global_flat", 0.0) / max(round_weight, 1),
                "train_acc": round_metrics_sum.get("train_acc", 0.0) / max(round_weight, 1),
                "curvature_mean": float(all_curvatures.mean()),
                "curvature_std": float(all_curvatures.std(ddof=0)),
                "global_curvature": global_curv,
            }

            if round_idx % eval_freq == 0 or round_idx == num_rounds - 1:
                eval_metrics = evaluate_global(
                    global_state,
                    dataset.input_shape,
                    dataset.num_classes,
                    hparams,
                    eval_loaders,
                    device,
                )
                metrics_out.update(eval_metrics)

            metrics_f.write(json.dumps(metrics_out) + "\n")
            metrics_f.flush()

    curvature_rounds_arr = np.stack(curvature_rounds, axis=0)
    np.save(output_dir / "curvature_rounds.npy", curvature_rounds_arr)
    np.save(output_dir / "curvature_final.npy", curvature_rounds_arr[-1])

    final_eval = evaluate_global(
        global_state,
        dataset.input_shape,
        dataset.num_classes,
        hparams,
        eval_loaders,
        device,
    )
    env_names = getattr(dataset_class, "ENVIRONMENTS", [f"env{i}" for i in range(len(dataset))])
    domain_rows = []
    for env_idx, env_name in enumerate(env_names):
        acc = final_eval.get(f"env{env_idx}_out_acc")
        if acc is None:
            continue
        domain_rows.append((dataset_name, env_name, acc))

    domain_csv = output_dir / "domain_accuracy.csv"
    with domain_csv.open("w", encoding="utf-8") as handle:
        handle.write("dataset,domain,accuracy\n")
        for dataset_label, env_label, acc in domain_rows:
            handle.write(f"{dataset_label},{env_label},{acc:.6f}\n")

    checkpoint_path = output_dir / "model_last.pt"
    torch.save(
        {
            "global_state": global_state,
            "client_log_curvatures": client_log_curvatures,
            "mu_global": mu_global,
            "curv_mean": curv_mean,
            "hparams": hparams,
            "rounds": num_rounds,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
