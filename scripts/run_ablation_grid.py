#!/usr/bin/env python3
"""Launch SFT ablations across all available GPUs."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT_DIR / "scripts" / "train.py"


def _deep_merge(target: Dict, overrides: Dict) -> Dict:
    for key, value in overrides.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            if not isinstance(node, dict):
                node = {}
                target[key] = node
            _deep_merge(node, value)
        else:
            target[key] = value
    return target


def _auto_detect_gpus() -> List[str]:
    env_override = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_override:
        parsed = [gpu.strip() for gpu in env_override.split(",") if gpu.strip()]
        if parsed:
            return parsed
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.splitlines():
                if line.strip().startswith("GPU"):
                    token = line.split()[1]
                    gpus.append(token.rstrip(":"))
            if gpus:
                return gpus
    except FileNotFoundError:
        pass
    raise RuntimeError(
        "Unable to detect GPUs automatically. Please provide them via --gpus."
    )


SFT_TABLE4_EXPERIMENTS = [
    {
        "name": "exp01_rho0_onehot",
        "description": "ERM-style: rho=0, one-hot labels",
        "overrides": {
            "hparams": {
                "rho": 0.0,
                "label_mode": "one_hot",
                "label_smoothing": 0.0,
                "lambda1": 0.0,
                "lambda2": 0.0,
                "use_pce": False,
            }
        },
    },
    {
        "name": "exp02_rho0_smoothed",
        "description": "ERM + label smoothing (rho=0)",
        "overrides": {
            "hparams": {
                "rho": 0.0,
                "label_mode": "smoothed",
                "label_smoothing": 0.1,
                "lambda1": 0.0,
                "lambda2": 0.0,
                "use_pce": False,
            }
        },
    },
    {
        "name": "exp03_rho03_onehot",
        "description": "SAM-style: rho=0.3, one-hot labels",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "one_hot",
                "label_smoothing": 0.0,
                "lambda1": 0.0,
                "lambda2": 0.0,
                "use_pce": False,
            }
        },
    },
    {
        "name": "exp04_rho03_smoothed",
        "description": "SAM + label smoothing (rho=0.3)",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "smoothed",
                "label_smoothing": 0.1,
                "lambda1": 0.0,
                "lambda2": 0.0,
                "use_pce": False,
            }
        },
    },
    {
        "name": "exp05_rho0_generated_no_pce",
        "description": "Generated labels, rho=0.0, no PCE",
        "overrides": {
            "hparams": {
                "rho": 0.0,
                "label_mode": "refined",
                "use_pce": False,
                "lambda1": 0.0,
                "lambda2": 0.0,
            }
        },
    },
    {
        "name": "exp06_rho03_generated_no_pce",
        "description": "Generated labels, rho=0.3, no PCE",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "refined",
                "use_pce": False,
                "lambda1": 0.0,
                "lambda2": 0.0,
            }
        },
    },
    {
        "name": "exp07_rho03_generated_lambda1",
        "description": "Generated labels, rho=0.3, lambda1 sweep",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "refined",
                "use_pce": False,
                "lambda1": 0.5,
                "lambda2": 0.0,
            }
        },
    },
    {
        "name": "exp08_rho03_generated_lambda2",
        "description": "Generated labels, rho=0.3, lambda2 sweep",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "refined",
                "use_pce": False,
                "lambda1": 0.0,
                "lambda2": 0.7,
            }
        },
    },
    {
        "name": "exp09_rho03_generated_lambda12",
        "description": "Generated labels, rho=0.3, lambda1+lambda2",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "refined",
                "use_pce": False,
                "lambda1": 0.5,
                "lambda2": 0.7,
            }
        },
    },
    {
        "name": "exp10_rho03_generated_pce",
        "description": "Full SFT: rho=0.3, lambda1+lambda2, PCE on",
        "overrides": {
            "hparams": {
                "rho": 0.3,
                "label_mode": "refined",
                "use_pce": True,
                "lambda1": 0.5,
                "lambda2": 0.7,
            }
        },
    },
]

HYPERDG_COMPONENTS_EXPERIMENTS = [
    {
        "name": "exp01_sup_only",
        "description": "HyperDG with only supervised loss (all penalties off)",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.0,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp02_curvature_only",
        "description": "Only curvature regularization active",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.0,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp03_alignment_only",
        "description": "Only tangent alignment active",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.5,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp04_sharp_only",
        "description": "Only sharpness penalty active",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.0,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp05_local_flat_only",
        "description": "Only local flatness regularizer active",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.0,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.1,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp06_global_flat_only",
        "description": "Only global flatness regularizer active",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.0,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.1,
            }
        },
    },
    {
        "name": "exp07_curv_align",
        "description": "Curvature + alignment terms",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp08_curv_align_sharp",
        "description": "Curvature + alignment + sharpness terms",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp09_curv_align_local",
        "description": "Curvature + alignment + sharpness + local flatness",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.1,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "exp10_full",
        "description": "Full HyperDG objective",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.1,
                "delta_global_flat": 0.1,
            }
        },
    },
]

HYPERDG_PENALTY_SWEEP = [
    {
        "name": "exp07_curv_align",
        "description": "Curvature + alignment terms (rebalance sweep)",
        "variants": [
            {
                "suffix": "alpha03_beta03",
                "description": "alpha=0.3, beta=0.3",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.3,
                        "beta_align": 0.3,
                        "gamma_sharp": 0.0,
                        "delta_local_flat": 0.0,
                        "delta_global_flat": 0.0,
                    }
                },
            },
            {
                "suffix": "alpha04_beta02",
                "description": "alpha=0.4, beta=0.2",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.4,
                        "beta_align": 0.2,
                        "gamma_sharp": 0.0,
                        "delta_local_flat": 0.0,
                        "delta_global_flat": 0.0,
                    }
                },
            },
        ],
    },
    {
        "name": "exp08_curv_align_sharp",
        "description": "Curvature + alignment + sharpness (rebalance sweep)",
        "variants": [
            {
                "suffix": "alpha03_beta03_gamma002",
                "description": "alpha=0.3, beta=0.3, gamma=0.02",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.3,
                        "beta_align": 0.3,
                        "gamma_sharp": 0.02,
                        "delta_local_flat": 0.0,
                        "delta_global_flat": 0.0,
                    }
                },
            },
            {
                "suffix": "alpha04_beta02_gamma002",
                "description": "alpha=0.4, beta=0.2, gamma=0.02",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.4,
                        "beta_align": 0.2,
                        "gamma_sharp": 0.02,
                        "delta_local_flat": 0.0,
                        "delta_global_flat": 0.0,
                    }
                },
            },
        ],
    },
    {
        "name": "exp09_curv_align_local",
        "description": "Curvature + alignment + sharpness + local flatness (rebalance sweep)",
        "variants": [
            {
                "suffix": "alpha03_beta03_gamma002_delta005",
                "description": "alpha=0.3, beta=0.3, gamma=0.02, delta_local=0.05",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.3,
                        "beta_align": 0.3,
                        "gamma_sharp": 0.02,
                        "delta_local_flat": 0.05,
                        "delta_global_flat": 0.0,
                    }
                },
            },
            {
                "suffix": "alpha04_beta02_gamma002_delta005",
                "description": "alpha=0.4, beta=0.2, gamma=0.02, delta_local=0.05",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.4,
                        "beta_align": 0.2,
                        "gamma_sharp": 0.02,
                        "delta_local_flat": 0.05,
                        "delta_global_flat": 0.0,
                    }
                },
            },
        ],
    },
    {
        "name": "exp10_full",
        "description": "Full HyperDG objective (rebalance sweep)",
        "variants": [
            {
                "suffix": "alpha03_beta03_gamma002_deltas005",
                "description": "alpha=0.3, beta=0.3, gamma=0.02, deltas=0.05",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.3,
                        "beta_align": 0.3,
                        "gamma_sharp": 0.02,
                        "delta_local_flat": 0.05,
                        "delta_global_flat": 0.05,
                    }
                },
            },
            {
                "suffix": "alpha035_beta02_gamma002_deltas005",
                "description": "alpha=0.35, beta=0.2, gamma=0.02, deltas=0.05",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.35,
                        "beta_align": 0.2,
                        "gamma_sharp": 0.02,
                        "delta_local_flat": 0.05,
                        "delta_global_flat": 0.05,
                    }
                },
            },
            {
                "suffix": "alpha04_beta02_gamma001_deltas002",
                "description": "alpha=0.4, beta=0.2, gamma=0.01, delta_local=0.05, delta_global=0.02",
                "overrides": {
                    "hparams": {
                        "alpha_curv": 0.4,
                        "beta_align": 0.2,
                        "gamma_sharp": 0.01,
                        "delta_local_flat": 0.05,
                        "delta_global_flat": 0.02,
                    }
                },
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# SAM ablation schedule  (reviewer request: "Ablate SAM")
#
# Goal: isolate the curvature-alignment contribution from the SAM-like
# flatness terms (gamma_sharp, delta_local_flat, delta_global_flat).
#
# Four conditions:
#   1. ERM baseline     – all penalties off
#   2. SAM-like only    – flatness terms on, curvature/alignment off
#   3. HyperDG w/o SAM – curvature + alignment on, flatness terms off
#   4. HyperDG full     – all terms on (matches exp10_full)
# ---------------------------------------------------------------------------
HYPERDG_SAM_ABLATION = [
    {
        "name": "sam_abl_01_erm",
        "description": "ERM baseline: all HyperDG penalties off",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.0,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "sam_abl_02_sam_only",
        "description": "SAM-like only: flatness terms on, curvature/alignment off",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.0,
                "beta_align": 0.0,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.1,
                "delta_global_flat": 0.1,
            }
        },
    },
    {
        "name": "sam_abl_03_hyperdg_no_sam",
        "description": "HyperDG w/o SAM: curvature + alignment only, flatness off",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.0,
                "delta_local_flat": 0.0,
                "delta_global_flat": 0.0,
            }
        },
    },
    {
        "name": "sam_abl_04_hyperdg_full",
        "description": "HyperDG full: curvature + alignment + SAM-like flatness",
        "overrides": {
            "hparams": {
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.1,
                "delta_global_flat": 0.1,
            }
        },
    },
]

SCHEDULES = {
    "sft_table4": SFT_TABLE4_EXPERIMENTS,
    "hyperdg_components": HYPERDG_COMPONENTS_EXPERIMENTS,
    "hyperdg_penalty_sweep": HYPERDG_PENALTY_SWEEP,
    "hyperdg_sam_ablation": HYPERDG_SAM_ABLATION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ablation experiments across multiple GPUs."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base YAML config (e.g. configs/training/resnet50_sft_officehome.yaml)",
    )
    parser.add_argument(
        "--schedule",
        choices=sorted(SCHEDULES.keys()),
        default="sft_table4",
        help="Predefined ablation schedule to run (e.g. sft_table4, hyperdg_components, "
        "hyperdg_sam_ablation).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids. Defaults to auto detection.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to use.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Path to train.py entry point.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "experiments" / "ablations",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=ROOT_DIR / "experiments" / "configs" / "ablations",
        help="Directory to store generated configs.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=ROOT_DIR / "logs" / "ablations",
        help="Directory for stdout/stderr logs.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Subset of experiment names to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force --device flag (e.g. cuda). Defaults to config value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without launching jobs.",
    )
    return parser.parse_args()


def expand_experiments(experiments: List[Dict]) -> List[Dict]:
    expanded: List[Dict] = []
    for exp in experiments:
        variants = exp.get("variants")
        if variants:
            base_overrides = copy.deepcopy(exp.get("overrides", {}))
            for variant in variants:
                variant_name = variant.get("suffix")
                if not variant_name:
                    raise ValueError(
                        f"Variant for {exp['name']} missing 'suffix' identifier."
                    )
                overrides = copy.deepcopy(base_overrides)
                _deep_merge(overrides, variant.get("overrides", {}))
                expanded.append(
                    {
                        "name": f"{exp['name']}_{variant_name}",
                        "description": variant.get("description", exp.get("description", "")),
                        "overrides": overrides,
                        "base_name": exp["name"],
                    }
                )
        else:
            expanded.append(
                {
                    "name": exp["name"],
                    "description": exp.get("description", ""),
                    "overrides": copy.deepcopy(exp.get("overrides", {})),
                    "base_name": exp["name"],
                }
            )
    return expanded


def prepare_jobs(args: argparse.Namespace) -> List[Dict]:
    with args.base_config.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)
    base_cfg = base_cfg or {}
    base_cfg.setdefault("hparams", {})
    base_name = base_cfg.get("experiment_name", args.base_config.stem)

    experiments = copy.deepcopy(SCHEDULES[args.schedule])
    experiments = expand_experiments(experiments)
    if args.only:
        selected = set(args.only)
        experiments = [
            exp for exp in experiments if exp["name"] in selected or exp["base_name"] in selected
        ]
        missing = selected.difference(
            {exp["name"] for exp in experiments} | {exp["base_name"] for exp in experiments}
        )
        if missing:
            raise ValueError(f"Unknown experiment names in --only: {sorted(missing)}")

    if not experiments:
        raise ValueError("No experiments selected.")

    config_dir = args.config_root / args.schedule
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_root / args.schedule
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_root / args.schedule
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict] = []
    for exp in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("hparams", {})
        cfg["experiment_name"] = f"{base_name}_{exp['name']}"
        cfg.setdefault("output_dir", str(output_dir / exp["name"]))
        cfg["output_dir"] = str(output_dir / exp["name"])
        _deep_merge(cfg, exp.get("overrides", {}))
        cfg.setdefault("notes", exp.get("description", ""))

        cfg_path = config_dir / f"{exp['name']}.yaml"
        with cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        job = {
            "name": exp["name"],
            "description": exp.get("description", ""),
            "config": cfg_path,
            "output_dir": Path(cfg["output_dir"]),
            "log_path": log_dir / f"{exp['name']}.log",
        }
        jobs.append(job)
    return jobs


def build_command(args: argparse.Namespace, config_path: Path) -> List[str]:
    cmd = [args.python_bin, str(args.train_script), "--config", str(config_path)]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def launch_jobs(args: argparse.Namespace, jobs: List[Dict], gpus: List[str]) -> int:
    if args.dry_run:
        for job in jobs:
            print(f"[DRY-RUN] {job['name']}: {' '.join(build_command(args, job['config']))}")
        return 0

    pending = deque(jobs)
    gpu_slots: Dict[str, Dict | None] = {gpu: None for gpu in gpus}
    running: List[Dict] = []
    completed: List[Dict] = []
    failed: List[Dict] = []

    try:
        while pending or any(slot is not None for slot in gpu_slots.values()):
            # Launch jobs on free GPUs
            for gpu, slot in gpu_slots.items():
                if slot is not None or not pending:
                    continue
                job = pending.popleft()
                cmd = build_command(args, job["config"])
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                domainbed_path = ROOT_DIR / "third_party" / "domainbed"
                env["PYTHONPATH"] = (
                    f"{env.get('PYTHONPATH', '')}:{domainbed_path}"
                    if env.get("PYTHONPATH")
                    else str(domainbed_path)
                )
                job["log_path"].parent.mkdir(parents=True, exist_ok=True)
                log_handle = job["log_path"].open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT_DIR),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                job.update(
                    {
                        "process": proc,
                        "log_handle": log_handle,
                        "gpu": gpu,
                        "start_time": time.time(),
                    }
                )
                gpu_slots[gpu] = job
                running.append(job)
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Launching {job['name']} on GPU {gpu} -> {job['log_path']}"
                )

            # Check running jobs
            for gpu, slot in list(gpu_slots.items()):
                if slot is None:
                    continue
                proc = slot["process"]
                retcode = proc.poll()
                if retcode is None:
                    continue
                duration = time.time() - slot["start_time"]
                slot["log_handle"].close()
                slot["duration"] = duration
                slot["returncode"] = retcode
                running.remove(slot)
                gpu_slots[gpu] = None
                if retcode == 0:
                    completed.append(slot)
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Completed {slot['name']} on GPU {gpu} in {duration/60:.1f} min"
                    )
                else:
                    failed.append(slot)
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"FAILED {slot['name']} on GPU {gpu} (code {retcode}). "
                        f"See {slot['log_path']}."
                    )
            if pending and not any(slot is None for slot in gpu_slots.values()):
                time.sleep(5)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, terminating running jobs...")
        for slot in running:
            slot["process"].terminate()
            slot["log_handle"].close()
        raise

    print("\n=== Summary ===")
    for job in completed:
        print(
            f"✓ {job['name']} ({job['description']}) "
            f"-> log: {job['log_path']} [{job['duration']/60:.1f} min]"
        )
    for job in failed:
        print(
            f"✗ {job['name']} "
            f"-> log: {job['log_path']} (exit {job['returncode']})"
        )

    return 0 if not failed else 1


def main() -> None:
    args = parse_args()
    gpus = (
        [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if args.gpus
        else _auto_detect_gpus()
    )
    if not gpus:
        raise RuntimeError("No GPUs available.")

    jobs = prepare_jobs(args)
    exit_code = launch_jobs(args, jobs, gpus)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
