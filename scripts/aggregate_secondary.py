#!/usr/bin/env python3
"""Aggregate exploratory and reviewer-requested HyperDG analyses."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUTS = ROOT / "outputs"
DATASETS = ["pacs", "vlcs", "officehome", "terraincognita"]


def last_record(path: Path) -> dict | None:
    try:
        lines = [line for line in path.read_text().splitlines() if line]
        return json.loads(lines[-1])
    except (OSError, IndexError, json.JSONDecodeError):
        return None


def heldout_accuracy(run_dir: Path, split: int, expected_step: int = 4999) -> float | None:
    record = last_record(run_dir / "metrics.jsonl")
    if record is None or record.get("step") != expected_step:
        return None
    if not (run_dir / "model_last.pt").exists():
        return None
    value = record.get(f"env{split}_out_acc")
    return None if value is None else float(value) * 100


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = OUTPUTS / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def summarize(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_shared() -> None:
    rows = []
    for dataset in DATASETS:
        per_domain, shared, used_splits = [], [], []
        for split in range(4):
            per = heldout_accuracy(
                EXPERIMENTS / "batch3" / "baseline" / f"{dataset}_split{split}",
                split,
            )
            one = heldout_accuracy(
                EXPERIMENTS / "reviewer_hQnG" / "single_curv" / f"{dataset}_split{split}",
                split,
            )
            if per is None or one is None:
                continue
            per_domain.append(per)
            shared.append(one)
            used_splits.append(split)
        per_mean, per_std = summarize(per_domain)
        shared_mean, shared_std = summarize(shared)
        rows.append(
            {
                "dataset": dataset,
                "n_splits": len(used_splits),
                "included_splits": ";".join(map(str, used_splits)),
                "per_domain_mean": f"{per_mean:.4f}",
                "per_domain_std": f"{per_std:.4f}",
                "shared_mean": f"{shared_mean:.4f}",
                "shared_std": f"{shared_std:.4f}",
                "shared_minus_per_domain": f"{shared_mean - per_mean:.4f}",
            }
        )
    write_csv("shared_curvature_seed0.csv", list(rows[0]), rows)


def aggregate_loo() -> None:
    conditions = {
        "full": EXPERIMENTS / "batch3" / "baseline",
        "drop_alignment": EXPERIMENTS / "batch2" / "no_align",
        "drop_norm_proxy": EXPERIMENTS / "batch2" / "no_sharp",
        "drop_local_global": EXPERIMENTS / "batch2" / "no_flat",
        "drop_variance": EXPERIMENTS / "batch1" / "novar" / "a0",
        "freeze_curvature": EXPERIMENTS / "batch1" / "fixed" / "frozen",
        "zero_penalty": EXPERIMENTS / "batch2" / "erm",
    }
    rows = []
    for condition, base in conditions.items():
        row = {"condition": condition}
        for dataset in ("pacs", "officehome"):
            values = [
                heldout_accuracy(base / f"{dataset}_split{split}", split)
                for split in range(4)
            ]
            clean = [value for value in values if value is not None]
            mean, std = summarize(clean)
            row[f"{dataset}_mean"] = f"{mean:.4f}"
            row[f"{dataset}_std"] = f"{std:.4f}"
        rows.append(row)
    write_csv("loo_seed0.csv", list(rows[0]), rows)


def aggregate_proxy_ablation() -> None:
    base = EXPERIMENTS / "ablations" / "hyperdg_sam_ablation_officehome_v6"
    conditions = {
        "zero_penalty": "sam_abl_01_erm",
        "proxy_terms_only": "sam_abl_02_sam_only",
        "curvature_alignment_only": "sam_abl_03_hyperdg_no_sam",
        "full": "sam_abl_04_hyperdg_full",
    }
    rows = []
    for label, directory in conditions.items():
        values = []
        row = {"condition": label}
        for split in range(4):
            run = base / f"split{split}" / "hyperdg_sam_ablation" / directory
            value = heldout_accuracy(run, split)
            values.append(value)
            row[f"split{split}"] = f"{value:.4f}" if value is not None else ""
        clean = [value for value in values if value is not None]
        mean, std = summarize(clean)
        row["mean"] = f"{mean:.4f}"
        row["std"] = f"{std:.4f}"
        rows.append(row)
    write_csv("proxy_ablation_officehome_seed0.csv", list(rows[0]), rows)


def aggregate_variance_sweep() -> None:
    conditions = {
        "0.00": EXPERIMENTS / "batch1" / "novar" / "a0",
        "0.05": EXPERIMENTS / "batch1" / "lvar" / "a0p05",
        "0.20": EXPERIMENTS / "batch3" / "baseline",
        "0.50": EXPERIMENTS / "batch1" / "lvar" / "a0p5",
        "2.00": EXPERIMENTS / "batch1" / "lvar" / "a2p0",
    }
    rows = []
    for weight, base in conditions.items():
        values = [
            heldout_accuracy(base / f"pacs_split{split}", split)
            for split in range(4)
        ]
        clean = [value for value in values if value is not None]
        mean, std = summarize(clean)
        rows.append(
            {
                "lambda_var": weight,
                "n_splits": len(clean),
                "mean": f"{mean:.4f}",
                "std": f"{std:.4f}",
            }
        )
    write_csv("variance_sweep_pacs_seed0.csv", list(rows[0]), rows)


def aggregate_camelyon() -> None:
    rows, values = [], []
    base = EXPERIMENTS / "hyperdg_wilds_camelyon"
    for split in range(5):
        value = heldout_accuracy(base / f"split{split}", split)
        if value is None:
            continue
        values.append(value)
        rows.append({"split": split, "heldout_accuracy": f"{value:.4f}"})
    mean, std = summarize(values)
    rows.append({"split": "mean", "heldout_accuracy": f"{mean:.4f}"})
    rows.append({"split": "std", "heldout_accuracy": f"{std:.4f}"})
    write_csv("camelyon17_hyperdg_only.csv", list(rows[0]), rows)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    aggregate_shared()
    aggregate_loo()
    aggregate_proxy_ablation()
    aggregate_variance_sweep()
    aggregate_camelyon()


if __name__ == "__main__":
    main()
