#!/usr/bin/env python3
"""Aggregate results for Reviewer hQnG experiments and print comparison tables.

Usage (run after all experiments finish):
  python scripts/summarize_reviewer_hQnG.py

Also runs Poincaré disk visualisation for each dataset using the best checkpoint
found in experiments/reviewer_hQnG/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT / "experiments" / "reviewer_hQnG"
PYBIN = str(ROOT.parent / ".conda" / "envs" / "sftdg" / "bin" / "python3")


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_best_acc(exp_dir: Path) -> float | None:
    """Return the best out-of-domain accuracy from metrics.jsonl."""
    metrics_path = exp_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    best = None
    try:
        with open(metrics_path) as f:
            for line in f:
                rec = json.loads(line)
                # Average accuracy over test-domain out-splits
                test_accs = [v for k, v in rec.items()
                             if k.endswith("_out_acc") and "env" in k]
                if test_accs:
                    avg = float(np.mean(test_accs))
                    if best is None or avg > best:
                        best = avg
    except Exception:
        pass
    return best


def collect_results(exp_type: str, datasets: list[str]) -> dict:
    """
    exp_type : "single_curv" | "meru" | "meru_single"
    Returns  : {dataset: [acc_split0, acc_split1, ...]}
    """
    results = {}
    base = EXP_ROOT / exp_type
    for ds in datasets:
        accs = []
        split = 0
        while True:
            exp_dir = base / f"{ds}_split{split}"
            if not exp_dir.exists():
                break
            acc = read_best_acc(exp_dir)
            accs.append(acc)
            split += 1
        if accs:
            results[ds] = accs
    return results


def fmt(x: float | None) -> str:
    return f"{100 * x:.1f}" if x is not None else "n/a"


def mean_std(accs: list) -> tuple[float | None, float | None]:
    valid = [a for a in accs if a is not None]
    if not valid:
        return None, None
    arr = np.array(valid)
    return arr.mean(), arr.std()


# ── Table printers ────────────────────────────────────────────────────────────

SINGLE_CURV_DATASETS = ["pacs", "vlcs", "officehome", "terraincognita", "domainnet"]
MERU_DATASETS = ["pacs", "vlcs", "officehome"]


def print_single_curv_table():
    print("\n" + "=" * 70)
    print("TABLE 1 — Single global α vs. per-domain α_i (Reviewer hQnG Q1)")
    print("=" * 70)

    # Original per-domain results live in experiments/ (non-reviewer dirs)
    orig_dirs = {
        "pacs": "hyperdg_pacs",
        "vlcs": "hyperdg_vlcs",
        "officehome": "hyperdg_officehome",
        "terraincognita": "hyperdg_terraincognita",
        "domainnet": "hyperdg_domainnet",
    }

    single_res = collect_results("single_curv", SINGLE_CURV_DATASETS)

    header = f"{'Dataset':<16} {'per-domain α (orig)':<22} {'single α (ablation)':<22} {'Δ':>6}"
    print(header)
    print("-" * 70)
    for ds in SINGLE_CURV_DATASETS:
        # Try to read original per-domain results
        orig_accs = []
        orig_dir = ROOT / "experiments" / orig_dirs.get(ds, "")
        split = 0
        while True:
            sp_dir = ROOT / "experiments" / f"{orig_dirs.get(ds,'xxx')}_split{split}"
            if not sp_dir.exists():
                # Try the simple (non-split) directory
                break
            orig_accs.append(read_best_acc(sp_dir))
            split += 1

        orig_mean, orig_std = mean_std(orig_accs)
        single_accs = single_res.get(ds, [])
        single_mean, single_std = mean_std(single_accs)

        orig_str = f"{fmt(orig_mean)} ± {fmt(orig_std)}" if orig_mean else "—"
        single_str = (
            f"{fmt(single_mean)} ± {fmt(single_std)}" if single_mean else "running…"
        )
        delta_str = (
            f"{(single_mean - orig_mean)*100:+.1f}"
            if (single_mean and orig_mean) else "—"
        )
        print(f"{ds:<16} {orig_str:<22} {single_str:<22} {delta_str:>6}")
    print("-" * 70)
    print("Positive Δ means single α outperforms per-domain α.")


def print_meru_table():
    print("\n" + "=" * 70)
    print("TABLE 2 — MERU backbone: per-domain α vs. single α (hQnG Q2a,b)")
    print("=" * 70)

    meru_res = collect_results("meru", MERU_DATASETS)
    meru_single_res = collect_results("meru_single", MERU_DATASETS)

    header = (
        f"{'Dataset':<16} {'MERU per-domain α':<22}"
        f" {'MERU single α':<22} {'Δ':>6}"
    )
    print(header)
    print("-" * 70)
    for ds in MERU_DATASETS:
        m_accs = meru_res.get(ds, [])
        ms_accs = meru_single_res.get(ds, [])
        m_mean, m_std = mean_std(m_accs)
        ms_mean, ms_std = mean_std(ms_accs)

        m_str = f"{fmt(m_mean)} ± {fmt(m_std)}" if m_mean else "running…"
        ms_str = f"{fmt(ms_mean)} ± {fmt(ms_std)}" if ms_mean else "running…"
        delta_str = (
            f"{(ms_mean - m_mean)*100:+.1f}"
            if (m_mean and ms_mean) else "—"
        )
        print(f"{ds:<16} {m_str:<22} {ms_str:<22} {delta_str:>6}")
    print("-" * 70)


def run_poincare_plots():
    """Generate Poincaré disk plots for each experiment type × dataset."""
    print("\n=== Generating Poincaré disk visualisations ===")
    plot_script = str(ROOT / "scripts" / "plot_poincare_disk.py")

    tasks = [
        # (experiment_dir, config_path, out_name, gpu)
        (EXP_ROOT / "single_curv" / "pacs_split0",
         ROOT / "configs" / "reviewer_hQnG" / "single_curv" / "pacs_split0.yaml",
         "outputs/poincare_single_curv_pacs.pdf", "cuda:2"),
        (EXP_ROOT / "meru" / "pacs_split0",
         ROOT / "configs" / "reviewer_hQnG" / "meru" / "pacs_split0.yaml",
         "outputs/poincare_meru_pacs.pdf", "cuda:2"),
        (EXP_ROOT / "meru_single" / "pacs_split0",
         ROOT / "configs" / "reviewer_hQnG" / "meru_single" / "pacs_split0.yaml",
         "outputs/poincare_meru_single_pacs.pdf", "cuda:2"),
    ]

    for exp_dir, cfg, out, device in tasks:
        ckpt = exp_dir / "model_last.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue
        print(f"  Plotting {out} …")
        cmd = [
            PYBIN, plot_script,
            "--config", str(cfg),
            "--ckpt", str(ckpt),
            "--out", str(ROOT / out),
            "--n_samples", "200",
            "--device", device,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  [ERROR] {result.stderr[-500:]}")
        else:
            print(f"  [OK]    {out}")


if __name__ == "__main__":
    print_single_curv_table()
    print_meru_table()

    # Only run plots if --plots flag given (slower)
    if "--plots" in sys.argv:
        run_poincare_plots()
    else:
        print("\nRun with --plots to also generate Poincaré disk figures.")
