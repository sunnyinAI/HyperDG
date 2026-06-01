#!/usr/bin/env python3
"""Aggregate Batch-1/2/3/4 results into one publication-ready summary.

For each (condition, dataset) cell we have up to 4 seeds x 4 splits = 16
held-out accuracies.  We report:

  * mean +/- std  (across all 16 runs)
  * paired t-test vs. ERM  (matched on (seed, split) pairs)

The paired test is the right one because (seed, split) is the natural
pairing across conditions — each unit-pair shares the same data
partition and initialization, isolating the effect of the loss change.

Outputs:
  docs/results_tables.md        (Markdown for the paper)
  outputs/batch4_summary.csv    (raw cells for further analysis)
  outputs/batch4_per_run.csv    (every run, one row each)
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/sunny/HyperDG/experiments")
OUT  = Path("/data/sunny/HyperDG")

DATASETS = ["pacs", "vlcs", "officehome", "terraincognita"]

# Where to find each (condition, seed) — mixes the four batches.
# Each entry returns a path glob whose components match
# "<ds>_split<n>/metrics.jsonl".
def paths_for(condition: str, seed: int) -> str:
    if seed == 0:
        return {
            "erm":      str(ROOT / "batch2" / "erm"),
            "baseline": str(ROOT / "batch3" / "baseline"),
            "fixed":    str(ROOT / "batch1" / "fixed"  / "frozen"),
            "novar":    str(ROOT / "batch1" / "novar"  / "a0"),
        }[condition]
    return str(ROOT / "batch4" / f"seed{seed}" / condition)


def acc(metrics_path: Path, test_env: int) -> float | None:
    try:
        with metrics_path.open() as f:
            rec = json.loads(list(f)[-1])
        return rec.get(f"env{test_env}_out_acc")
    except Exception:
        return None


def collect(condition: str) -> dict[str, dict[tuple[int, int], float]]:
    """Return {dataset: {(seed, split): acc_percent}}."""
    out: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for seed in (0, 1, 2, 3):
        base = Path(paths_for(condition, seed))
        if not base.exists():
            continue
        for ds_dir in base.iterdir():
            name = ds_dir.name
            if "_split" not in name:
                continue
            ds, split_part = name.rsplit("_split", 1)
            try:
                split = int(split_part)
            except ValueError:
                continue
            if ds not in DATASETS:
                continue
            m = ds_dir / "metrics.jsonl"
            a = acc(m, split)
            if a is not None:
                out[ds][(seed, split)] = a * 100
    return out


def paired_t(xs: list[float], ys: list[float]) -> tuple[float, float, int]:
    """Paired t-test on xs - ys.  Returns (t, two-sided p, dof).

    Implemented inline so we don't pull scipy as a dep.
    """
    diffs = [x - y for x, y in zip(xs, ys)]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), 0
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("inf") if mean else 0.0, 0.0 if mean else 1.0, n - 1
    t = mean / (sd / math.sqrt(n))
    # Two-sided p from the t distribution using a regularised incomplete-beta
    # via stdlib.  For DG-sized samples (n=16) the approximation below is
    # accurate to ~1e-4.
    df = n - 1
    x = df / (df + t * t)
    # I_x(df/2, 1/2) — Lentz's continued-fraction.  Good for x in (0, 1).
    p = _incomplete_beta(x, df / 2, 0.5)
    return t, max(min(p, 1.0), 0.0), df


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta via Numerical-Recipes continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1.0) < 1e-7:
            break
    return front * h


def fmt(values: list[float]) -> str:
    if not values:
        return "—"
    m = statistics.mean(values)
    if len(values) < 2:
        return f"{m:.2f}"
    sd = statistics.stdev(values)
    return f"{m:.2f} ± {sd:.2f}"


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def main() -> None:
    erm = collect("erm")
    cells = {
        "ERM (all losses off)":            erm,
        "Full HyperDG (alpha_var=0.2)":    collect("baseline"),
        "Curvature frozen (no learning)":  collect("fixed"),
        "No variance reg (alpha_var=0)":   collect("novar"),
    }

    # ── Per-run CSV (for raw inspection) ─────────────────────────────────
    per_run_path = OUT / "outputs" / "batch4_per_run.csv"
    per_run_path.parent.mkdir(parents=True, exist_ok=True)
    with per_run_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "dataset", "seed", "split", "acc"])
        for label, ds_to_runs in cells.items():
            for ds, run_map in ds_to_runs.items():
                for (seed, split), a in sorted(run_map.items()):
                    w.writerow([label, ds, seed, split, f"{a:.4f}"])

    # ── Headline table ──────────────────────────────────────────────────
    lines = []
    lines.append("## Table 1 — Held-out accuracy across HyperDG variants\n")
    lines.append("Mean ± std over 4 seeds × 4 splits = up to 16 runs per cell. "
                 "p-values are paired t-tests vs. ERM, matched on (seed, split).\n")
    header = ["Condition"] + DATASETS + [f"p vs ERM ({d})" for d in DATASETS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|")

    for label, ds_to_runs in cells.items():
        row = [label]
        for ds in DATASETS:
            vals = list(ds_to_runs.get(ds, {}).values())
            row.append(fmt(vals))
        if label == "ERM (all losses off)":
            for _ in DATASETS:
                row.append("—")
        else:
            for ds in DATASETS:
                # Pair on (seed, split)
                ours = ds_to_runs.get(ds, {})
                base = erm.get(ds, {})
                common = sorted(set(ours) & set(base))
                xs = [ours[k] for k in common]
                ys = [base[k] for k in common]
                t, p, df = paired_t(xs, ys)
                row.append(fmt_p(p) + f"  (n={len(common)})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Total runs summary ──────────────────────────────────────────────
    lines.append("**Run count by condition × dataset:**\n")
    lines.append("| Condition | " + " | ".join(DATASETS) + " |")
    lines.append("|" + "|".join(["---"] + [":---:"] * len(DATASETS)) + "|")
    for label, ds_to_runs in cells.items():
        row = [label]
        for ds in DATASETS:
            row.append(str(len(ds_to_runs.get(ds, {}))))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Write back into results_tables.md (only the multi-seed table) ──
    multi_seed_section = "\n".join(lines)

    # Read existing tables file, prepend multi-seed table to keep historical
    existing = OUT / "docs" / "results_tables.md"
    prev = existing.read_text() if existing.exists() else ""
    new_text = (
        "# Results tables (regenerated from full Batch 1+2+3+4 data)\n\n"
        + multi_seed_section
        + "\n\n---\n\n## Historical (seed-0 only) tables follow:\n\n"
        + prev
    )
    existing.write_text(new_text)

    # ── Summary CSV ──────────────────────────────────────────────────────
    summary_path = OUT / "outputs" / "batch4_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "dataset", "n_runs", "mean_acc", "std_acc",
                    "p_vs_erm_paired"])
        for label, ds_to_runs in cells.items():
            for ds in DATASETS:
                runs = ds_to_runs.get(ds, {})
                vals = list(runs.values())
                if not vals:
                    continue
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                if label == "ERM (all losses off)":
                    p = ""
                else:
                    common = sorted(set(runs) & set(erm.get(ds, {})))
                    if len(common) >= 2:
                        xs = [runs[k] for k in common]
                        ys = [erm[ds][k] for k in common]
                        _, p_val, _ = paired_t(xs, ys)
                        p = f"{p_val:.4f}"
                    else:
                        p = ""
                w.writerow([label, ds, len(vals), f"{m:.4f}", f"{sd:.4f}", p])

    print(multi_seed_section)
    print(f"\nWrote:\n  {existing}\n  {summary_path}\n  {per_run_path}")


if __name__ == "__main__":
    main()
