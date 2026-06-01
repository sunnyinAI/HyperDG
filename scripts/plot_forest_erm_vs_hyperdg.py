#!/usr/bin/env python3
"""Forest plot of the paired ERM-vs-HyperDG comparison.

Shows, for each dataset and each HyperDG variant:
  * mean (HyperDG - ERM) accuracy difference across 16 matched
    (seed, split) pairs,
  * 95% confidence interval of that difference (paired t).

A vertical line at 0 marks "no effect."  Points are colored by
significance at α = 0.05 (paired t-test, two-sided).

Output: outputs/forest_erm_vs_hyperdg.pdf
Also copies to HyperDG_TMLR/fig_forest_erm_vs_hyperdg.pdf.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/sunny/HyperDG")

DATASETS = ["pacs", "vlcs", "officehome", "terraincognita"]
DS_LABEL = {"pacs": "PACS", "vlcs": "VLCS",
            "officehome": "OfficeHome", "terraincognita": "TerraIncognita"}

# Re-derive (cond, seed) -> experiment root mapping (same as aggregate_batch4)
def paths_for(condition: str, seed: int) -> str:
    if seed == 0:
        return {
            "erm":      str(ROOT/"experiments/batch2/erm"),
            "baseline": str(ROOT/"experiments/batch3/baseline"),
            "fixed":    str(ROOT/"experiments/batch1/fixed/frozen"),
            "novar":    str(ROOT/"experiments/batch1/novar/a0"),
        }[condition]
    return str(ROOT/f"experiments/batch4/seed{seed}/{condition}")


def acc(metrics_path: Path, test_env: int):
    try:
        with metrics_path.open() as f:
            rec = json.loads(list(f)[-1])
        return rec.get(f"env{test_env}_out_acc")
    except Exception:
        return None


def collect(condition: str) -> dict[str, dict[tuple[int, int], float]]:
    out = defaultdict(dict)
    for seed in (0, 1, 2, 3):
        base = Path(paths_for(condition, seed))
        if not base.exists(): continue
        for d in base.iterdir():
            name = d.name
            if "_split" not in name: continue
            ds, split_part = name.rsplit("_split", 1)
            try:
                split = int(split_part)
            except ValueError:
                continue
            if ds not in DATASETS: continue
            m = d / "metrics.jsonl"
            a = acc(m, split)
            if a is not None:
                out[ds][(seed, split)] = a * 100
    return out


def incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x)*a + math.log(1-x)*b - lbeta) / a
    qab, qap, qam = a+b, a+1, a-1
    c, d = 1.0, 1.0 - qab*x/qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0/d; h = d
    for m in range(1, 200):
        m2 = 2*m
        aa = m*(b-m)*x / ((qam+m2)*(a+m2))
        d = 1 + aa*d
        if abs(d) < 1e-30: d = 1e-30
        c = 1 + aa/c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        d = 1 + aa*d
        if abs(d) < 1e-30: d = 1e-30
        c = 1 + aa/c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0/d; h *= d*c
        if abs(d*c - 1) < 1e-7: break
    return front*h


def two_sided_p_from_t(t: float, df: int) -> float:
    if df < 1 or math.isnan(t): return float("nan")
    x = df / (df + t*t)
    return max(0.0, min(1.0, incomplete_beta(x, df/2, 0.5)))


# 95% two-sided t critical values, df = n-1.  Hard-coded for n=16 (df=15).
T_CRIT_95 = {15: 2.1314, 14: 2.1448, 13: 2.1604, 12: 2.1788}


def paired_diff(xs: list[float], ys: list[float]):
    diffs = [a - b for a, b in zip(xs, ys)]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    t_crit = T_CRIT_95.get(n - 1, 2.13)
    ci_half = t_crit * se
    t_stat = mean / se if se else float("inf") if mean else 0.0
    p = two_sided_p_from_t(t_stat, n - 1)
    return mean, ci_half, p, n


def main() -> None:
    erm = collect("erm")
    conds = [
        ("Full HyperDG",     collect("baseline")),
        ("Curvature frozen", collect("fixed")),
        ("No variance reg",  collect("novar")),
    ]
    cond_colors = {
        "Full HyperDG":     "#C44E52",
        "Curvature frozen": "#4C72B0",
        "No variance reg":  "#55A868",
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    rows = []
    y_pos = 0
    yticks, yticklabels = [], []
    for ds in reversed(DATASETS):       # plot top-to-bottom
        ax.axhspan(y_pos - 0.5, y_pos + len(conds) - 0.5,
                   facecolor="0.97" if (DATASETS.index(ds) % 2) else "1.0",
                   zorder=0)
        ds_centre = y_pos + (len(conds) - 1) / 2
        yticks.append(ds_centre); yticklabels.append(DS_LABEL[ds])
        for i, (label, runs) in enumerate(conds):
            ours = runs.get(ds, {})
            base = erm.get(ds, {})
            common = sorted(set(ours) & set(base))
            xs = [ours[k] for k in common]
            ys = [base[k] for k in common]
            mean, ci, p, n = paired_diff(xs, ys)
            sig = (not math.isnan(p)) and (p <= 0.05)
            color = cond_colors[label]
            ax.errorbar(mean, y_pos, xerr=ci,
                        fmt="o", color=color,
                        markersize=8 if sig else 6,
                        markeredgecolor="black" if sig else "none",
                        markeredgewidth=1.2 if sig else 0,
                        capsize=4, elinewidth=1.4, capthick=1.2)
            star = r"$\bigstar$" if sig else ""
            text = f"{mean:+.2f}  (p={p:.3f}{', ' if sig else ', '}n={n}) {star}"
            ax.text(mean + ci + 0.18, y_pos, text,
                    va="center", fontsize=7.5)
            rows.append((ds, label, mean, ci, p, n))
            y_pos += 1
        y_pos += 0.5

    ax.axvline(0, color="black", lw=1.0, linestyle="--", alpha=0.6)
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("HyperDG variant accuracy  $-$  ERM accuracy   (percentage points)",
                  fontsize=9)
    ax.set_title("Paired effect of HyperDG variants vs. ERM "
                 "(matched on seed $\\times$ split, n = 16)\n"
                 "error bar = 95% CI of paired difference; "
                 "$\\bigstar$ = significant at $\\alpha = 0.05$",
                 fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Move legend above the plot to avoid colliding with bottom-row labels
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="",
                           markersize=7, label=l)
               for l, c in cond_colors.items()]
    ax.legend(handles=handles, fontsize=8,
              loc="lower center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, ncol=3)
    # Stretch x-axis a bit so the right-hand annotations don't get cut off
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.4)

    out_pdf = ROOT/"outputs/forest_erm_vs_hyperdg.pdf"
    fig.tight_layout()
    fig.savefig(str(out_pdf), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {out_pdf}")

    # Print summary table
    print("\nDataset           Condition           mean Δ   ±CI95   p")
    for ds, lbl, m, c, p, n in rows:
        sig = "*" if (not math.isnan(p)) and p <= 0.05 else " "
        print(f"  {ds:<16}{lbl:<20}{m:+6.2f}  ±{c:5.2f}  {p:.3f}{sig}")


if __name__ == "__main__":
    main()
