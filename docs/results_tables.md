# Results tables (regenerated from full Batch 1+2+3+4 data)

## Table 1 — Held-out accuracy across HyperDG variants

Mean ± std over 4 seeds × 4 splits = up to 16 runs per cell. p-values are paired t-tests vs. ERM, matched on (seed, split).

| Condition | pacs | vlcs | officehome | terraincognita | p vs ERM (pacs) | p vs ERM (vlcs) | p vs ERM (officehome) | p vs ERM (terraincognita) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ERM (all losses off) | 84.04 ± 7.10 | 75.63 ± 13.32 | 67.05 ± 10.26 | 43.90 ± 8.49 | — | — | — | — |
| Full HyperDG (alpha_var=0.2) | 84.50 ± 7.45 | 75.47 ± 14.14 | 66.03 ± 9.98 | 43.17 ± 4.97 | 0.534  (n=16) | 0.713  (n=16) | 0.013  (n=16) | 0.631  (n=16) |
| Curvature frozen (no learning) | 84.67 ± 7.42 | 76.27 ± 13.17 | 66.13 ± 9.78 | 44.30 ± 6.74 | 0.301  (n=16) | 0.237  (n=16) | 0.020  (n=16) | 0.784  (n=16) |
| No variance reg (alpha_var=0) | 84.57 ± 6.92 | 75.04 ± 14.25 | 66.03 ± 9.75 | 43.90 ± 6.94 | 0.422  (n=16) | 0.348  (n=16) | 0.017  (n=16) | 0.021  (n=16) |

**Run count by condition × dataset:**

| Condition | pacs | vlcs | officehome | terraincognita |
|---|:---:|:---:|:---:|:---:|
| ERM (all losses off) | 16 | 16 | 16 | 16 |
| Full HyperDG (alpha_var=0.2) | 16 | 16 | 16 | 16 |
| Curvature frozen (no learning) | 16 | 16 | 16 | 16 |
| No variance reg (alpha_var=0) | 16 | 16 | 16 | 16 |


---

## Historical (seed-0 only) tables follow:

## Table 1 — Held-out accuracy across HyperDG variants (seed 0, 4 splits)

All numbers are mean ± std across the 4 leave-one-domain-out splits.

| Condition | PACS | VLCS | OfficeHome | TerraIncognita |
|---|---:|---:|---:|---:|
| Full HyperDG (α_curv=0.2) | 84.76 ± 7.55 | 77.49 ± 15.38 | 66.73 ± 10.34 | 43.41 ± 5.45 |
| ERM (all losses off) | 84.09 ± 6.87 | 76.76 ± 14.38 | 67.25 ± 12.02 | 46.13 ± 8.88 |
| Curvature frozen (no learning) | 85.28 ± 7.48 | 76.94 ± 14.82 | 65.64 ± 11.18 | 46.49 ± 8.46 |
| No variance reg (α_curv=0) | 83.70 ± 8.22 | 75.36 ± 16.11 | 65.63 ± 10.99 | 44.34 ± 3.20 |

## Table 2 — Component leave-one-out (seed 0, 4 splits)

Each row drops ONE component from the full HyperDG loss.

| Drop component | PACS | OfficeHome |
|---|---:|---:|
| none (full HyperDG) | 84.76 ± 7.55 | 66.73 ± 10.34 |
| alignment loss | 84.80 ± 7.91 | 68.14 ± 9.70 |
| sharpness loss | 87.06 ± 7.02 | 66.14 ± 10.90 |
| flatness regularization | 84.25 ± 8.17 | 66.70 ± 11.81 |
| variance regularization | 83.70 ± 8.22 | 65.63 ± 10.99 |
| curvature learning | 85.28 ± 7.48 | 65.64 ± 11.18 |
| everything (= ERM) | 84.09 ± 6.87 | 67.25 ± 12.02 |

## Table 3 — Curvature variance penalty (λ_var) sweep on PACS (seed 0, 4 splits)

| λ_var | PACS held-out acc |
|---:|---:|
| 0.00 (none) | 83.70 ± 8.22 |
| 0.05 | 83.78 ± 9.10 |
| 0.20 (default) | 84.76 ± 7.55 |
| 0.50 | 85.92 ± 6.09 |
| 2.00 (very strong) | 86.09 ± 7.63 |

## Table 4 — Final per-domain curvature spread (max − min)

Spread is measured at the last training step, averaged across the 4 splits.

| Dataset | α_var = 0 (none) | α_var = 0.2 (default) | Ratio |
|---|---:|---:|---:|
| PACS | 0.0041 | 0.0023 | 0.55× |
| VLCS | 0.0040 | 0.0028 | 0.68× |
| OFFICEHOME | 0.0011 | 0.0011 | 1.03× |
| TERRAINCOGNITA | 0.0064 | 0.0045 | 0.71× |
