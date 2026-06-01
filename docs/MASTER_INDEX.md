# HyperDG resubmission — Master index of runs, results, and files

**Purpose.** This is the single source of truth for the four reviewer-response batches
we ran for the TMLR resubmission.  Every claim in the rewrite is traceable
through this document to a specific config, metrics file, plot, or
aggregation script.

**One-line story.** 292 runs across four batches show that the full HyperDG
framework is statistically indistinguishable from ERM through the same
codepath on three of four DG benchmarks and statistically *worse* than
ERM on the fourth (OfficeHome).

**Top-level directories used in this index** (all under `/data/sunny/HyperDG/`):

```
configs/   batch1/  batch2/  batch3/  batch4/      # YAML configs (236 files)
experiments/ batch1/  batch2/  batch3/  batch4/    # checkpoints + metrics.jsonl (292 dirs)
logs/      batch1/  batch2/  batch3/  batch4/      # per-run stdout logs
scripts/                                            # generation + plotting + aggregation
outputs/                                            # final figures + summary CSVs
docs/                                               # rewrite_draft.md, results_tables.md, this file
```

---

## 1.  All 292 runs — bird's-eye view

| Batch | Purpose | # runs | Datasets | Config root | Experiment root | Launcher |
|---|---|---:|---|---|---|---|
| **Batch 1** | First-pass ablations for the AE's two open questions: (a) per-domain α with λ_var = 0, (b) curvature frozen, (c) λ_var sweep on PACS | 44 | PACS, VLCS, OH, Terra (+ PACS only for λ_var sweep) | `configs/batch1/{novar,fixed,lvar}/` | `experiments/batch1/{novar/a0,fixed/frozen,lvar/aXpYY}/` | `scripts/run_batch1_gpu{0,3}.sh` |
| **Batch 2** | Component leave-one-out (LOO) on the HyperDG loss: ERM, −align, −sharp, −flat | 40 | ERM on all 4; LOO on PACS + OH only | `configs/batch2/{erm,no_align,no_sharp,no_flat}/` | `experiments/batch2/<family>/` | `scripts/run_batch2_gpu{0,1,3}.sh` |
| **Batch 3** | Full HyperDG baseline (α_var = 0.2) with per-domain curvature logging, so curvature trajectories can be paired with Batch 1 novar | 16 | PACS, VLCS, OH, Terra | `configs/batch3/baseline/` | `experiments/batch3/baseline/` | `scripts/run_batch3_gpu{0,1,3}.sh` |
| **Batch 4** | Multi-seed (seeds 1, 2, 3) for variance bars and paired t-tests vs ERM | 192 | PACS, VLCS, OH, Terra × 4 conditions × 4 splits × 3 seeds | `configs/batch4/seed{1,2,3}/{erm,baseline,fixed,novar}/` | `experiments/batch4/seed{1,2,3}/<cond>/` | `scripts/run_batch4_gpu{0,1,3}.sh` |

Each per-run output directory contains:

```
metrics.jsonl       # one JSON record per checkpoint (every 200 steps; 25 records per 5000-step run)
model_last.pt       # final-step checkpoint (used by all aggregation scripts)
```

`metrics.jsonl` records have these keys we rely on:

| Key | Meaning |
|---|---|
| `step` | training step (0–4999) |
| `env{i}_in_acc` | training-domain in-split accuracy on env i |
| `env{i}_out_acc` | training-domain out-split accuracy on env i (used for *training-domain* model selection) |
| `curvature_domain_{0,1,2,3}` | per-domain curvature κ_i = softplus(log_κ) + ε at this step |
| `curvature_mean`, `curvature_std`, `curvature_var` | aggregates across domains |
| `mean_curvature`, `global_curvature` | scalar curvature aggregates |
| `loss_sup`, `loss_align`, `loss_sharp`, `loss_local_flat`, `loss_global_flat`, `loss_curv`, `loss_total` | per-component loss values |
| `train_acc`, `single_curvature_mode` | bookkeeping |

The held-out (target-domain) accuracy used in every results table is
**`env{test_env}_out_acc` from the LAST line of metrics.jsonl**, where
`test_env` matches the split index in the directory name.

---

## 2.  Batch 1 — primary ablations (44 runs)

**Question 1 (AE):** what happens to per-domain curvatures *without*
variance regularization?  → `novar` family.

**Question 2 (us):** does freezing curvature hurt?  → `fixed` family.

**Question 3 (sensitivity):** how does accuracy vary with λ_var on PACS?
→ `lvar` family.

### 2.1  `novar` — per-domain α with α_curv = 0

| | Path |
|---|---|
| Configs | `configs/batch1/novar/{pacs,vlcs,officehome,terraincognita}_split{0..3}.yaml`  (16 files) |
| Generator | `scripts/generate_batch1_configs.py` |
| Outputs | `experiments/batch1/novar/a0/{ds}_split{n}/{metrics.jsonl, model_last.pt}` |
| Logs | `logs/batch1/gpu{0,3}/novar_{ds}_split{n}.log` |

Key finding (from `outputs/curvature_summary.csv`):

| Dataset | Final per-domain curvature spread (mean across 4 splits) |
|---|---:|
| PACS | 0.0041 |
| VLCS | 0.0040 |
| OfficeHome | 0.0011 |
| TerraIncognita | 0.0064 |

→ Without any penalty, curvatures naturally cluster to within <1.5% of the
initial value (κ_init = softplus(0)+1e-4 ≈ 0.6932).

### 2.2  `fixed` — curvature frozen at init (κ ≈ 0.6932)

| | Path |
|---|---|
| Configs | `configs/batch1/fixed/{ds}_split{n}.yaml`  (16 files) |
| Outputs | `experiments/batch1/fixed/frozen/{ds}_split{n}/...` |
| Logs | `logs/batch1/gpu{0,3}/fixed_{ds}_split{n}.log` |
| Hparam change vs baseline | `curvature_lr = 0.0`, `alpha_curv = 0.0` |

Key finding: matches or beats full HyperDG on every benchmark — curvature
learning machinery does no work that gradient descent on the rest of the
model can't already do.

### 2.3  `lvar` — variance-penalty sweep on PACS

| | Path |
|---|---|
| Configs | `configs/batch1/lvar/a{0p05,0p5,2p0}_pacs_split{0..3}.yaml`  (12 files) |
| Outputs | `experiments/batch1/lvar/{a0p05,a0p5,a2p0}/pacs_split{n}/...` |
| Logs | `logs/batch1/gpu{0,3}/lvar_a{0p05,0p5,2p0}_pacs_split{n}.log` |

Sweep result (PACS, seed 0, 4 splits — see `docs/results_tables.md` historical section):

| λ_var | PACS held-out acc |
|---:|---:|
| 0.00 (none) | 83.70 ± 8.22 |
| 0.05 | 83.78 ± 9.10 |
| 0.20 (paper default) | 84.76 ± 7.55 |
| 0.50 | 85.92 ± 6.09 |
| 2.00 (very strong) | 86.09 ± 7.63 |

→ Performance varies ~2 pp across a 40× sweep of λ_var, all within the
±6–9 pp cross-split noise floor.

---

## 3.  Batch 2 — component leave-one-out (40 runs)

Each row of the LOO table drops ONE component of the HyperDG loss and
keeps all others at default.

| Family | Hparam change | Datasets | # runs |
|---|---|---|---:|
| `erm` | All coefficients = 0 *and* `curvature_lr = 0` (pure ERM through HyperDG codepath) | all 4 | 16 |
| `no_align` | `beta_align = 0` | PACS + OH | 8 |
| `no_sharp` | `gamma_sharp = 0` | PACS + OH | 8 |
| `no_flat` | `delta_local_flat = 0` and `delta_global_flat = 0` | PACS + OH | 8 |

| | Path |
|---|---|
| Configs | `configs/batch2/{erm,no_align,no_sharp,no_flat}/{ds}_split{n}.yaml`  (40 files) |
| Generator | `scripts/generate_batch2_configs.py` |
| Outputs | `experiments/batch2/{family}/{ds}_split{n}/...` |
| Logs | `logs/batch2/gpu{0,1,3}/{family}_{ds}_split{n}.log` |

Headline (seed 0; for multi-seed see §5):

| Drop component | PACS | OfficeHome |
|---|---:|---:|
| none (full HyperDG) | 84.76 ± 7.55 | 66.73 ± 10.34 |
| alignment loss | 84.80 ± 7.91 | 68.14 ± 9.70 |
| sharpness loss | **87.06 ± 7.02** | 66.14 ± 10.90 |
| flatness regularization | 84.25 ± 8.17 | 66.70 ± 11.81 |
| variance regularization (= novar) | 83.70 ± 8.22 | 65.63 ± 10.99 |
| curvature learning (= fixed) | 85.28 ± 7.48 | 65.64 ± 11.18 |
| **everything (= ERM)** | **84.09 ± 6.87** | **67.25 ± 12.02** |

→ Removing the sharpness loss *improves* PACS by ~2.3 pp.  No single
component contributes a measurable positive gain.

---

## 4.  Batch 3 — full-HyperDG baseline with curvature logging (16 runs)

The pre-existing `experiments/hyperdg_pacs/`-style baselines only logged
*aggregate* curvature stats.  Batch 3 re-runs the default HyperDG
configuration (α_curv = 0.2 + all defaults) with per-domain logging so
trajectories can be plotted alongside Batch 1's `novar` arm.

| | Path |
|---|---|
| Configs | `configs/batch3/baseline/{ds}_split{n}.yaml`  (16 files) |
| Generator | `scripts/generate_batch3_configs.py` |
| Outputs | `experiments/batch3/baseline/{ds}_split{n}/...` |
| Logs | `logs/batch3/gpu{0,1,3}/baseline_{ds}_split{n}.log` |
| Launchers | `scripts/run_batch3_gpu{0,1,3}.sh` (these *wait* for Batch-2 master logs to print "All assigned runs finished" before starting) |

---

## 5.  Batch 4 — multi-seed (192 runs)

**4 conditions × 4 datasets × 4 splits × 3 seeds = 192 runs.**  Seed 0 is
covered by the relevant Batch 1/2/3 entry (see §6 for the mapping).

| Condition | Hparam change vs reference |
|---|---|
| `erm` | All coefficients = 0, `curvature_lr = 0` |
| `baseline` | Defaults (α_curv = 0.2, β = 0.5, γ = 0.05, δ_{local,global} = 0.1) |
| `fixed` | `curvature_lr = 0` and `alpha_curv = 0` (curvature frozen + no spread penalty needed) |
| `novar` | `alpha_curv = 0` (per-domain curvature still learned) |

| | Path |
|---|---|
| Configs | `configs/batch4/seed{1,2,3}/{erm,baseline,fixed,novar}/{ds}_split{n}.yaml`  (192 files) |
| Generator | `scripts/generate_batch4_configs.py` |
| Outputs | `experiments/batch4/seed{S}/{cond}/{ds}_split{n}/...` |
| Logs | `logs/batch4/gpu{0,1,3}/seed{S}_{cond}_{ds}_split{n}.log` |
| Launcher generator | `scripts/generate_batch4_launchers.py` |
| Launchers | `scripts/run_batch4_gpu{0,1,3}.sh` (round-robin assignment, ~64 runs each, ~27h per GPU) |

### 5.1  Headline multi-seed table (n = 16 = 4 seeds × 4 splits per cell)

This is the **load-bearing table for the rewritten paper**.  Numbers from
`docs/results_tables.md`, top section.  Aggregation script:
`scripts/aggregate_batch4.py`.  Pairing for the t-test is by (seed, split).

| Condition | PACS | VLCS | OfficeHome | TerraIncognita |
|---|---:|---:|---:|---:|
| ERM (all losses off) | 84.04 ± 7.10 | 75.63 ± 13.32 | **67.05 ± 10.26** | 43.90 ± 8.49 |
| Full HyperDG (α_var = 0.2) | 84.50 ± 7.45 | 75.47 ± 14.14 | **66.03 ± 9.98** | 43.17 ± 4.97 |
| Curvature frozen | 84.67 ± 7.42 | 76.27 ± 13.17 | 66.13 ± 9.78 | 44.30 ± 6.74 |
| No variance reg | 84.57 ± 6.92 | 75.04 ± 14.25 | 66.03 ± 9.75 | 43.90 ± 6.94 |

Paired t-test p-values vs ERM (`scripts/aggregate_batch4.py` computes these inline):

| | PACS | VLCS | OfficeHome | Terra |
|---|---:|---:|---:|---:|
| Full HyperDG vs ERM | 0.534 | 0.713 | **0.013** | 0.631 |
| Frozen curvature vs ERM | 0.301 | 0.237 | **0.020** | 0.784 |
| No variance reg vs ERM | 0.422 | 0.348 | **0.017** | 0.021 |

**Bold = significant at α = 0.05.**  All three HyperDG variants are
statistically *worse* than ERM on OfficeHome; the other three benchmarks
show no significant effect.

---

## 6.  Mapping: seed 0 ↔ batch directories

Batch 4 covers seeds 1–3 only.  The seed-0 runs already exist in earlier
batches — `aggregate_batch4.py` knows this and pulls them automatically.
For reference:

| Condition | Seed-0 location | Seed-{1,2,3} location |
|---|---|---|
| `erm` | `experiments/batch2/erm/{ds}_split{n}/` | `experiments/batch4/seed{1,2,3}/erm/{ds}_split{n}/` |
| `baseline` | `experiments/batch3/baseline/{ds}_split{n}/` | `experiments/batch4/seed{1,2,3}/baseline/{ds}_split{n}/` |
| `fixed` | `experiments/batch1/fixed/frozen/{ds}_split{n}/` | `experiments/batch4/seed{1,2,3}/fixed/{ds}_split{n}/` |
| `novar` | `experiments/batch1/novar/a0/{ds}_split{n}/` | `experiments/batch4/seed{1,2,3}/novar/{ds}_split{n}/` |

---

## 7.  Figures

| Figure | File on disk | Generator | Notes |
|---|---|---|---|
| **Fig. 2 (paper headline) — per-domain curvature trajectories, all 4 datasets** | `outputs/curvature_trajectories.pdf` (grid) and `outputs/curvature_trajectories_{ds}.pdf` (per dataset) | `scripts/plot_curvature_trajectories.py` | Pulls from Batch 1 `novar` (left column) and Batch 3 `baseline` (right column).  Each panel: 4 splits × 3 domains = 12 lines.  Dashed horizontal at κ_init = 0.6932. |
| Curvature spread CSV | `outputs/curvature_summary.csv` | same script | Final-step max−min spread, mean across splits, both conditions |
| **Fig. 3 (supplementary) — class-prototype Poincaré disks** | `outputs/poincare_hyperdg_{pacs,vlcs,officehome,terra,domainnet}_{perdomain,singlecurv}.pdf` | `scripts/plot_poincare_disk.py` (fixed in this session — see §10) | 20 plots total.  `_perdomain` = per-domain α, `_singlecurv` = single α.  MERU variants also available (`poincare_meru_*.pdf`). |
| Curvature evolution (single PACS run, pre-existing) | `outputs/curvature_per_domain_pacs.pdf` | `scripts/plot_curvature_per_domain.py` | Kept from prior submission |
| Geometry overview (pre-existing) | `outputs/fig1_hyperdg_geometry.png` | `scripts/plot_hyperdg_geometry.py` | Kept from prior submission |

### 7.1  Notable fix this session — Poincaré plot

The original `scripts/plot_poincare_disk.py` did per-row L2 normalisation
before the exp-map.  This forces every point to the same Poincaré
radius, producing a useless thin-ring artefact.  Fixed by switching to
**global quantile-based scaling** (`tangent_to_poincare_disk` now uses
`radius_quantile = 0.98` so a single outlier doesn't squash the scene).
The fix is also visible in the way class centroids are now rendered
(white-edged hollow rings).  Both `outputs/poincare_test_fixed.pdf` and
the regenerated batch demonstrate the fix.

---

## 8.  Tables (Markdown sources for paste-into-LaTeX)

| Table | File on disk | Section in `docs/rewrite_draft.md` |
|---|---|---|
| **T1 — multi-seed headline (load-bearing)** | `docs/results_tables.md` top section | §3.4 of the rewrite |
| T2 — component LOO (seed-0 only currently) | `docs/results_tables.md` historical §2 | §3.5 |
| T3 — λ_var sweep on PACS | `docs/results_tables.md` historical §3 | §3.8 |
| T4 — curvature spread with vs without variance reg | `docs/results_tables.md` historical §4 + `outputs/curvature_summary.csv` | §3.6 |
| T5 — paired t-tests vs ERM | embedded in T1; raw values in `outputs/batch4_summary.csv` | §3.4 |
| Run-level dump (every individual run) | `outputs/batch4_per_run.csv` | not in paper, but useful for any reviewer-requested re-analysis |

---

## 9.  Scripts catalogue

All under `scripts/`.  Group by what they do.

### 9.1  Generation

| Script | Produces |
|---|---|
| `generate_batch1_configs.py` | 44 configs for `novar` / `fixed` / `lvar` families |
| `generate_batch2_configs.py` | 40 LOO configs |
| `generate_batch3_configs.py` | 16 baseline (α_curv = 0.2) configs |
| `generate_batch4_configs.py` | 192 multi-seed configs across 4 conditions |
| `generate_batch4_launchers.py` | 3 round-robin GPU launchers for Batch 4 |

### 9.2  Launchers (hand-written)

`run_batch{1..3}_gpu{0,1,3}.sh` — each one source-of-truth for which
runs go to which GPU.  All use `CUDA_DEVICE_ORDER=PCI_BUS_ID` so
`CUDA_VISIBLE_DEVICES={0,1,3}` map to Ada cards (the Blackwell on
physical index 2 is **incompatible** with this PyTorch build —
torch 2.6.0+cu124, compiled sm_50..sm_90; Blackwell needs sm_120).

Skip-on-checkpoint logic in every launcher: if
`experiments/<batch>/.../<run_dir>/model_last.pt` already exists, the
launcher prints `SKIP (done)` and moves on.  So all launchers are
re-runnable / resumable.

To survive a shell disconnect (Claude session reset, terminal close),
launch with `setsid nohup bash <script>.sh > <log> 2>&1 < /dev/null &`.
This was important for the Batch-4 relaunch after the initial run died
when the shell was terminated.

### 9.3  Aggregation + plotting

| Script | Output |
|---|---|
| `aggregate_batch4.py` | Multi-seed table + paired t-tests → `docs/results_tables.md` (prepends multi-seed section), `outputs/batch4_summary.csv`, `outputs/batch4_per_run.csv` |
| `plot_curvature_trajectories.py` | `outputs/curvature_trajectories.pdf` (grid) + per-dataset PDFs + `outputs/curvature_summary.csv` |
| `plot_poincare_disk.py` | Per-checkpoint Poincaré disks (see §7) — fixed in this session |
| `plot_all_poincare.sh` | Regenerates all 20 Poincaré plots in one pass |
| `aggregate_metrics.py` | Original per-split aggregator (kept for backward-compat) |

---

## 10.  Sanity-check commands

Copy-paste any of these to re-verify state.

```bash
# Total checkpoint count per batch
for b in batch1 batch2 batch3 batch4; do
  n=$(find /data/sunny/HyperDG/experiments/$b -name model_last.pt | wc -l)
  echo "$b: $n"
done
# Expected: batch1=44, batch2=40, batch3=16, batch4=192

# Re-generate every table and CSV from the raw metrics:
~/.conda/envs/sftdg/bin/python3 /data/sunny/HyperDG/scripts/aggregate_batch4.py
~/.conda/envs/sftdg/bin/python3 /data/sunny/HyperDG/scripts/plot_curvature_trajectories.py

# Re-generate every Poincaré plot:
bash /data/sunny/HyperDG/scripts/plot_all_poincare.sh --device cuda:1

# Pull held-out acc for any single run:
~/.conda/envs/sftdg/bin/python3 -c "
import json; from pathlib import Path
p = Path('experiments/batch4/seed1/baseline/pacs_split0/metrics.jsonl')
rec = json.loads(list(p.open())[-1])
print('env0_out:', rec['env0_out_acc'])
"
```

---

## 11.  Documents written during this resubmission cycle

| File | Purpose |
|---|---|
| `docs/MASTER_INDEX.md` | **(this file)** Single reference linking every claim to its source files |
| `docs/rewrite_draft.md` | Drop-in rewrite of abstract, contributions, results §, discussion, limitations, cover-letter template.  Headline numbers updated to multi-seed. |
| `docs/results_tables.md` | All tables in Markdown form, ready to convert to LaTeX |
| `docs/roadmap.md` | Pre-existing roadmap (not touched) |

---

## 12.  Open items (not blocking submission, but worth noting)

| # | Item | Notes |
|---|---|---|
| 14 | **Paper-vs-rerun number gap.** Original paper reports PACS 92.2, our reruns (same configs) land at ~84.5.  Best-val checkpoint selection closes <1 pp of the gap. | `docs/rewrite_draft.md` §9 recommends reporting the corrected numbers transparently rather than chasing the original protocol. |
| 15 | **MERU + ERM ablation.**  Would extend the negative result to hyperbolic-pretrained features.  ~16 runs, ~4h on 3 GPUs.  Existing MERU configs at `configs/reviewer_hQnG/{meru,meru_single}/`; copy and zero out the loss coefficients. | Not load-bearing — paired t-tests on ResNet-50 already make the case. |
| — | **WILDS Camelyon17 split-4 outlier (63.86%).**  Already in earlier rebuttal; mention explicitly in limitations. | `docs/rewrite_draft.md` §5 mentions this. |
| — | **DomainNet for new conditions.**  Currently only seed-0 baseline exists. | Mark as "preliminary, in supplementary" in rewrite. |

---

## 13.  Quick provenance — *"where did this number come from?"*

If a referee or co-author challenges any cell in the paper, the lookup
chain is:

```
paper Table X cell → docs/results_tables.md row →
  paths_for(condition, seed) in scripts/aggregate_batch4.py →
  experiments/<batch>/.../<ds>_split<n>/metrics.jsonl (last line) →
  rec[f"env{test_env}_out_acc"]
```

For figures:

```
paper Figure N → outputs/<file>.pdf →
  scripts/plot_*.py (its own header docstring explains inputs) →
  experiments/<batch>/.../metrics.jsonl
```

Every result in the rewrite is reconstructable from the raw `metrics.jsonl`
files alone — the aggregation and plot scripts hold no state.
