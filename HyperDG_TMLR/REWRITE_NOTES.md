# HyperDG paper rewrite — change log and reading guide

This file documents the rewrite of `main.tex` and `supplementary.tex`
for the TMLR resubmission. Originals are preserved as
`main.tex.backup_<timestamp>` and `supplementary.tex.backup_<timestamp>`.

## What changed at a glance

| Section | Old | New |
|---|---|---|
| Title | "HyperDG: Hyperbolic Representation Alignment ..." (positive framing) | "Does Per-Domain Hyperbolic Curvature Help Domain Generalization? A Controlled Test on a Pretrained Backbone" |
| Abstract | "consistent improvements in accuracy, robustness, and OOD performance" | Honest: statistically indistinguishable from ERM on 3/4 benchmarks, significantly worse on OfficeHome |
| §1 Introduction | Pitches HyperDG as the solution | Frames as a rigorous test of the per-domain hyperbolic conjecture; ends with negative finding |
| §1 Contributions | "We propose HyperDG / self-aligned optimization / curvature-consistency regularizers" | New 5-bullet list: framework, controlled multi-seed test, LOO source-of-gains decomposition, curvature-trajectory diagnostic, full reproducibility release |
| §2 Related Work | unchanged | unchanged |
| §3 Methodology | unchanged | unchanged (method is the apparatus we tested) |
| §4 Experiments | "HyperDG surpasses SFT by +2.7% ..." with the 92.2 PACS row | Controlled multi-seed comparison: full HyperDG vs ERM vs frozen vs no-var, all with paired t-tests |
| §4 ablation tables (`abl_officehome`, `ablation_lfb_lvar`) | Numbers claim ablation gaps that single-seed runs cannot support | Replaced by new leave-one-out (LOO) table on PACS + OH, and an explicit acknowledgement that the earlier single-seed gaps were not reproducible |
| Curvature trajectory figure | Did not exist | New Figure 2: per-domain curvature trajectories across 4 datasets, with/without variance reg |
| Curvature sensitivity figure (Fig `curv_plots`) | Claimed as evidence of generalization benefit | Kept but reframed as *descriptive* geometric stability, not as a generalization claim |
| §5 Discussion | did not exist | NEW: four candidate explanations for why the hyperbolic story doesn't land here, and what a strong positive test would look like |
| §6 Limitations | did not exist | NEW: single-architecture FT, benchmark coverage, locality of the negative result, the 7-10pp gap between earlier reported numbers and our reruns |
| §7 Conclusion | "HyperDG achieves consistent and superior generalization ..." | Honest negative-result conclusion + invitation to test stronger positive instantiations |
| Appendix theoretical analysis | Bounds implicitly framed as predicting the claimed gain | Prefaced with a reader's note explaining that this was the *motivating intuition* and that the empirical test came back negative; the math is kept as a record of the conjecture |
| Supplementary - per-dataset trajectory figs | did not exist | NEW: 4 dedicated trajectory figures, one per dataset |
| Supplementary - single vs per-domain table | Had old 92.2 / 81.8 / 76.8 / 53.1 numbers we cannot reproduce | Replaced with our reproducible numbers (PACS 84.76, VLCS 77.49, OH 66.73, Terra 43.41) |
| Supplementary - acknowledgement | "thank reviewers, HyperDG is robust" | Reframed: thanks AE for invitation to embrace the negative finding |

## New figures added to the paper directory

All are in `/data/sunny/HyperDG/HyperDG_TMLR/`:

| File | Used in | Source |
|---|---|---|
| `fig_curvature_trajectories_grid.pdf` | main.tex Figure 2 (Fig `curv_trajectories`) | `outputs/curvature_trajectories.pdf` |
| `fig_curvature_trajectories_pacs.pdf` | supplementary.tex Figure `supp_traj_pacs` | per-dataset variant |
| `fig_curvature_trajectories_vlcs.pdf` | supplementary.tex Figure `supp_traj_vlcs` | per-dataset variant |
| `fig_curvature_trajectories_officehome.pdf` | supplementary.tex Figure `supp_traj_oh` | per-dataset variant |
| `fig_curvature_trajectories_terra.pdf` | supplementary.tex Figure `supp_traj_terra` | per-dataset variant |

Regenerate any of these via:
```bash
cd /data/sunny/HyperDG
~/.conda/envs/sftdg/bin/python3 scripts/plot_curvature_trajectories.py
cp outputs/curvature_trajectories.pdf            HyperDG_TMLR/fig_curvature_trajectories_grid.pdf
cp outputs/curvature_trajectories_pacs.pdf       HyperDG_TMLR/fig_curvature_trajectories_pacs.pdf
cp outputs/curvature_trajectories_vlcs.pdf       HyperDG_TMLR/fig_curvature_trajectories_vlcs.pdf
cp outputs/curvature_trajectories_officehome.pdf HyperDG_TMLR/fig_curvature_trajectories_officehome.pdf
cp outputs/curvature_trajectories_terraincognita.pdf HyperDG_TMLR/fig_curvature_trajectories_terra.pdf
```

## New bib entries (appended to `main.bib`)

- `fan2025curvaturelearning` — IJCV 2025 (reviewer qLsj asked us to cite)
- `bi2025hyperbolic` — AAAI 2025 (reviewer qLsj asked us to cite)
- `desai2023meru` — ICML 2023 (reviewer hQnG asked us to cite)
- `chami2020trees` — NeurIPS 2020 (additional context for §5 Discussion)

## Tables that changed numbers

| Table | Old numbers (paper) | New numbers (our reruns) |
|---|---|---|
| `tab:fedcuri_dg` HyperDG row | PACS 92.2 / VLCS 82.4 / OH 76.8 / Terra 53.9 / DomainNet 52.1 | Replaced with 4 controlled-protocol rows (ERM, full, frozen, no var.) with $n{=}16$ and paired $t$-tests |
| `tab:loo` (new) | did not exist | LOO accuracy on PACS + OH, seed 0, 4 splits |
| `tab:lvar_sweep` (new) | did not exist | $\lambda_{\mathrm{var}}$ sweep on PACS |
| `tab:curv_spread` (new) | did not exist | Final-step curvature spread with/without variance reg, 4 datasets |
| `tab:abl_officehome` and `tab:ablation_lfb_lvar` | Single-seed ablation gaps | Deleted; replaced by §4.3 ("Note on prior single-seed ablation numbers") which explicitly disavows those single-seed gaps |
| Supplementary single-vs-multi-curvature table | PACS 92.2 / VLCS 81.8 / OH 76.8 / DomainNet 52.1 / Terra 53.1 | PACS 84.76 / VLCS 77.49 / OH 66.73 / Terra 43.41 / DomainNet 52.1\* (unreplicated) |

## Section structure after the rewrite

```
Title
Abstract                                            [REWRITTEN]
§1 Introduction                                     [REWRITTEN]
    contributions block                             [REWRITTEN]
§2 Related Work                                     [unchanged]
§3 Methodology                                      [unchanged]
    §3.1 Lorentz Hyperbolic Embedding
    §3.2 Cross-Domain Alignment in a Shared Tangent Space
    §3.3 Curvature Refinement via Variance Regularization
    §3.4 Two-Phase Optimization
    §3.5 Sharpness-Aware Optimization
§4 Experiments                                      [REWRITTEN]
    §4.1 Experimental Setting
    §4.2 Main result: HyperDG is statistically indistinguishable from ERM, and worse on OH   [NEW]
    §4.3 Source-of-gains decomposition (LOO)        [NEW]
    §4.4 Variance regularizer suppresses a spread that doesn't exist   [NEW]
    §4.5 Sensitivity to lambda_var                  [NEW]
    §4.6 Note on prior single-seed ablation numbers [NEW]
    §4.7 Curvature response diagnostics (descriptive only)   [REFRAMED]
§5 Discussion                                       [NEW]
§6 Limitations                                      [NEW]
§7 Conclusion                                       [REWRITTEN]
Appendix
    Theoretical Analysis and Intuition (Motivational)  [PREFACED with reader's note]
    Riemannian Generalization Bound                 [unchanged math, framing softened]
    ... (rest of appendix unchanged)
```

## Headline numbers in the paper now

From `aggregate_batch4.py`, multi-seed (n = 16 per cell):

| Condition | PACS | VLCS | OfficeHome | Terra |
|---|---:|---:|---:|---:|
| ERM (all losses off) | 84.04 ± 7.10 | 75.63 ± 13.32 | **67.05 ± 10.26** | 43.90 ± 8.49 |
| Full HyperDG | 84.50 ± 7.45 | 75.47 ± 14.14 | 66.03 ± 9.98 | 43.17 ± 4.97 |
| Curvature frozen | 84.67 ± 7.42 | 76.27 ± 13.17 | 66.13 ± 9.78 | 44.30 ± 6.74 |
| No variance reg | 84.57 ± 6.92 | 75.04 ± 14.25 | 66.03 ± 9.75 | 43.90 ± 6.94 |

Paired t-test p-values vs ERM:

| | PACS | VLCS | OH | Terra |
|---|---:|---:|---:|---:|
| Full vs ERM | 0.534 | 0.713 | **0.013** | 0.631 |
| Frozen vs ERM | 0.301 | 0.237 | **0.020** | 0.784 |
| No var. vs ERM | 0.422 | 0.348 | **0.017** | 0.021 |

## Sanity check (no LaTeX toolchain on this box)

All environments balanced, brace count balanced, all `\includegraphics`
targets present (one commented-out reference to `curvature_training_curve.pdf`
is harmless). To compile locally:

```bash
cd /data/sunny/HyperDG/HyperDG_TMLR
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary
```

## To restore the originals

```bash
cd /data/sunny/HyperDG/HyperDG_TMLR
cp main.tex.backup_<timestamp>          main.tex
cp supplementary.tex.backup_<timestamp> supplementary.tex
```

## Files in `/data/sunny/HyperDG/docs/` you should also read

- `MASTER_INDEX.md` — single source of truth for every run, file, and result
- `rewrite_draft.md` — narrative draft (markdown form) backing the new paper text
- `results_tables.md` — all tables in markdown for paste-into-LaTeX
