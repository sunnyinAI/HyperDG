# HyperDG Manuscript Rewrite Notes

The current revision replaces the original positive-method narrative with an implementation-faithful controlled re-evaluation.

## Major Changes

- Retitled the paper as *Do Domain-Specific Curvature Scales Improve Domain Generalization? A Controlled Re-evaluation of HyperDG*.
- Rewrote the abstract, introduction, related work, method, experiments, discussion, limitations, broader impact, and conclusion.
- Replaced the intended Lorentz/SAM description with the operations executed by `src/sftdg/algorithms/hyperdg.py`.
- Renamed the primary baseline as a matched zero-penalty control rather than standard ERM.
- Removed the unsupported theoretical appendix and all claims of provable improvement.
- Removed non-comparable literature tables from the controlled result section.
- Corrected the paired t-test implementation using SciPy and added 95% confidence intervals and Holm correction over 12 comparisons.
- Corrected TerraIncognita/no-variance from the erroneous `p=0.0206` to `p=0.99985`.
- Rebuilt the supplement around explicit evidence status, completion-filtered secondary tables, all 20 post-hoc Poincare plots, runtime, and incomplete-experiment disclosures.
- Added explicit comparison with Fan et al. (2025), Bi et al. (2025), and MERU.
- Added limitations and broader-impact sections.
- Reduced the main paper to at most 10 compiled pages, including references.

## Evidence Artifacts

- `outputs/batch4_per_run.csv`
- `outputs/batch4_summary.csv`
- `outputs/forest_erm_vs_hyperdg.pdf`
- `outputs/curvature_summary.csv`
- `outputs/shared_curvature_seed0.csv`
- `outputs/loo_seed0.csv`
- `outputs/proxy_ablation_officehome_seed0.csv`
- `outputs/variance_sweep_pacs_seed0.csv`

See `REVISION_AUDIT.md` for the reviewer-by-reviewer completion matrix and the remaining experiment plan.

## Compile

```bash
cd HyperDG_TMLR
tectonic main.tex
tectonic supplementary.tex
```
