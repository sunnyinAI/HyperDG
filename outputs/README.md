# Output Artifact Status

## Confirmatory

- `batch4_per_run.csv`
- `batch4_summary.csv`
- `forest_erm_vs_hyperdg.pdf`
- `curvature_summary.csv`
- `curvature_trajectories*.pdf`

These files support the controlled four-condition, four-dataset study described in the revised paper.

## Exploratory

- `shared_curvature_seed0.csv`: seed-0 only; TerraIncognita contains three complete splits.
- `loo_seed0.csv`: seed-0 leave-one-component-out analysis.
- `proxy_ablation_officehome_seed0.csv`: proxy-term ablation, not true SAM.
- `variance_sweep_pacs_seed0.csv`: seed-0 one-factor variance-weight sweep.
- `camelyon17_hyperdg_only.csv`: custom HyperDG-only leave-one-hospital-out results without a matched control.
- `poincare_*.pdf`: post-hoc PCA/fixed-curvature projections.

## Superseded or Non-confirmatory

- `timing.csv` and `timing_erm_baseline.csv`: single-run tables; one used method-order normalization and the other mislabeled a `flatland` curvature configuration as ERM. Use the repeated genuine-ERM timing summary instead.
- `hparam_sensitivity.pdf`: generated from configurations that changed multiple coefficients and did not form a controlled cross-dataset one-factor sweep. It is retained for provenance but is not used in the revised paper.
- Legacy loss and visualization files predating the controlled Batch 1-4 study are not evidence for the revised claims.

The corrected repeated timing artifacts are `timing_repeated_per_run.csv` and
`timing_repeated_summary.csv`.
