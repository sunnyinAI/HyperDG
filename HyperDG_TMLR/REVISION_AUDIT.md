# HyperDG Negative-Result Revision Audit

## Submission Decision

The controlled negative-result core is complete, but the paper should not claim that every original reviewer request has been experimentally solved. The revised manuscript is written to match the current evidence and explicitly scopes incomplete analyses.

## Implemented and Integrated

| Item | Repository evidence | Manuscript status |
|---|---|---|
| Four-condition controlled grid | 256/256 runs complete | Main result |
| Component and variance diagnostics | 36/36 runs complete | Exploratory supplement |
| Correct paired statistics | SciPy paired tests, 95% CIs, Holm correction | Main + supplement |
| Curvature trajectories | Four datasets, with and without variance penalty | Main + supplement |
| Variance-weight sweep | PACS, seed 0, four splits | Exploratory supplement |
| OfficeHome proxy decomposition | 16/16 runs complete | Relabeled accurately; supplement |
| Poincare artifacts | 20 PDFs available | All shown with post-hoc caveat |
| Prior-work discussion | Fan et al., Bi et al., MERU | Related Work |
| Limitations and broader impact | Added | Main paper |
| Page limit | Main PDF reduced below 12 pages | Resolved |
| Statistical Terra/no-variance error | Corrected from p=0.0206 to p=0.99985 | Resolved |
| Runtime methodology | Genuine DomainBed ERM config and repeated benchmark | Supplement |

## Partially Implemented

| Item | What exists | What is missing |
|---|---|---|
| Single shared curvature | Seed 0; 4 splits on PACS/VLCS/OfficeHome, 3 complete Terra splits | Multi-seed controlled condition; Terra split 1 rerun |
| DomainNet | Legacy and reviewer runs exist | Matched step budget and primary controlled grid |
| MERU | 44 metric files and 40 checkpoints | Four completed reruns, controlled accuracy table, multiple seeds |
| Camelyon17 | Five HyperDG-only leave-one-hospital-out runs | ERM/SAM controls, multiple seeds, official WILDS protocol |
| Hyperparameter analysis | Valid PACS variance-weight sweep | One-factor lambda_align and proxy/feedback sweeps across datasets and seeds |
| Public artifacts | All configs and aggregate primary CSV are tracked | Raw per-run metrics archive or permanent release deposit |

## Not Implemented

1. A standard DomainBed ERM **accuracy** baseline under the primary four-seed/four-split protocol. The current primary control is HyperDG with all auxiliary weights and curvature learning disabled.
2. A true SAM condition. The legacy `gamma_sharp`, `delta_local_flat`, and `delta_global_flat` terms are proxies; the code has no adversarial SAM inner step.
3. Explicit inter-domain versus intra-domain MERU contrastive modes. The current positive mask loses domain identity and mixes both.
4. A corrected MERU objective: `delta_global * global_flat_loss` is currently added twice.
5. A multi-seed shared-curvature study sufficient to assess equivalence with per-domain curvature.
6. A valid cross-dataset one-factor sensitivity study for `lambda_align` and the feedback/proxy weights.
7. An official WILDS comparison with a matched baseline.

## Reviewer-by-Reviewer Status

| Concern | Status after rewrite |
|---|---|
| Action editor: correct positive claims | Addressed in writing |
| Action editor: per-domain versus shared curvature | Partial; seed-0 diagnostic only |
| Action editor: no variance regularization | Addressed with multi-seed accuracy and trajectories |
| hQnG: Poincare visualizations | Addressed qualitatively; all 20 included |
| hQnG: MERU backbone | Partial and excluded from conclusions |
| hQnG: inter versus intra contrastive learning | Not addressed experimentally |
| qLsj: computational cost | Addressed with corrected repeated timing |
| qLsj: Fan/Bi comparison | Addressed in Related Work |
| szsk: curvature evolution | Addressed |
| szsk: SAM ablation | Original premise corrected; true SAM remains missing |
| szsk: hyperparameter sensitivity | Partial; only variance weight is valid |
| szsk: WILDS/new OOD benchmark | Partial; HyperDG-only custom Camelyon17 is insufficient |
| szsk: broader impact | Addressed |

## Minimum Experiments Before a Strong Resubmission

### Priority 0

1. Run standard DomainBed ERM accuracy on the same four datasets, four seeds, and all splits: 64 runs.
2. Complete shared-curvature coverage. If the existing seed-0 runs remain configuration-compatible, run 48 seed 1-3 configurations plus the incomplete Terra seed-0 split: 49 runs. Otherwise rerun all 64.
3. Decide whether SAM remains part of the paper's scope. If yes, implement and run true SAM under the same protocol. If no, explicitly state in the response that the earlier SAM terminology was incorrect, as the revised paper now does.
4. Deposit primary raw metrics in a stable public artifact and link the exact commit/release.

### Priority 1

1. Add a controlled DomainNet subset or explain its exclusion without reusing unmatched legacy values.
2. Remove the duplicated MERU global-consistency term, implement `contrast_mode={inter,intra,all}`, finish the four incomplete runs, and report a controlled table.
3. Run one-factor `lambda_align` and proxy-weight sweeps on at least PACS and OfficeHome with multiple seeds.
4. Evaluate HyperDG and ERM with the official Camelyon17/WILDS protocol.

## Claims That Must Not Reappear

- “HyperDG significantly outperforms ERM/SFT.”
- “Per-domain curvature improves generalization.”
- “The variance regularizer validates curvature consensus.”
- “The primary HyperDG implementation performs full Lorentz log/exp mapping.”
- “The norm-dispersion proxy is SAM.”
- “The MERU contrastive loss is inter-domain only.”
- “The current Camelyon17 result establishes WILDS robustness.”
- “Failure to reject proves equivalence.”
