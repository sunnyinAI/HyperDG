# HyperDG

**Sunny Gupta and Amit Sethi** · Indian Institute of Technology Bombay

**Historical manuscript:** *HyperDG: Hyperbolic Representation Alignment for Robust Domain Generalization via Curvature Refinement*

- [Author-attributed repository PDF](paper/7459/main.pdf)
- [Original supplementary PDF](paper/7459/supplementary.pdf)
- [Original OpenReview submission #7459](https://openreview.net/forum?id=TSshrjqnXu)
- [Accepted follow-up and its six-condition experiment artifact](https://github.com/sunnyinAI/HyperDG-Plus)
- [Reproduction instructions](reproduction/README.md)

The repository PDF restores the authorship shown in the original supplement. The manuscript is a historical submission. The archived runs and code document the available implementation, which rescales Euclidean features. The original manuscript describes a Lorentzian method and contains claims that the available code and archived results do not establish. See the follow-up for the audited formulation and findings.

## Intermediate controlled re-evaluation

The files under `HyperDG_TMLR/` and the four-condition results below are an intermediate re-evaluation titled *Do Domain-Specific Curvature Scales Improve Domain Generalization? A Controlled Re-evaluation of HyperDG*.

## Main Finding

Across PACS, VLCS, OfficeHome, and TerraIncognita, the full objective was compared with a matched zero-penalty control using four seeds and all four leave-one-domain-out splits (`n=16` matched runs per cell). No comparison remained significant after Holm correction over the 12 planned variant-by-dataset tests.

| Dataset | Control | Full objective | Difference | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| PACS | 84.04 +/- 7.10 | 84.50 +/- 7.45 | +0.45 pp | 0.534 | 1.000 |
| VLCS | 75.63 +/- 13.32 | 75.47 +/- 14.14 | -0.15 pp | 0.713 | 1.000 |
| OfficeHome | 67.05 +/- 10.26 | 66.03 +/- 9.98 | -1.02 pp | 0.013 | 0.153 |
| TerraIncognita | 43.90 +/- 8.49 | 43.17 +/- 4.97 | -0.73 pp | 0.631 | 1.000 |

The control shares the HyperDG code path with all auxiliary coefficients and curvature learning disabled. It is an ERM-like matched control, not a separately implemented standard DomainBed ERM.

## What the Code Implements

For source domain `i`, the released algorithm learns a positive scalar

```text
c_i = softplus(a_i) + epsilon
```

and rescales encoder features as

```text
z_i = h_theta(x) / sqrt(c_i).
```

The training objective combines:

- supervised cross-entropy on the scaled features;
- alignment of source-domain minibatch means;
- pairwise curvature-scale dispersion;
- a curvature-weighted feature-norm variance proxy;
- the magnitude of the supervised-loss derivative with respect to curvature;
- disagreement between domain-specific and global feature scales.

The primary `hyperdg` implementation does **not** use Lorentz log/exp maps, geodesic classification, prototype transport, or a SAM inner maximization. At evaluation, it predicts from the unscaled encoder features because an unseen target has no source-domain curvature index. The paper and supplementary material use terminology that reflects these executed computations.

## Controlled Experiment Inventory

All Batch 1-4 controlled configurations completed:

| Batch | Runs | Purpose |
|---|---:|---|
| Batch 1 | 44 | Frozen curvature, no variance, PACS variance-weight sweep |
| Batch 2 | 40 | Zero-penalty control and component removal |
| Batch 3 | 16 | Seed-0 full objective |
| Batch 4 | 192 | Seeds 1-3 for all four primary conditions |
| **Total** | **292** | 256 primary + 36 diagnostic runs |

The direct shared-curvature study is only seed 0. PACS, VLCS, and OfficeHome have four complete splits; TerraIncognita has three because split 1 stopped at step 2600. It should not be described as a completed multi-seed equivalence test.

## Reproduce the Tables and Figures

```bash
# From a fresh clone, stage the recorded metrics in the ignored output folder.
cp -R reproduction/recorded/experiments .
python -m pip install -r reproduction/requirements-analysis.txt

python scripts/aggregate_batch4.py
python scripts/plot_forest_erm_vs_hyperdg.py
python scripts/plot_curvature_trajectories.py
```

Generated and previously archived evidence is under `outputs/`. The secondary aggregation script requires additional completed checkpoints.

- `batch4_per_run.csv`: all 256 primary accuracy observations;
- `batch4_summary.csv`: means, paired effects, confidence intervals, raw p-values, and Holm-adjusted p-values;
- `forest_erm_vs_hyperdg.pdf`: paired-effect forest plot;
- `curvature_trajectories*.pdf` and `curvature_summary.csv`: curvature dynamics;
- `shared_curvature_seed0.csv`: completion-filtered shared-curvature diagnostic;
- `loo_seed0.csv`: leave-one-component-out results;
- `proxy_ablation_officehome_seed0.csv`: legacy sharpness/flatness proxy decomposition;
- `variance_sweep_pacs_seed0.csv`: one-factor variance-weight sweep.

## Train a Configuration

```bash
PYTHONPATH=src:third_party/domainbed python scripts/train.py \
  --config configs/training/resnet50_hyperdg_pacs.yaml \
  --device cuda
```

Primary controlled configurations are under:

```text
configs/batch1/
configs/batch2/
configs/batch3/
configs/batch4/
```

## Runtime Benchmark

A genuine DomainBed ERM timing configuration is provided at:

```text
configs/training/resnet50_erm_pacs.yaml
```

Example:

```bash
PYTHONPATH=src:third_party/domainbed python scripts/measure_training_time.py \
  --configs \
    configs/training/resnet50_erm_pacs.yaml \
    configs/training/resnet50_hyperdg_pacs.yaml \
    configs/training/resnet50_sft_pacs.yaml \
  --labels ERM HyperDG SFT \
  --baseline-label ERM \
  --warmup 20 --steps 200 --device cuda
```

Across five order-rotated repeats, ERM measured `456.16 +/- 26.89` ms/step,
HyperDG `464.27 +/- 8.04` ms/step, and SFT `583.67 +/- 4.39` ms/step. HyperDG's
within-repeat relative cost was `1.020 +/- 0.060x` versus ERM. The previous
timing table labeled a `flatland` curvature configuration as ERM and is
superseded by this repeated genuine-ERM benchmark.

## Manuscript

- Main source: `HyperDG_TMLR/main.tex`
- Main PDF: `HyperDG_TMLR/main.pdf`
- Supplement source: `HyperDG_TMLR/supplementary.tex`
- Supplement PDF: `HyperDG_TMLR/supplementary.pdf`

Compile with Tectonic 0.15 or later:

```bash
cd HyperDG_TMLR
tectonic main.tex
tectonic supplementary.tex
```

## Scope and Remaining Work

The following artifacts exist but are not confirmatory evidence:

- MERU runs are incomplete, the contrastive mask mixes intra- and inter-domain positives, and the current MERU objective adds the global-consistency term twice.
- Camelyon17 has five HyperDG-only custom leave-one-hospital-out runs, with no matched baseline and no official WILDS protocol result.
- The available `lambda_align`/legacy proxy sweep changes several coefficients simultaneously and does not establish one-factor cross-dataset sensitivity.
- Poincare figures are post-hoc PCA/fixed-curvature projections, not proof of end-to-end hyperbolic training.

A stronger follow-up requires a standard ERM accuracy baseline, a true SAM baseline if SAM is claimed, multi-seed shared-curvature runs, controlled MERU inter/intra contrastive experiments, and a matched official WILDS evaluation.
