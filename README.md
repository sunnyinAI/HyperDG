# HyperDG

**Paper title:** *HyperDG: Hyperbolic Representation Alignment for Robust Domain Generalization via Curvature Refinement*  
(Under review as a TMLR submission; see `docs/HyperDG_TMLR.pdf`)

This repository contains the PyTorch + DomainBed implementation of **HyperDG**.

## Method Summary

HyperDG optimizes a supervised objective with geometry-aware regularization:

\[
\mathcal{L} = \mathcal{L}_{sup}
+ \alpha\,\mathcal{L}_{curv}
+ \beta\,\mathcal{L}_{align}
+ \gamma\,\mathcal{L}_{sharp}
+ \delta_l\,\mathcal{L}_{local}
+ \delta_g\,\mathcal{L}_{global}
\]

Implemented in `src/sftdg/algorithms/hyperdg.py`:
- Learnable per-domain and global curvature.
- Tangent-space feature alignment across source domains.
- Sharpness/local/global flatness regularization.
- Unified optimization of backbone/classifier and curvature parameters.

## Repository Layout

```text
.
├── configs/             # Dataset and training YAMLs
├── docs/                # Paper PDF and notes
├── experiments/         # Outputs/checkpoints/config copies (mostly gitignored)
├── figures/             # Figure scripts/assets
├── scripts/             # Train/eval/aggregation/ablation utilities
├── src/sftdg/           # Algorithms, models, utils
├── environment.yml      # Conda environment
└── README.md
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate sftdg
python -m pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/domainbed
```

## Dataset Setup

DomainBed-style data roots are expected via dataset config files in `configs/datasets/`.

Quick download helper (DomainBed + WILDS):

```bash
python scripts/download_datasets.py --root data --datasets VLCS PACS
python scripts/download_datasets.py --root data/wilds --datasets WILDSCamelyon
```

For OfficeHome, TerraIncognita, and DomainNet, place datasets in the paths specified by:
- `configs/datasets/officehome.yaml`
- `configs/datasets/terraincognita.yaml`
- `configs/datasets/domainnet.yaml`

WILDS dataset configs:
- `configs/datasets/wilds_camelyon.yaml`
- `configs/datasets/wilds_fmow.yaml`

## Training

### ResNet-50 HyperDG (single run)

```bash
python scripts/train.py --config configs/training/resnet50_hyperdg_vlcs.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_pacs.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_officehome.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_wilds_camelyon.yaml --device cuda
```

### Split sweeps (all domains)

Pre-generated split configs are in `experiments/configs/`.

```bash
./run_all_hyperdg.sh vlcs 4
./run_all_hyperdg.sh pacs 4
./run_all_hyperdg.sh officehome 4
./run_all_hyperdg.sh domainnet 6
```

For WILDSCamelyon (5 hospitals / 5 leave-one-hospital-out splits), split configs can be
materialized under `experiments/configs/wilds_camelyon/` and launched similarly:

```bash
for split in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((split % 4)) python scripts/train.py \
    --config experiments/configs/wilds_camelyon/hyperdg_wilds_camelyon_split${split}.yaml \
    --device cuda
done
```

### ViT HyperDG sweeps

```bash
python scripts/create_vit_configs.py --backbones vitb16 vitl14 --datasets vlcs pacs officehome terraincognita domainnet
./scripts/run_vit_sweeps.sh vitb16 officehome 4
./scripts/run_vit_sweeps.sh vitl14 pacs 4
```

**Federated**
FedAvg-style HyperDG with per-domain clients, aligned via global tangent statistics. Federated configs live in `configs/federated/` and outputs are kept under `experiments/federated/`.

```bash
python scripts/train_federated.py \
  --config configs/federated/resnet50_hyperdg_fed_vlcs.yaml --device cuda

./scripts/run_federated.sh all
```

Federated runs emit `curvature_rounds.npy`, `curvature_final.npy`, and `domain_accuracy.csv` alongside `metrics.jsonl`.

## Ablations

### Key ablations used in your Table-5 style comparison

- **w/o \(L_{fb}\)**: `gamma_sharp=0`, `delta_local_flat=0`, `delta_global_flat=0`
  - Config base: `configs/training/vitb16_hyperdg_officehome_wo_lfb.yaml`
- **w/o \(L_{var}\)**: `alpha_curv=0`
  - Config base: `configs/training/vitb16_hyperdg_officehome_wo_lvar.yaml`

Run split configs already materialized in `experiments/configs/`:

```bash
for split in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((split % 4)) python3 scripts/train.py \
    --config experiments/configs/vitb16_hyperdg_officehome_wo_lfb_split${split}.yaml \
    --device cuda
done

for split in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((split % 4)) python3 scripts/train.py \
    --config experiments/configs/vitb16_hyperdg_officehome_wo_lvar_split${split}.yaml \
    --device cuda
done
```

## Metrics Aggregation

```bash
python scripts/aggregate_metrics.py --root experiments/vitb16_hyperdg_officehome --num-envs 4 --verbose
python scripts/aggregate_metrics.py --root experiments/vitb16_hyperdg_officehome_wo_lfb --num-envs 4 --verbose
python scripts/aggregate_metrics.py --root experiments/vitb16_hyperdg_officehome_wo_lvar --num-envs 4 --verbose
```

## Reviewer Response Experiments

The following scripts were added to address TMLR reviewer szsk's requested changes.

### R1 — Curvature Visualization

Training now logs **per-domain curvature** values (`curvature_domain_0`, `curvature_domain_1`, …)
at every checkpoint step in `metrics.jsonl`.

Plot per-domain K_i evolution (shows convergence and final values):

```bash
python scripts/plot_curvature_per_domain.py \
    --metrics experiments/hyperdg_pacs_curvtraj/split*/metrics.jsonl \
    --domain-names art_painting cartoon photo \
    --output outputs/curvature_per_domain_pacs.pdf
```

Combine with existing scripts to show curvature variance vs accuracy:

```bash
python scripts/plot_curvature_vs_accuracy.py \
    --metrics experiments/resnet50_hyperdg_vlcs/split0/metrics.jsonl \
               experiments/resnet50_hyperdg_pacs/split0/metrics.jsonl \
               experiments/resnet50_hyperdg_officehome/split0/metrics.jsonl \
    --labels VLCS PACS OfficeHome \
    --num-envs 4 4 4 \
    --output outputs/curvature_vs_accuracy.pdf --scatter
```

### R2 — SAM Ablation

A new ablation schedule `hyperdg_sam_ablation` isolates the contribution of
curvature alignment from the SAM-like flatness terms:

| Condition | alpha_curv | beta_align | gamma_sharp | delta_* |
|---|---|---|---|---|
| ERM baseline | 0 | 0 | 0 | 0 |
| SAM-like only | 0 | 0 | 0.05 | 0.1 |
| HyperDG w/o SAM | 0.2 | 0.5 | 0 | 0 |
| HyperDG full | 0.2 | 0.5 | 0.05 | 0.1 |

```bash
python scripts/run_ablation_grid.py \
    --base-config configs/training/resnet50_hyperdg_officehome.yaml \
    --schedule hyperdg_sam_ablation \
    --gpus 0,1,2,3

python scripts/summarize_ablation.py \
    --runs-root experiments/ablations/hyperdg_sam_ablation \
    --env-names Art Clipart Product RealWorld
```

Completed run set used in rebuttal: `experiments/ablations/hyperdg_sam_ablation_officehome_v6`

| Condition | split0 | split1 | split2 | split3 | Mean ± Std |
|---|---:|---:|---:|---:|---:|
| ERM baseline | 63.71 | 54.87 | 75.20 | 76.58 | **67.59 ± 8.88** |
| SAM-like only | 59.79 | 49.94 | 75.54 | 79.10 | 66.09 ± 11.82 |
| HyperDG w/o SAM | 62.89 | 50.29 | 75.87 | 76.23 | 66.32 ± 10.71 |
| HyperDG full | 59.79 | 50.86 | 74.52 | 77.73 | 65.73 ± 10.93 |

### R3 — Computational Cost

Benchmark wall-clock training time per step vs baselines:

```bash
python scripts/measure_training_time.py \
    --configs \
        configs/training/resnet50_flatland_pacs.yaml \
        configs/training/resnet50_sft_pacs.yaml \
        configs/training/resnet50_hyperdg_pacs.yaml \
    --labels ERM SFT HyperDG \
    --warmup 20 --steps 200 --device cuda \
    --output outputs/timing_erm_baseline.csv
```

Measured runtime (200 steps, PACS):

| Method | ms/step | Relative |
|---|---:|---:|
| ERM | 464.81 | 1.00x |
| SFT | 581.40 | 1.25x |
| HyperDG | 458.69 | 0.99x |

### R4 — Hyperparameter Sensitivity

Run the penalty sweep then plot sensitivity:

```bash
python scripts/run_ablation_grid.py \
    --base-config configs/training/resnet50_hyperdg_officehome.yaml \
    --schedule hyperdg_penalty_sweep \
    --gpus 0,1,2,3

python scripts/plot_hparam_sensitivity.py \
    --sweep-root experiments/ablations/hyperdg_penalty_sweep \
    --num-envs 4 \
    --dataset-label OfficeHome \
    --output outputs/hparam_sensitivity.pdf
```

Or supply a hand-crafted CSV (columns: `hparam,value,dataset,accuracy`):

```bash
python scripts/plot_hparam_sensitivity.py \
    --csv sensitivity_data.csv \
    --output outputs/hparam_sensitivity.pdf
```

Observed OfficeHome stability band (from `outputs/hparam_sensitivity.pdf`):
- `λ_var` (`α_curv`): 77.28–77.72%
- `λ_align` (`β_align`): 77.31–77.72%
- `λ_fb` (`γ_sharp`): 77.40–77.50%

### R5 — WILDS / Additional OOD Benchmarks

WILDS adapters are available via DomainBed dataset classes:
- `WILDSCamelyon`
- `WILDSFMoW`

Example run (Camelyon17, split-0):

```bash
python scripts/train.py \
    --config experiments/configs/wilds_camelyon/hyperdg_wilds_camelyon_split0.yaml \
    --device cuda
```

Completed WILDS Camelyon HyperDG splits (`experiments/hyperdg_wilds_camelyon/split0..4`):

| Split | Held-out out_acc |
|---|---:|
| split0 | 97.03% |
| split1 | 93.34% |
| split2 | 95.67% |
| split3 | 93.43% |
| split4 | 63.86% |
| Mean ± Std | **88.67% ± 12.48** |

## Notes

- The package path remains `sftdg` for compatibility with existing code.
- `algorithm: hyperdg` is the active method entry in training configs.
- If you regenerate configs, keep naming consistent with `hyperdg_*` for scripts that expect those paths.

## Citation

```bibtex
@article{hyperdg2026,
  title   = {HyperDG: Hyperbolic Representation Alignment for Robust Domain Generalization via Curvature Refinement},
  author  = {Anonymous},
  journal = {TMLR (under review)},
  year    = {2026}
}
```
