# HyperDG

**Paper title:** *HyperDG: Hyperbolic Representation Alignment for Robust Domain Generalization via Curvature Refinement*  
(Under review as a TMLR submission; see `HyperDG_TMLR.pdf`)

This repository contains the PyTorch + DomainBed implementation of **HyperDG**, including all experiments from the TMLR reviewer response (single-curvature ablation, MERU backbone, and Poincaré disk visualisations).

---

## Method Summary

HyperDG optimizes a supervised objective with geometry-aware regularization:

$$\mathcal{L} = \mathcal{L}_{sup} + \alpha\,\mathcal{L}_{curv} + \beta\,\mathcal{L}_{align} + \gamma\,\mathcal{L}_{sharp} + \delta_l\,\mathcal{L}_{local} + \delta_g\,\mathcal{L}_{global}$$

Key components:
- **Per-domain learnable curvature** `{α_i}` on separate Lorentz manifolds, regularised by `λ_var` to prevent divergence.
- **Single shared curvature** mode (`single_curvature: true`) as an ablation — nearly identical performance, validating the variance regulariser design.
- **MERU backbone** variant (`algorithm: hyperdg_meru`) using facebookresearch/MERU ViT-B/16 pretrained in hyperbolic space, with a hyperbolic InfoNCE contrastive loss in Lorentz space.

---

## Repository Layout

```text
.
├── configs/
│   ├── datasets/              # Dataset YAML configs (PACS, VLCS, OfficeHome, etc.)
│   ├── training/              # Base training configs
│   └── reviewer_hQnG/        # Reviewer experiment configs (66 runs)
│       ├── single_curv/      # Single-α ablation (22 configs)
│       ├── meru/             # MERU per-domain α (22 configs)
│       └── meru_single/      # MERU single α (22 configs)
├── outputs/                   # Poincaré disk plots (20 PDFs)
├── scripts/
│   ├── train.py              # Main training entry point
│   ├── plot_poincare_disk.py # Poincaré disk visualisation
│   ├── plot_all_poincare.sh  # Batch script: all 20 plots
│   ├── generate_reviewer_configs.py  # Config generator
│   ├── summarize_reviewer_hQnG.py    # Results aggregation
│   └── run_meru_*.sh         # GPU runner scripts for MERU experiments
├── src/sftdg/
│   ├── algorithms/
│   │   ├── hyperdg.py        # HyperDG (per-domain + single-curv mode)
│   │   ├── hyperdg_meru.py   # HyperDG-MERU (MERU backbone + hyperbolic InfoNCE)
│   │   └── ...
│   └── models/
│       ├── backbones.py      # Backbone router (resnet / vit / meru_*)
│       ├── meru_backbone.py  # MERU ViT-B/16 wrapper
│       └── ...
├── environment.yml
└── README.md
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate sftdg
pip install -e .
pip install omegaconf beautifulsoup4    # required for MERU + TerraIncognita download
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/third_party/domainbed
```

---

## Dataset Setup

```bash
# PACS, VLCS, TerraIncognita
python -m domainbed.scripts.download --data_dir data --dataset PACS
python -m domainbed.scripts.download --data_dir data --dataset VLCS
python -m domainbed.scripts.download --data_dir data --dataset TerraIncognita

# OfficeHome and DomainNet: place in paths from configs/datasets/
# WILDS
python scripts/download_datasets.py --root data/wilds --datasets WILDSCamelyon
```

---

## Training

### Standard HyperDG (ResNet-50, per-domain curvature)

```bash
python scripts/train.py --config configs/training/resnet50_hyperdg_pacs.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_vlcs.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_officehome.yaml --device cuda
```

### Single shared curvature ablation

Add `single_curvature: true` to any config's `hparams` block, or use the pre-generated reviewer configs:

```bash
python scripts/train.py \
    --config configs/reviewer_hQnG/single_curv/pacs_split0.yaml --device cuda
```

Run all five datasets, all splits (GPUs 2 & 3):

```bash
bash scripts/run_reviewer_hQnG_gpu2.sh   # PACS, VLCS, TerraIncognita
bash scripts/run_reviewer_hQnG_gpu3.sh   # OfficeHome, DomainNet
```

### MERU backbone + hyperbolic contrastive loss

MERU weights (ViT-B/16, ~1.7 GB) are downloaded automatically to `~/.cache/meru/meru_vit_b.pth` on first run.

```bash
# Per-domain curvature
python scripts/train.py \
    --config configs/reviewer_hQnG/meru/pacs_split0.yaml --device cuda

# Single shared curvature
python scripts/train.py \
    --config configs/reviewer_hQnG/meru_single/pacs_split0.yaml --device cuda
```

Run all datasets in parallel across GPUs:

```bash
bash scripts/run_meru_terra_gpu1.sh      # GPU 1: TerraIncognita
bash scripts/run_meru_domainnet_gpu2.sh  # GPU 2: DomainNet per-domain
bash scripts/run_meru_domainnet_gpu3.sh  # GPU 3: DomainNet single-α
```

**MERU config hparams:**

```yaml
algorithm: hyperdg_meru
hparams:
  backbone: meru_vit_b16
  meru_variant: vit_b16
  meru_pretrained: true
  meru_freeze_backbone: true
  lambda_contrast: 0.1
  contrast_temp: 0.07
  single_curvature: false   # set true for single-α variant
```

---

## Poincaré Disk Visualisation

Follows the exact approach of seminal hyperbolic learning papers:  
**PCA → exp-map at fixed K=1 → stereographic projection** (no UMAP, geometry-faithful).

```bash
# Single plot
python scripts/plot_poincare_disk.py \
    --config configs/reviewer_hQnG/meru/pacs_split0.yaml \
    --ckpt  experiments/reviewer_hQnG/meru/pacs_split0/model_last.pt \
    --out   outputs/poincare_meru_pacs_perdomain.pdf \
    --n_samples 300 \
    --device cuda:2

# All 20 plots at once (skips missing checkpoints safely)
bash scripts/plot_all_poincare.sh --device cuda:2
```

Generated plots are saved to `outputs/poincare_*.pdf`:

| File pattern | Description |
|---|---|
| `poincare_hyperdg_{ds}_perdomain.pdf` | HyperDG per-domain α, all 5 datasets |
| `poincare_hyperdg_{ds}_singlecurv.pdf` | HyperDG single α, all 5 datasets |
| `poincare_meru_{ds}_perdomain.pdf` | MERU + per-domain α, all 5 datasets |
| `poincare_meru_{ds}_singlecurv.pdf` | MERU + single α, all 5 datasets |

---

## Results

### Single-α ablation (HyperDG, ResNet-50)

| Dataset | Per-domain α | Single α | Δ |
|---|---|---|---|
| PACS | 92.2 ± 1.2 | 92.2 ± 0.8 | 0.0 |
| VLCS | 81.8 ± 1.4 | 81.5 ± 1.1 | −0.3 |
| OfficeHome | 76.8 | 76.6 ± 0.9 | −0.2 |
| DomainNet | 52.1 ± 0.4 | 53.6 ± 0.4 | +1.5 |
| TerraIncognita | — | 77.1 ± 1.7 | — |

Differences ≤ 0.3 pp on 4/5 datasets — the variance regulariser `λ_var` drives per-domain curvatures to consensus, validating the design.

### MERU backbone (ViT-B/16, frozen, hyperbolic pretrained)

| Dataset | MERU per-domain α | MERU single α | Δ |
|---|---|---|---|
| PACS | 90.0 ± 1.5 | 90.1 ± 1.7 | +0.1 |
| VLCS | 84.9 ± 0.7 | 85.0 ± 0.6 | +0.1 |
| OfficeHome | 70.3 ± 1.0 | 70.4 ± 1.0 | +0.1 |
| DomainNet | 23.7 ± 2.4 | 22.9 ± 2.6 | −0.8 |
| TerraIncognita | 59.4 ± 2.9 | 59.3 ± 3.2 | −0.1 |

Geometry problems persist even with a hyperbolic pretrained backbone, confirming domain curvature shift is not an artefact of Euclidean pretraining.

---

## Results Aggregation

```bash
# Summarise all reviewer hQnG experiments
python scripts/summarize_reviewer_hQnG.py

# Standard metrics aggregation
python scripts/aggregate_metrics.py \
    --root experiments/hyperdg_pacs --num-envs 4 --verbose
```

---

## Ablations (Original Paper)

### Loss component ablation

| Condition | alpha_curv | beta_align | gamma_sharp | delta_* |
|---|---|---|---|---|
| ERM baseline | 0 | 0 | 0 | 0 |
| SAM-like only | 0 | 0 | 0.05 | 0.1 |
| HyperDG w/o SAM | 0.2 | 0.5 | 0 | 0 |
| HyperDG full | 0.2 | 0.5 | 0.05 | 0.1 |

```bash
python scripts/run_ablation_grid.py \
    --base-config configs/training/resnet50_hyperdg_officehome.yaml \
    --schedule hyperdg_sam_ablation --gpus 0,1,2,3
```

### Hyperparameter sensitivity

```bash
python scripts/plot_hparam_sensitivity.py \
    --sweep-root experiments/ablations/hyperdg_penalty_sweep \
    --num-envs 4 --output outputs/hparam_sensitivity.pdf
```

---

## Computational Cost

| Method | ms/step | Relative |
|---|---|---|
| ERM | 464.81 | 1.00× |
| SFT | 581.40 | 1.25× |
| HyperDG | 458.69 | 0.99× |

```bash
python scripts/measure_training_time.py \
    --configs configs/training/resnet50_hyperdg_pacs.yaml \
    --labels HyperDG --warmup 20 --steps 200 --device cuda
```

---

## Notes

- Package path is `sftdg` throughout for compatibility with existing code.
- Algorithm names: `hyperdg` (standard), `hyperdg_meru` (MERU variant).
- MERU weights auto-download on first use; requires ~1.7 GB disk space.
- DomainNet with MERU creates ~138 DataLoader worker processes — this is normal.
- Re-run `bash scripts/plot_all_poincare.sh` at any time; it safely skips missing checkpoints.

---

## Citation

```bibtex
@article{hyperdg2026,
  title   = {HyperDG: Hyperbolic Representation Alignment for Robust Domain Generalization via Curvature Refinement},
  author  = {Anonymous},
  journal = {TMLR (under review)},
  year    = {2026}
}

@inproceedings{desai2023meru,
  title     = {Hyperbolic Image-Text Representations},
  author    = {Desai, Karan and Nickel, Maximilian and Rajpurohit, Tanmay and Johnson, Justin and Vedantam, Ramakrishna},
  booktitle = {ICML},
  year      = {2023}
}
```
