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

Quick download helper (currently VLCS/PACS):

```bash
python scripts/download_datasets.py --root data --datasets VLCS PACS
```

For OfficeHome, TerraIncognita, and DomainNet, place datasets in the paths specified by:
- `configs/datasets/officehome.yaml`
- `configs/datasets/terraincognita.yaml`
- `configs/datasets/domainnet.yaml`

## Training

### ResNet-50 HyperDG (single run)

```bash
python scripts/train.py --config configs/training/resnet50_hyperdg_vlcs.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_pacs.yaml --device cuda
python scripts/train.py --config configs/training/resnet50_hyperdg_officehome.yaml --device cuda
```

### Split sweeps (all domains)

Pre-generated split configs are in `experiments/configs/`.

```bash
./run_all_hyperdg.sh vlcs 4
./run_all_hyperdg.sh pacs 4
./run_all_hyperdg.sh officehome 4
./run_all_hyperdg.sh domainnet 6
```

### ViT HyperDG sweeps

```bash
python scripts/create_vit_configs.py --backbones vitb16 vitl14 --datasets vlcs pacs officehome terraincognita domainnet
./scripts/run_vit_sweeps.sh vitb16 officehome 4
./scripts/run_vit_sweeps.sh vitl14 pacs 4
```

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
