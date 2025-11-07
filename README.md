# Seeking Consistent Flat Minima (SFT) for Domain Generalization

This repository scaffolds a PyTorch implementation of **SFT** from the CVPR 2025 paper *“Seeking Consistent Flat Minima for Better Domain Generalization via Refining Loss Landscapes”*. The long–term goal is to reproduce the VLCS and PACS results reported in Table&nbsp;1 of the paper using a ResNet‑50 backbone on the DomainBed protocol.

We currently provide:
- A project structure aligned with DomainBed experiments.
- Conda environment specification for PyTorch + torchvision + supporting libraries.
- Configuration templates for VLCS and PACS.
- Training script skeleton that will host the SFT algorithm.

## Repository layout

```
.
├── configs/             # Dataset/backbone/hyperparameter configs
├── experiments/         # Checkpoints, logs, metrics (gitignored)
├── logs/                # TensorBoard event files (gitignored)
├── scripts/             # Utility entry points (train/eval/download)
├── src/sftdg/           # Python package with algorithms, data, models, utils
├── environment.yml      # Conda environment spec
└── README.md
```

## Environment setup (Miniconda)

```bash
conda env create -f environment.yml
conda activate sftdg
python -m pip install -e .
```

The environment ships with CUDA‐enabled PyTorch 2.2/torchvision 0.17 (adjust the CUDA build in `environment.yml` if you are on a different driver version). DomainBed and other utilities install via `pip` in the post–create step.

## Dataset preparation

Experiments assume the DomainBed data layout:

```
data/
  VLCS/
    Caltech101/
    LabelMe/
    SUN09/
    VOC2007/
  PACS/
    art_painting/
    cartoon/
    photo/
    sketch/
```

Prepare datasets with:

```bash
conda activate sftdg
python scripts/download_datasets.py --root data --datasets VLCS PACS
```

The download script reuses DomainBed mirrors (Google Drive + AWS). If you already maintain a shared dataset store, pass `--root /path/to/datasets` instead and update the config files.

## Running training (placeholder)

Training and evaluation entry points live in `scripts/train.py` and `scripts/evaluate.py`. After implementing the SFT algorithm, commands will follow the template:

```bash
python scripts/train.py --config configs/training/resnet50_sft_vlcs.yaml
python scripts/train.py --config configs/training/resnet50_sft_pacs.yaml
```

Outputs (checkpoints, metrics, TensorBoard traces) default to `experiments/` and `logs/`.

### FlatLand curvature alignment

The `flatland` algorithm replaces the original SFT objective with curvature alignment in Lorentz space. Base configs are available:

```bash
python scripts/train.py --config configs/training/resnet50_flatland_vlcs.yaml
python scripts/train.py --config configs/training/resnet50_flatland_pacs.yaml
```

Split-specific configs live in `experiments/configs/flatland_{vlcs,pacs}_split*.yaml`. Use the helper to run all splits sequentially:

```bash
conda activate sftdg
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/domainbed
./scripts/run_splits.sh flatland_vlcs
./scripts/run_splits.sh flatland_pacs
```

### FedCuri (Curvature-flatness refinement)

```bash
python scripts/train.py --config configs/training/resnet50_fedcuri_vlcs.yaml
python scripts/train.py --config configs/training/resnet50_fedcuri_pacs.yaml
```

Split configs: `experiments/configs/fedcuri_{vlcs,pacs}_split*.yaml`

```bash
conda activate sftdg
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/domainbed
./scripts/run_splits.sh fedcuri_vlcs
./scripts/run_splits.sh fedcuri_pacs
```

## Reproduction checklist

| Task | Status | Notes |
| --- | --- | --- |
| Project scaffolding | ✅ | Base structure + configs + scripts |
| Dataset download utility | ⏳ | Implement mirrored download helpers |
| SFT algorithm core (feedback + refinement) | ⏳ | Requires model/refiner code & training loop |
| Validation loop + DomainBed metrics | ⏳ | |
| VLCS reproduction (ResNet‑50) | ⏳ | Target: 79.8% ± 0.1 |
| PACS reproduction (ResNet‑50) | ⏳ | Target: 88.3% ± 0.3 |

## Next steps

1. Fill in `download_datasets.py` to materialize VLCS and PACS.
2. Implement `src/sftdg/data/domainbed_datasets.py` to wrap DomainBed splits.
3. Implement `src/sftdg/models/resnet_backbone.py` to load ImageNet‑pretrained ResNet‑50 with frozen BN.
4. Implement `src/sftdg/algorithms/sft.py`, covering Algorithm&nbsp;1 & 2 (feedback/refinement) plus SAM w/ soft labels.
5. Finalize `scripts/train.py` to orchestrate config loading, seeding, logging, checkpointing, and evaluation loops.
6. Run VLCS/PACS sweeps, compute domain‑wise accuracy, and compare to Table&nbsp;1.

Track progress in the checklist above and update configs as hyperparameters converge toward reported numbers.
