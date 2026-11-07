# Reproduce the archived HyperDG runs

This archive accompanies the original HyperDG manuscript by Sunny Gupta and Amit Sethi and the intermediate four-condition re-evaluation. The accepted six-condition study has its own experiment artifact in [HyperDG-Plus](https://github.com/sunnyinAI/HyperDG-Plus).

## Recompute the intermediate results without a GPU

From a fresh clone, with Python 3.10–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements-analysis.txt
cp -R reproduction/recorded/experiments .
python scripts/aggregate_batch4.py
python scripts/plot_forest_erm_vs_hyperdg.py
python scripts/plot_curvature_trajectories.py
```

The recorded metrics include all 292 runs of the intermediate study. The main analysis uses 256 runs in four conditions, across four datasets, four held-out domains, and four seeds. It aggregates final-step target results. See [the 256 per-run observations](../outputs/batch4_per_run.csv) and [the grouped results](../outputs/batch4_summary.csv). The additional exploratory scripts may require checkpoints or logs that are absent from the release.

The archive also contains recovered historical split configs and available metric logs for the ResNet-50 and ViT experiments. For example, the following displays the historical OfficeHome ViT-B/16 ablation aggregates using the original reporting utility:

```bash
for run in vitb16_hyperdg_officehome vitb16_hyperdg_officehome_wo_lfb vitb16_hyperdg_officehome_wo_lvar; do
  python scripts/aggregate_metrics.py --root "experiments/$run" --num-envs 4 --verbose
done
```

The historical archive is partial. The OfficeHome metrics above are available for inspection, but do not regenerate the numbers in Table 5. The archive does not provide a run-to-table mapping for every result in submission #7459, and some runs, third-party baseline results, checkpoints, and runtime measurement inputs are absent. The repository's `hyperdg` implementation scales Euclidean features during training and uses unscaled features during prediction. The archived evidence cannot reproduce the original submission's Lorentzian implementation claims.

## Train a run on a Linux CUDA machine

Use a fresh checkout for training to keep new logs separate from the analysis replay. The recorded DomainBed revision is a Git submodule.

```bash
git submodule update --init
conda env create -f environment.yml
conda activate sftdg
export PYTHONPATH="$PWD/src:$PWD/third_party/domainbed"
python scripts/train.py --help
```

Provision the datasets separately under the roots in [the dataset configs](../configs/datasets/). DomainBed expects `data/PACS`, `data/VLCS`, `data/terra_incognita`, `data/domain_net`, and `data/officehome_clean/office_home`. Follow the [pinned DomainBed dataset instructions](https://github.com/facebookresearch/DomainBed/tree/b93c22a1cfc3b2428398272c1a116c8de1f4139e#download-the-datasets) for access and download. The ResNet-50 and ViT configurations may download pretrained weights on first use.

Train an intermediate-study run from the tracked configuration:

```bash
python scripts/train.py --config configs/batch3/baseline/pacs_split0.yaml --device cuda
```

The historical configs preserve the older `fedcuri` algorithm name and original output paths. To run one through this repository's `hyperdg` entry point, create a working copy with the current name:

```bash
mkdir -p experiments
sed 's/^algorithm: fedcuri$/algorithm: hyperdg/' \
  reproduction/recorded/experiments/configs/hyperdg_pacs_split0.yaml \
  > experiments/hyperdg_pacs_split0.yaml
python scripts/train.py --config experiments/hyperdg_pacs_split0.yaml --device cuda
```

Datasets, pretrained weights, and training checkpoints are obtained or generated separately.

## Build the attributed historical manuscript

```bash
cd paper/7459
latexmk -pdf main.tex
```
