# Implementation Roadmap

This document breaks the reproduction effort into actionable phases. Use it as a living checklist while implementing the Self-Feedback Training (SFT) algorithm and benchmarking on VLCS + PACS.

## Phase 1 — Data & Baselines

- [ ] Finish `scripts/download_datasets.py` verification on the target machine (`python scripts/download_datasets.py --root /data/sunny/datasets --datasets VLCS PACS`).
- [ ] Validate the training harness with ERM (`python scripts/train.py --config configs/training/resnet50_sft_vlcs.yaml --algo ERM`) to guarantee the dataloaders, logging, and evaluation steps work end‑to‑end.
- [ ] Add basic unit smoke test with a tiny synthetic dataset (optional `pytest` target) to guard against regressions.

## Phase 2 — Model Components

- [ ] Implement the Refiner network (`src/sftdg/models/refiner.py`): a light MLP that accepts flattened features or logits and outputs soft labels; expose hyperparameters for width, depth, dropout.
- [ ] Implement SAM with soft labels:
  - Compute `L_CE^{-SL}` using refiner outputs (`Equation 7`).
  - Derive perturbation `ϵ̄` using gradient normalization (`Equation 8`).
  - Evaluate sharpness `L_CE^{S-SL}` (`Equation 9`).
- [ ] Construct feedback signals (`|L_CE^{S-SL}(Dd) - L_CE^{S-SL}(Dd')|`) by sampling two domains per Algorithm 1.
- [ ] Implement Algorithm 2 (KL divergence projection onto constrained label simplex) in `src/sftdg/utils/label_projection.py`.

## Phase 3 — Training Loop Integration

- [ ] Extend `SFTAlgorithm.update` to:
  1. Sample two source domains, run SAM update on the “active” domain.
  2. Compute feedback signal using held-out domain pair.
  3. Accumulate losses for the refiner (`L_P^{CE}` + consistency penalties) and update `ϕ`.
  4. Track diagnostics (`sharpness_gap`, `refiner_loss`, `sam_loss`, `kl_loss`).
- [ ] Add exponential moving averages for sharpness metrics to stabilize logging.
- [ ] Implement gradient clipping / mixed precision (if necessary for stability).

## Phase 4 — Evaluation & Sweeps

- [ ] Implement leave-one-domain-out orchestration:
  - Run `scripts/train.py` four times per dataset with `test_envs` cycling over domain indices.
  - Save metrics to `experiments/<dataset>/<split>/metrics.jsonl`.
  - Aggregate domain accuracies via `scripts/evaluate.py`.
- [ ] Automate average ± std computation (per paper) in a new helper `scripts/aggregate_results.py`.
- [ ] Compare against reported numbers (VLCS ≈ `79.8 ± 0.1`, PACS ≈ `88.3 ± 0.3`).

## Phase 5 — Documentation & Release

- [ ] Update `README.md` with finalized run commands, expected runtimes, and hardware.
- [ ] Provide a `results/` table summarizing reproduction accuracy vs. paper.
- [ ] Package experiment configs (`.yaml`) for each test environment.
- [ ] Optional: integrate Weights & Biases or TensorBoard config for tracking.

### Risk Mitigation

- Schedule gradient-norm monitoring to detect divergence during Phase 3.
- Cache dataset integrity checksums after Phase 1 to avoid silent corruption.
- Run ablations (ERM, SAM) to ensure implementation is competitive before SFT tuning.

Keep this roadmap up-to-date as progress is made; checking off items will make final reporting straightforward.
