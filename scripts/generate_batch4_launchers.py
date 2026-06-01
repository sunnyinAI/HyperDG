#!/usr/bin/env python3
"""Generate three bash launchers that evenly split 192 Batch-4 configs
across GPU 0, 1, 3 (Ada cards).

Interleaved assignment by config index → load is naturally balanced across
datasets and conditions, so no single GPU gets stuck with all the slow
runs (e.g. all OfficeHome).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPUS = [0, 1, 3]

TEMPLATE_HEADER = """#!/usr/bin/env bash
# ============================================================
# GPU {gpu} — Batch-4 multi-seed (auto-generated)
#   {n_runs} runs (every {stride}-th config starting at index {offset}).
#   Each run: ~25 min  →  total ~{total_min} min  ~{total_h:.1f} h.
# ============================================================
set -uo pipefail

PYBIN="${{HOME}}/.conda/envs/sftdg/bin/python3"
ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
export PYTHONPATH="${{ROOT_DIR}}/src:${{ROOT_DIR}}/third_party/domainbed:${{PYTHONPATH:-}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES={gpu}

LOG_ROOT="${{ROOT_DIR}}/logs/batch4/gpu{gpu}"
mkdir -p "${{LOG_ROOT}}"

run_exp() {{
  local CONFIG="$1"
  local TAG="$2"
  local OUTPUT_DIR="$3"
  local LOG="${{LOG_ROOT}}/${{TAG}}.log"
  if [[ -f "${{ROOT_DIR}}/${{OUTPUT_DIR}}/model_last.pt" ]]; then
    echo "[GPU{gpu}] $(date '+%H:%M:%S')  SKIP (done) ${{TAG}}"
    return
  fi
  echo "[GPU{gpu}] $(date '+%H:%M:%S')  START  ${{TAG}}"
  "${{PYBIN}}" "${{ROOT_DIR}}/scripts/train.py" --config "${{CONFIG}}" --device cuda \\
      2>&1 | tee "${{LOG}}"
  echo "[GPU{gpu}] $(date '+%H:%M:%S')  DONE   ${{TAG}}"
}}

"""

TEMPLATE_FOOTER = """
echo "[GPU{gpu}] All assigned runs finished at $(date)"
"""


def main() -> None:
    configs = sorted(ROOT.glob("configs/batch4/seed*/*/*.yaml"))
    assert len(configs) == 192, f"expected 192 configs, found {len(configs)}"

    # Round-robin assignment for natural load balance.
    assignment: dict[int, list[Path]] = {g: [] for g in GPUS}
    for i, cfg in enumerate(configs):
        gpu = GPUS[i % len(GPUS)]
        assignment[gpu].append(cfg)

    for idx, gpu in enumerate(GPUS):
        runs = assignment[gpu]
        n = len(runs)
        total_min = n * 25
        body_lines = []
        for cfg in runs:
            rel = cfg.relative_to(ROOT)
            stem = cfg.stem                        # e.g. pacs_split0
            # output_dir lives at experiments/batch4/seed{s}/{cond}/{stem}
            parts = rel.parts                      # configs/batch4/seedS/cond/<stem>.yaml
            seed_dir = parts[2]                    # seedX
            cond     = parts[3]                    # cond
            out_dir  = f"experiments/batch4/{seed_dir}/{cond}/{stem}"
            tag      = f"{seed_dir}_{cond}_{stem}"
            body_lines.append(
                f'run_exp "${{ROOT_DIR}}/{rel}" \\\n'
                f'        "{tag}" \\\n'
                f'        "{out_dir}"'
            )

        script_path = ROOT / f"scripts/run_batch4_gpu{gpu}.sh"
        with script_path.open("w") as f:
            f.write(TEMPLATE_HEADER.format(
                gpu=gpu, n_runs=n, stride=len(GPUS), offset=idx,
                total_min=total_min, total_h=total_min/60,
            ))
            f.write("\n".join(body_lines))
            f.write(TEMPLATE_FOOTER.format(gpu=gpu))
        script_path.chmod(0o755)
        print(f"Wrote {script_path.name}: {n} runs (~{total_min/60:.1f}h)")

    # Verify no overlap and full coverage
    all_assigned = set()
    for g, runs in assignment.items():
        for r in runs:
            assert r not in all_assigned, f"overlap: {r}"
            all_assigned.add(r)
    assert all_assigned == set(configs), "missing some configs"
    print(f"\nCoverage OK: {len(all_assigned)}/{len(configs)} configs, no overlap.")


if __name__ == "__main__":
    main()
