# Reproducibility guide

The repository separates three tasks that require different inputs and software. Run all commands from the repository root with Python 3.12.

## 1. Rebuild the released evidence

This path uses the frozen, machine-readable inputs included in Git. It rebuilds the manuscript figure data, nine figures, tables and the post hoc matching analyses, then compares the deterministic outputs with the released files.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/reproduce.py release
```

This is the automated path run in continuous integration. It does not refit the clustering methods.

## 2. Verify original data and run a raw-data smoke test

Place inputs at `DATA_ROOT/<dataset>/<filename>`. Exact filenames, byte sizes and SHA-256 hashes are recorded in `data/input_manifest.csv`.

```bash
python tools/reproduce.py verify-inputs --data-root DATA_ROOT
python -m pip install -r requirements-smoke.txt
python tools/reproduce.py smoke \
  --data-root DATA_ROOT \
  --dataset Levine_13dim \
  --seeds 1 \
  --output reproduction_runs/levine13_smoke
```

The smoke test verifies the input and runs the paired all-event versus labeled-only K-means analysis from the raw matrix through saved predictions and metrics. For Levine_13dim, it also compares the seed-0 ARI, macro precision, macro recall, macro F1 and iteration count with the accepted run in `experiments/expected/levine13_kmeans_seed0.csv`. With `--seeds 30`, the command additionally checks the means, standard deviations, paired differences and better-result fractions against the released `analysis/inputs/kmeans_event_inclusion.csv` values.

## 3. Refit the named clustering implementations

The scripts under `experiments/src` are the retained result-generation and evaluation code. Their method-specific environments are recorded under `experiments/environments`. FlowSOM, PhenoGraph, Vortex X-shift and Deterministic-SPADE remain external implementations and are not vendored here. Several finalizers also require the accepted parent artifacts named by their command-line arguments.

This repository therefore does not claim that every method can be refitted from raw data with one command. In addition, two early X-shift evaluations were produced by historical revisions of `evaluate_xshift_cross_dataset.py` that are no longer available. Their accepted summaries are frozen under `analysis/inputs`; the current evaluator covers the later path. These boundaries are stated explicitly so that rebuilding manuscript evidence is not confused with recreating every historical computation.

## Individual commands

```bash
python tools/reproduce.py verify-release
python tools/reproduce.py paper
python tools/reproduce.py posthoc --output posthoc/output
python tools/reproduce.py verify-inputs --data-root DATA_ROOT --datasets Levine_13dim
```

All commands return a nonzero exit status when an identity check or output comparison fails.
