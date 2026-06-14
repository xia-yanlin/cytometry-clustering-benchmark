# Cytometry Clustering Benchmark

Code for the paper:

> **Benchmarking Unsupervised Clustering in High-Dimensional Cytometry:
> Convergence Stability and Evaluation Metric Applicability**  
> *Cytometry Part A*, 2025

---

## Overview

This repository reproduces the benchmark results of ten unsupervised clustering
algorithms across five public cytometry datasets, with a focus on two
reliability boundaries that are implicit in common workflows:

1. **FlowSOM convergence stability** — statistically significant bimodal
   convergence on high-dimensional panels (≥ 32 markers), with ≈ 27–33 % of
   single runs converging to a suboptimal mode.
2. **ARI metric applicability** — ARI loses its parameter-selection gradient on
   highly imbalanced rare-population datasets; target-population F1 is the
   appropriate substitute.

---

## Requirements

Python ≥ 3.10 is required (uses built-in type-union syntax `X | Y`).

```bash
pip install -r requirements.txt
```

### BL-FlowSOM PoC (optional)

To reproduce results with the Sony official reference implementation:

```bash
git clone https://github.com/sony/bl-flowsom_poc
cd bl-flowsom_poc && pip install -r requirements.txt
```

Place the cloned folder at the project root as `bl-flowsom_poc/`.

---

## Data Preparation

All datasets are publicly available.

| Dataset | Source | Description |
|---------|--------|-------------|
| `Levine_13dim` | HDCytoData (Bioconductor) | CyTOF, human BM, 13 markers, 24 populations |
| `Levine_32dim` | HDCytoData (Bioconductor) | CyTOF, human BM, 32 markers, 14 populations |
| `Samusik_01`   | HDCytoData (Bioconductor) | CyTOF, mouse BM, 39 markers, 24 populations |
| `Nilsson_rare` | Nilsson et al. 2013 (FlowRepository) | Flow, human BM, HSC target (0.81 %) |
| `Mosmann_rare` | Mosmann et al. 2014 (FlowRepository) | Flow, human PBMC, activated CD4 T target (0.028 %) |

Download instructions using R (HDCytoData):

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("HDCytoData")

library(HDCytoData)
# Download and export each dataset as CSV
```

Alternatively, the CSV files are available from the repositories associated
with the original studies (see paper Data Availability section).

**Directory layout expected by `data_loader.py`:**

```
<DATA_ROOT>/
├── Levine_13dim/
│   └── *.csv or *.txt
├── Levine_32dim/
│   └── *.csv or *.txt
├── Samusik_01/
│   └── *.csv or *.txt
├── Nilsson_rare/
│   └── *.csv or *.txt
└── Mosmann_rare/
    └── *.csv or *.txt
```

Set the data root **before running any script**:

```bash
# Linux / macOS
export CYTOMETRY_DATA_ROOT=/path/to/datasets

# Windows CMD
set CYTOMETRY_DATA_ROOT=C:\path\to\datasets

# Windows PowerShell
$env:CYTOMETRY_DATA_ROOT = "C:\path\to\datasets"
```

Alternatively, edit the `DATA_ROOT` variable in `data_loader.py` directly.

---

## Project Structure

```
cytometry-clustering-benchmark/
│
├── data_loader.py          # Dataset loading & pre-processing (shared)
├── evaluation.py           # Evaluation metrics (ARI, Macro F1, P/R/F1; shared)
├── requirements.txt
│
├── 01_baseline/            # Main benchmark — one script per algorithm
│   ├── run_kmeans.py       # k-means         (K=true)
│   ├── run_gmm.py          # GMM             (K=true)
│   ├── run_ward.py         # Ward            (K=true, skipped when n > 60 k)
│   ├── run_flowsom.py      # FlowSOM         (K=true)
│   ├── run_blflowsom.py    # BL-FlowSOM      (K=true, in-house re-implementation)
│   ├── run_blflowsom_poc.py# BL-FlowSOM-PoC  (K=true, Sony official PoC)
│   ├── run_phenograph.py   # PhenoGraph      (K=auto, oracle parameter)
│   ├── run_leiden.py       # Leiden          (K=auto, oracle parameter)
│   └── run_hdbscan.py      # HDBSCAN         (K=auto, oracle parameter)
│
├── 02_extended/            # Extended experiments (SPADE, X-shift; to be added)
├── 03_stability/           # 30-run stability analysis & Dip test (to be added)
├── 04_diagnostics/         # Diagnostic plots (to be added)
│
└── results/
    └── baseline/           # CSV outputs from 01_baseline scripts
```

---

## Usage

Run each algorithm script independently.  Results are appended to CSV files in
`results/baseline/`.

```bash
# From the project root:
python 01_baseline/run_kmeans.py
python 01_baseline/run_gmm.py
python 01_baseline/run_ward.py
python 01_baseline/run_flowsom.py
python 01_baseline/run_blflowsom.py
python 01_baseline/run_phenograph.py
python 01_baseline/run_leiden.py
python 01_baseline/run_hdbscan.py

# Sony PoC (requires bl-flowsom_poc/ to be cloned):
python 01_baseline/run_blflowsom_poc.py
```

---

## Algorithms

### K=true (cluster count fixed to true population count)

| Script | Algorithm | Key parameters |
|--------|-----------|----------------|
| `run_kmeans.py`     | k-means   | `n_init=20`, `random_state=42` |
| `run_gmm.py`        | GMM       | `covariance_type='full'`, `n_init=5` |
| `run_ward.py`       | Ward      | `linkage='ward'`; skipped when n > 60,000 |
| `run_flowsom.py`    | FlowSOM   | `xdim=ydim=10`, `seed=42` |
| `run_blflowsom.py`  | BL-FlowSOM| `rlen=100`, `sigma_end=0.3`, `pca_scale=2.0` |

### K=auto (cluster count determined by the algorithm)

Oracle parameter selection: the parameter value that maximises ARI against
ground-truth labels is reported (upper bound; not achievable without labels).

| Script | Algorithm | Parameter scan |
|--------|-----------|----------------|
| `run_phenograph.py` | PhenoGraph | k ∈ {15, 20, 30, 45, 60} |
| `run_leiden.py`     | Leiden     | resolution ∈ {0.3, 0.5, 0.8, 1.0, 1.5} |
| `run_hdbscan.py`    | HDBSCAN    | min_cluster_size ∈ {5, 10, 20, 50, 100} (large: {50, 100, 200}) |

---

## Evaluation Metrics

| Metric | Used for | Notes |
|--------|----------|-------|
| ARI | Multi-population datasets (primary) | Global partition agreement |
| Macro F1 | Multi-population datasets (secondary) | Hungarian alignment; equal weight per population |
| Populations with F1 = 0 | Multi-population datasets | Coverage completeness |
| Target-population F1 / P / R | Rare-population datasets (primary) | HSC or activated CD4 T |
| F2 (β = 2) | Rare-population datasets | Recall-prioritised auxiliary metric |

**Key finding:** ARI is unsuitable as a parameter-selection criterion on rare-
population datasets (Mosmann_rare: ARI < 0.001 across all parameters while F1
varies 0.003–0.684).  Use target-population F1 instead.

---

## BL-FlowSOM Re-implementation Notes

`run_blflowsom.py` contains an independent re-implementation of BL-FlowSOM
(Otsuka et al., 2025) with three bug-fixes applied relative to the prototype:

| # | Issue | Fix |
|---|-------|-----|
| 1 | σ decay clamped by `max(1.0, …)`, freezing final 30 % of iterations | Linear decay from `sigma_start` to `sigma_end` without clamping |
| 2 | `rlen=10` — correct for online SOM, far too few for batch learning | `rlen=100` (literature recommendation: 50–200) |
| 3 | PCA initialisation covered only ±1 σ per PC axis | Extended to ±`pca_scale`=2 σ for better data-space coverage |

The corrected re-implementation was validated against the Sony PoC
(max ΔARI = 0.082 across all datasets; Supplementary Table ST1).

---

## Citation

If you use this code, please cite:

```bibtex
@article{TODO_citation_key,
  title   = {Benchmarking Unsupervised Clustering in High-Dimensional Cytometry:
             Convergence Stability and Evaluation Metric Applicability},
  journal = {Cytometry Part A},
  year    = {2025},
  doi     = {TODO}
}
```

---

## License

MIT License — see `LICENSE` for details.
