"""
01_baseline/run_hdbscan.py
==========================
Run HDBSCAN on all five benchmark datasets and save results.

Algorithm
---------
HDBSCAN (Campello et al., 2013) via sklearn ≥ 1.3 (no separate install
required).  HDBSCAN makes no shape assumptions and is robust to
variable-density data.  K is determined automatically (K=auto).

Noise handling
--------------
Cells assigned label −1 (noise) are consolidated into a single additional
cluster so that all cells receive a label and evaluation metrics are
well-defined.

Parameter scan
--------------
min_cluster_size ∈ {5, 10, 20, 50, 100}  for datasets with ≤ 100,000 cells.
min_cluster_size ∈ {50, 100, 200}         for datasets with > 100,000 cells.
The value that maximises ARI is selected (oracle/upper-bound conditions).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.cluster import HDBSCAN

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM      = "HDBSCAN"
K_MODE         = "K=auto"
MCS_SCAN       = [5, 10, 20, 50, 100]     # standard datasets (≤ 100 k cells)
MCS_SCAN_LARGE = [50, 100, 200]           # large datasets (> 100 k cells)
MAX_CELLS_FULL = 100_000
RESULTS_CSV    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "hdbscan_results.csv",
)


def run_hdbscan(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Fit HDBSCAN; noise points (−1) are merged into a single extra cluster."""
    labels = HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_cluster_size,
        n_jobs           = 1,
    ).fit_predict(X)

    if -1 in labels:
        noise_id = int(labels.max()) + 1
        labels   = np.where(labels == -1, noise_id, labels)
    return labels.astype(int)


def scan_and_select(X: np.ndarray, y: np.ndarray, dataset_name: str) -> tuple:
    """Scan min_cluster_size values; return the best metrics and parameter."""
    mcs_list = MCS_SCAN if X.shape[0] <= MAX_CELLS_FULL else MCS_SCAN_LARGE
    if X.shape[0] > MAX_CELLS_FULL:
        print(f"  ({X.shape[0]:,} cells → using scan range {mcs_list})")

    best_ari, best_mcs, best_metrics = -1.0, None, None

    for mcs in mcs_list:
        print(f"    mcs={mcs:<4} ...", end="", flush=True)
        t0     = time.time()
        y_pred = run_hdbscan(X, min_cluster_size=mcs)
        rt     = round(time.time() - t0, 1)

        metrics = evaluate(y, y_pred, dataset_name=dataset_name)
        ari     = metrics["ARI"]
        print(f" ARI={ari:.4f}  K_found={metrics['n_clusters_found']}  ({rt}s)")

        if ari > best_ari:
            best_ari, best_mcs, best_metrics = ari, mcs, metrics

    return best_metrics, best_mcs


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print(f"min_cluster_size scan (standard): {MCS_SCAN}")
    print(f"min_cluster_size scan (large):    {MCS_SCAN_LARGE}")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}")

        t_total            = time.time()
        best_metrics, best_mcs = scan_and_select(X, y, name)
        total_rt           = round(time.time() - t_total, 1)

        print(f"  → best mcs={best_mcs}  ", end="")
        print_summary(best_metrics, name, ALGORITHM)

        row = {
            "dataset":               name,
            "algorithm":             ALGORITHM,
            "K_mode":                K_MODE,
            "n_clusters_true":       TRUE_K[name],
            "best_min_cluster_size": best_mcs,
            "mcs_scan":              str(MCS_SCAN if X.shape[0] <= MAX_CELLS_FULL
                                         else MCS_SCAN_LARGE),
            "runtime_sec":           total_rt,
        }
        for k2, v in best_metrics.items():
            if k2 not in ("per_pop_f1", "per_pop_pr"):
                row[k2] = v
        row["per_pop_f1"] = best_metrics.get("per_pop_f1", {})
        row["per_pop_pr"] = best_metrics.get("per_pop_pr", {})
        all_rows.append(row)

    save_results(all_rows, RESULTS_CSV)
    print(f"\nResults saved to {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
