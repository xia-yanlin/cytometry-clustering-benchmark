"""
01_baseline/run_ward.py
=======================
Run Ward hierarchical clustering on all five benchmark datasets.

Algorithm
---------
sklearn AgglomerativeClustering with Ward linkage and K = true population
count (K=true setting).

Scalability limit
-----------------
Ward clustering has O(n²) time and memory complexity.  Datasets with more
than MAX_CELLS annotated cells are automatically skipped and recorded as N/A,
consistent with the paper (Weber & Robinson 2016 threshold = 60,000 cells).

Expected skip behaviour
-----------------------
Levine_13dim : 81,747 annotated cells → skipped
Levine_32dim : 104,184 annotated cells → skipped
Samusik_01   : 53,173 annotated cells → evaluated
Nilsson_rare : 44,140 cells           → evaluated
Mosmann_rare : 396,460 cells          → skipped
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "Ward"
K_MODE      = "K=true"
MAX_CELLS   = 60_000
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "ward_results.csv",
)


def run_ward(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit Ward hierarchical clustering and return 0-based cluster labels."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print(f"Cell-count limit: {MAX_CELLS:,} (O(n²) complexity)")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]
        k = TRUE_K[name]
        n = X.shape[0]

        print(f"\n[{name}]  cells={n:,}  markers={X.shape[1]}  K={k}")

        if n > MAX_CELLS:
            print(f"  SKIP: {n:,} cells exceeds limit {MAX_CELLS:,} (recorded as N/A)")
            row = {
                "dataset":          name,
                "algorithm":        ALGORITHM,
                "K_mode":           K_MODE,
                "n_clusters_true":  k,
                "n_clusters_found": None,
                "n_unmatched_cells": None,
                "ARI":              None,
                "MacroF1":          None,
                "accuracy":         None,
                "runtime_sec":      None,
                "skip_reason":      f"cells={n} > MAX_CELLS={MAX_CELLS}",
                "per_pop_f1":       {},
                "per_pop_pr":       {},
            }
            all_rows.append(row)
            continue

        print("  Running...", end="", flush=True)

        t0     = time.time()
        y_pred = run_ward(X, n_clusters=k)
        rt     = round(time.time() - t0, 1)

        print(f" done ({rt}s)")

        metrics = evaluate(y, y_pred, dataset_name=name)
        print_summary(metrics, name, ALGORITHM)

        row = {
            "dataset":         name,
            "algorithm":       ALGORITHM,
            "K_mode":          K_MODE,
            "n_clusters_true": k,
            "runtime_sec":     rt,
        }
        for k2, v in metrics.items():
            if k2 not in ("per_pop_f1", "per_pop_pr"):
                row[k2] = v
        row["per_pop_f1"] = metrics.get("per_pop_f1", {})
        row["per_pop_pr"] = metrics.get("per_pop_pr", {})
        all_rows.append(row)

    save_results(all_rows, RESULTS_CSV)
    print(f"\nResults saved to {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
