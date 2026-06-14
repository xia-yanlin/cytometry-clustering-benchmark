"""
01_baseline/run_kmeans.py
=========================
Run k-means clustering on all five benchmark datasets and save results.

Algorithm
---------
sklearn KMeans with K = true population count (K=true setting).
k-means is the simplest parametric baseline: it assumes spherical,
equal-density clusters.  n_init=20 reduces sensitivity to random
initialisation, consistent with Weber & Robinson (2016).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
from sklearn.cluster import KMeans

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "k-means"
K_MODE      = "K=true"
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "kmeans_results.csv",
)

KMEANS_PARAMS = dict(
    n_init       = 20,    # number of restarts (Weber 2016 setting)
    max_iter     = 300,
    random_state = 42,
)


def run_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit k-means and return 0-based cluster labels."""
    return KMeans(n_clusters=n_clusters, **KMEANS_PARAMS).fit_predict(X)


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]
        k = TRUE_K[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}  K={k}")
        print("  Running...", end="", flush=True)

        t0     = time.time()
        y_pred = run_kmeans(X, n_clusters=k)
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
