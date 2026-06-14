"""
01_baseline/run_gmm.py
======================
Run Gaussian Mixture Model (GMM) clustering on all five benchmark datasets.

Algorithm
---------
sklearn GaussianMixture with full covariance matrices and K = true population
count (K=true setting), consistent with Weber & Robinson (2016).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
from sklearn.mixture import GaussianMixture

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "GMM"
K_MODE      = "K=true"
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "gmm_results.csv",
)

GMM_PARAMS = dict(
    covariance_type = "full",
    max_iter        = 200,
    n_init          = 5,
    random_state    = 42,
)


def run_gmm(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit GMM and return 0-based component labels."""
    return GaussianMixture(n_components=n_clusters, **GMM_PARAMS).fit_predict(X)


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
        y_pred = run_gmm(X, n_clusters=k)
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
