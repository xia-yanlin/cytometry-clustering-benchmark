"""
01_baseline/run_phenograph.py
=============================
Run PhenoGraph on all five benchmark datasets and save results.

Algorithm
---------
PhenoGraph (Levine et al., 2015, Cell): kNN graph construction followed by
Louvain community detection.  K is determined automatically (K=auto).

Parameter scan
--------------
k ∈ {15, 20, 30, 45, 60} (kNN neighbour count).
The k that maximises ARI against ground-truth labels is selected and
reported (oracle/upper-bound conditions, as in Weber & Robinson 2016).

Dependencies
------------
    pip install phenograph

Windows note
------------
PhenoGraph uses multiprocessing internally; the ``if __name__ == '__main__':``
guard is required when running on Windows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "PhenoGraph"
K_MODE      = "K=auto"
K_SCAN      = [15, 20, 30, 45, 60]   # kNN neighbour counts to scan
SEED        = 42
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "phenograph_results.csv",
)


def run_phenograph(X: np.ndarray, k: int) -> np.ndarray:
    """
    Fit PhenoGraph with ``k`` neighbours; return 0-based cluster labels.
    Noise points (label = −1) are reassigned to a separate cluster.
    """
    import phenograph
    labels, _, _ = phenograph.cluster(X, k=k, seed=SEED, n_jobs=1)
    if -1 in labels:
        labels = np.where(labels == -1, int(labels.max()) + 1, labels)
    return labels.astype(int)


def scan_and_select(X: np.ndarray, y: np.ndarray, dataset_name: str) -> tuple:
    """
    Scan all k values; return the metrics, best k, and predicted labels for
    the run that achieves the highest ARI (oracle selection).
    """
    best_ari, best_k, best_metrics, best_pred = -1.0, None, None, None

    for k in K_SCAN:
        print(f"    k={k:<3} ...", end="", flush=True)
        t0     = time.time()
        y_pred = run_phenograph(X, k=k)
        rt     = round(time.time() - t0, 1)

        metrics = evaluate(y, y_pred, dataset_name=dataset_name)
        ari     = metrics["ARI"]
        print(f" ARI={ari:.4f}  K_found={metrics['n_clusters_found']}  ({rt}s)")

        if ari > best_ari:
            best_ari, best_k, best_metrics, best_pred = ari, k, metrics, y_pred

    return best_metrics, best_k, best_pred


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}  |  seed={SEED}")
    print(f"k scan: {K_SCAN}")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}")

        t_total              = time.time()
        best_metrics, best_k, _ = scan_and_select(X, y, name)
        total_rt             = round(time.time() - t_total, 1)

        print(f"  → best k={best_k}  ", end="")
        print_summary(best_metrics, name, ALGORITHM)

        row = {
            "dataset":         name,
            "algorithm":       ALGORITHM,
            "K_mode":          K_MODE,
            "seed":            SEED,
            "n_clusters_true": TRUE_K[name],
            "best_k":          best_k,
            "k_scan":          str(K_SCAN),
            "runtime_sec":     total_rt,
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
