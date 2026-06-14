"""
01_baseline/run_leiden.py
=========================
Run Leiden clustering on all five benchmark datasets and save results.

Algorithm
---------
Leiden (Traag et al., 2019, Scientific Reports) via scanpy: build a kNN
graph then apply Leiden community detection.  K is determined automatically
(K=auto).  Leiden is an improvement over Louvain (used by PhenoGraph),
guaranteeing internally connected communities.

Parameter scan
--------------
resolution ∈ {0.3, 0.5, 0.8, 1.0, 1.5}; fixed n_neighbors = 15.
The resolution that maximises ARI is selected (oracle/upper-bound
conditions, consistent with Weber & Robinson 2016).

Dependencies
------------
    pip install scanpy leidenalg
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM       = "Leiden"
K_MODE          = "K=auto"
N_NEIGHBORS     = 15
RESOLUTION_SCAN = [0.3, 0.5, 0.8, 1.0, 1.5]
RESULTS_CSV     = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "leiden_results.csv",
)


def run_leiden(
    X: np.ndarray, resolution: float, n_neighbors: int = N_NEIGHBORS, seed: int = 42
) -> np.ndarray:
    """Build a kNN graph and run Leiden; return 0-based integer cluster labels."""
    import anndata as ad
    import scanpy  as sc

    adata = ad.AnnData(X=X.copy())
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    return adata.obs["leiden"].values.astype(int)


def scan_and_select(X: np.ndarray, y: np.ndarray, dataset_name: str) -> tuple:
    """Scan all resolution values; return the best metrics and resolution."""
    best_ari, best_res, best_metrics = -1.0, None, None

    for res in RESOLUTION_SCAN:
        print(f"    res={res:<4} ...", end="", flush=True)
        t0     = time.time()
        y_pred = run_leiden(X, resolution=res)
        rt     = round(time.time() - t0, 1)

        metrics = evaluate(y, y_pred, dataset_name=dataset_name)
        ari     = metrics["ARI"]
        print(f" ARI={ari:.4f}  K_found={metrics['n_clusters_found']}  ({rt}s)")

        if ari > best_ari:
            best_ari, best_res, best_metrics = ari, res, metrics

    return best_metrics, best_res


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print(f"n_neighbors={N_NEIGHBORS}  resolution scan: {RESOLUTION_SCAN}")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}")

        t_total            = time.time()
        best_metrics, best_res = scan_and_select(X, y, name)
        total_rt           = round(time.time() - t_total, 1)

        print(f"  → best res={best_res}  ", end="")
        print_summary(best_metrics, name, ALGORITHM)

        row = {
            "dataset":         name,
            "algorithm":       ALGORITHM,
            "K_mode":          K_MODE,
            "n_clusters_true": TRUE_K[name],
            "best_resolution": best_res,
            "resolution_scan": str(RESOLUTION_SCAN),
            "n_neighbors":     N_NEIGHBORS,
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
