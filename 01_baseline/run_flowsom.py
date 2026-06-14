"""
01_baseline/run_flowsom.py
==========================
Run FlowSOM on all five benchmark datasets and save results.

Algorithm
---------
Official Python ``flowsom`` package (Couckuyt et al., 2024, Bioinformatics).
Parameters follow Weber & Robinson (2016): 10 × 10 SOM grid; meta-cluster
count = true population count per dataset (K=true setting).

Dependencies
------------
    pip install flowsom anndata

Validation target
-----------------
On Levine_13dim, FlowSOM MacroF1 should be within ±0.05 of the value
reported in Weber (2016) Table 3 (≈ 0.49).  A larger discrepancy likely
indicates a package-version or parameter mismatch.

Stochastic variability note
---------------------------
FlowSOM exhibits bimodal convergence on high-dimensional panels (≥ 32
markers); single-run results on Levine_32dim and Samusik_01 should be
interpreted with caution.  See the stability scripts in 03_stability/ for
the full 30-run analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
import pandas as pd

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "FlowSOM"
K_MODE      = "K=true"
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "flowsom_results.csv",
)


def run_flowsom(X: np.ndarray, marker_cols: list[str], n_clusters: int, seed: int = 42) -> np.ndarray:
    """
    Fit FlowSOM and return 0-based meta-cluster labels.

    Internally, FlowSOM trains a 10 × 10 = 100-node SOM and then merges
    the nodes into ``n_clusters`` meta-clusters via hierarchical clustering.
    Each cell is labelled by the meta-cluster of its winning SOM node.
    """
    import anndata as ad
    import flowsom as fs

    adata = ad.AnnData(X=pd.DataFrame(X, columns=marker_cols))
    fsom  = fs.FlowSOM(
        adata,
        n_clusters  = n_clusters,
        cols_to_use = marker_cols,
        xdim        = 10,
        ydim        = 10,
        seed        = seed,
    )

    raw_labels = np.array(fsom.get_cell_data().obs["metaclustering"])
    unique     = sorted(set(raw_labels))
    label_map  = {v: i for i, v in enumerate(unique)}
    return np.array([label_map[v] for v in raw_labels])


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print("=" * 60)

    try:
        import flowsom
        import anndata
        print(f"flowsom {flowsom.__version__}  anndata {anndata.__version__}")
    except Exception:
        pass

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]
        k = TRUE_K[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}  K={k}")
        print("  Running...", end="", flush=True)

        t0     = time.time()
        y_pred = run_flowsom(X, markers, n_clusters=k)
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

    # Validation cross-check
    print("\n--- Validation (Weber 2016 Table 3) ---")
    print("Expected FlowSOM MacroF1 on Levine_13dim ≈ 0.49 (±0.05)")
    for row in all_rows:
        if row["dataset"] == "Levine_13dim":
            mf1  = row.get("MacroF1")
            diff = abs(mf1 - 0.49) if isinstance(mf1, float) else None
            ok   = "PASS" if diff is not None and diff <= 0.05 else "CHECK parameters"
            print(f"  MacroF1={mf1}  |diff|={diff:.4f if diff else 'N/A'}  [{ok}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
