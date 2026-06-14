"""
01_baseline/run_blflowsom_poc.py
================================
Run BL-FlowSOM using the Sony official PoC implementation.

Purpose
-------
This script wraps the Sony reference implementation
(https://github.com/sony/bl-flowsom_poc) to produce results that are
directly comparable to the original paper (Otsuka et al., 2025).
For the in-house re-implementation see run_blflowsom.py.

Prerequisites
-------------
Clone and install the Sony PoC repository at the project root:

    git clone https://github.com/sony/bl-flowsom_poc
    cd bl-flowsom_poc && pip install -r requirements.txt

Expected directory layout
-------------------------
    <project_root>/
    ├── bl-flowsom_poc/          ← Sony PoC clone
    │   └── build_batchSOM.py
    ├── data_loader.py
    ├── evaluation.py
    └── 01_baseline/
        └── run_blflowsom_poc.py ← this file

Pipeline (per dataset)
----------------------
1. Export X to a temporary CSV (no header; Sony PoC reads raw values).
2. Call:  python build_batchSOM.py -d 10 -s 1 -i <input> -o <output> -c <codes>
3. Read back per-cell node assignments (1-based → 0-based) and node vectors.
4. Ward meta-clustering on node vectors → meta-cluster labels.
5. Map node labels back to cell labels.
6. Clean up temporary files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "BL-FlowSOM-PoC"
K_MODE      = "K=true"
SOM_DIM     = 10    # SOM grid side length  (-d parameter)
POC_SEED    = 1     # Sony PoC random seed  (-s parameter)

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POC_DIR    = os.path.join(_ROOT, "bl-flowsom_poc")
POC_SCRIPT = os.path.join(POC_DIR, "build_batchSOM.py")

RESULTS_CSV = os.path.join(
    _ROOT, "results", "baseline", "blflowsom_poc_results.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_poc_installed() -> None:
    """Raise FileNotFoundError with a clear installation message if PoC is absent."""
    if not os.path.isfile(POC_SCRIPT):
        raise FileNotFoundError(
            f"\nSony PoC script not found: {POC_SCRIPT}\n\n"
            "Please run the following commands at the project root:\n"
            f"  cd {_ROOT}\n"
            "  git clone https://github.com/sony/bl-flowsom_poc\n"
            "  cd bl-flowsom_poc\n"
            "  pip install -r requirements.txt\n"
        )


def run_poc(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Call the Sony PoC and return 0-based meta-cluster labels.

    Parameters
    ----------
    X          : np.ndarray (n_cells, n_markers)
    n_clusters : Number of meta-clusters (= true population count)

    Returns
    -------
    np.ndarray (n_cells,) of 0-based int labels
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv  = os.path.join(tmpdir, "input.csv")
        output_csv = os.path.join(tmpdir, "output.csv")   # node assignments
        codes_csv  = os.path.join(tmpdir, "codes.csv")    # node vectors

        # Step 1: export data (no header; PoC reads numeric columns directly)
        np.savetxt(input_csv, X, delimiter=",")

        # Step 2: invoke Sony PoC
        cmd = [
            sys.executable,
            POC_SCRIPT,
            "-d", str(SOM_DIM),
            "-s", str(POC_SEED),
            "-i", input_csv,
            "-o", output_csv,
            "-c", codes_csv,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=POC_DIR
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Sony PoC failed (returncode={result.returncode})\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Step 3: read node assignments (PoC outputs 1-based → convert to 0-based)
        node_assignments = (
            pd.read_csv(output_csv, header=None).values.flatten().astype(int) - 1
        )

        # Step 4: read node vectors
        codes = pd.read_csv(codes_csv, header=None).values

    # Step 5: Ward meta-clustering
    Z         = linkage(codes, method="ward")
    node_meta = fcluster(Z, n_clusters, criterion="maxclust") - 1   # 0-based

    # Step 6: map back to cell labels
    return node_meta[node_assignments].astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_poc_installed()

    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print(f"Sony PoC: {POC_SCRIPT}")
    print(f"SOM grid: {SOM_DIM}×{SOM_DIM} = {SOM_DIM**2} nodes  seed={POC_SEED}")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]
        k = TRUE_K[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}  K_true={k}")

        t0     = time.time()
        y_pred = run_poc(X, n_clusters=k)
        rt     = round(time.time() - t0, 1)

        metrics = evaluate(y, y_pred, dataset_name=name)
        print_summary(metrics, name, ALGORITHM)
        print(f"  runtime: {rt}s")

        row = {
            "dataset":         name,
            "algorithm":       ALGORITHM,
            "K_mode":          K_MODE,
            "n_clusters_true": k,
            "som_dim":         SOM_DIM,
            "poc_seed":        POC_SEED,
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
