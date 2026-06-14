"""
01_baseline/run_blflowsom.py
============================
Run BL-FlowSOM (our re-implementation) on all five benchmark datasets.

Algorithm
---------
BL-FlowSOM (Otsuka et al., Cytometry Part A, 2025) extends FlowSOM with:
  1. PCA initialisation — eliminates the random-seed dependence that causes
     FlowSOM's bimodal convergence.
  2. Batch learning — all cells update the SOM weights simultaneously in
     each iteration, removing dependence on data-presentation order.

This re-implementation was validated against the Sony PoC reference
(max ΔARI = 0.082 across all datasets; see Supplementary Table ST1).
For results using the authors' official code see run_blflowsom_poc.py.

Bug-fixes vs. original prototype
---------------------------------
Fix 1  sigma decay — original code clamped sigma with max(1.0, …),
       freezing the final ~30 % of iterations at σ = 1.  Corrected to a
       clean linear decay from sigma_start down to SIGMA_END.
Fix 2  rlen — original used rlen=10, matching online FlowSOM convention
       but far too few passes for batch learning.  Set to RLEN=100
       (literature recommendation: 50–200).
Fix 3  PCA initialisation spread — original covered ±1 σ along each PC
       axis.  Corrected to ±PCA_SCALE=2 σ for better coverage of data space.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

ALGORITHM   = "BL-FlowSOM"
K_MODE      = "K=true"
SOM_DIM     = 10       # SOM grid side length → 10 × 10 = 100 nodes

# Corrected hyper-parameters
RLEN        = 100      # Fix 2: batch SOM requires 50–200 passes
SIGMA_START = None     # None → auto = max(SOM_DIM, SOM_DIM) / 2 = 5.0
SIGMA_END   = 0.3      # Fix 1: linear decay target (no clamping)
PCA_SCALE   = 2.0      # Fix 3: initialisation spread ± 2 σ along each PC

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "baseline", "blflowsom_results.csv",
)


# ---------------------------------------------------------------------------
# Batch SOM implementation
# ---------------------------------------------------------------------------

class BatchSOM:
    """
    Batch-learning Self-Organising Map (BL-SOM).

    Key differences from standard FlowSOM
    --------------------------------------
    1. PCA initialisation: node vectors are placed along the first two
       principal components of the data, removing random initialisation.
    2. Batch update: all cells determine their best-matching unit (BMU)
       first; then all node weights are updated at once.  The result is
       independent of data-presentation order, guaranteeing determinism.

    Parameters
    ----------
    grid_size   : (rows, cols) of the SOM grid.
    sigma_start : Initial neighbourhood width; None → max(rows, cols) / 2.
    sigma_end   : Final neighbourhood width (no clamping applied).
    rlen        : Number of training epochs (recommended 50–200).
    pca_scale   : PC-axis spread for initialisation (recommended 2.0 = ±2 σ).
    """

    def __init__(
        self,
        grid_size:   tuple[int, int] = (10, 10),
        sigma_start: float | None = None,
        sigma_end:   float = 1.0,
        rlen:        int   = 100,
        pca_scale:   float = 2.0,
    ) -> None:
        self.n_rows, self.n_cols = grid_size
        self.n_nodes  = self.n_rows * self.n_cols
        self.sigma_start = (
            sigma_start if sigma_start is not None
            else max(self.n_rows, self.n_cols) / 2.0
        )
        self.sigma_end = sigma_end
        self.rlen      = rlen
        self.pca_scale = pca_scale
        self.codes: np.ndarray | None = None

        # Grid coordinates for neighbourhood distance computation
        self._positions = np.array(
            [(i, j) for i in range(self.n_rows) for j in range(self.n_cols)],
            dtype=float,
        )

    def _pca_init(self, X: np.ndarray) -> None:
        """
        Initialise node vectors along the first two principal components.

        Nodes are placed on a regular grid spanning ±pca_scale standard
        deviations along each PC axis (Fix 3: was ±1 σ).
        """
        n_comp = min(2, X.shape[1])
        pca    = PCA(n_components=n_comp).fit(X)
        mean   = X.mean(axis=0)

        rs = np.linspace(-self.pca_scale, self.pca_scale, self.n_rows)
        cs = np.linspace(-self.pca_scale, self.pca_scale, self.n_cols)

        codes = np.zeros((self.n_nodes, X.shape[1]))
        idx   = 0
        for r in rs:
            for c in cs:
                codes[idx] = mean.copy()
                codes[idx] += r * np.sqrt(pca.explained_variance_[0]) * pca.components_[0]
                if n_comp > 1:
                    codes[idx] += (
                        c * np.sqrt(pca.explained_variance_[1]) * pca.components_[1]
                    )
                idx += 1
        self.codes = codes

    def _neighbourhood(self, sigma: float) -> np.ndarray:
        """Gaussian neighbourhood weight matrix H[i,j] = exp(−d²(i,j) / 2σ²)."""
        sq = cdist(self._positions, self._positions, "sqeuclidean")
        return np.exp(-sq / (2.0 * sigma ** 2))

    def _find_bmu(self, X: np.ndarray, chunk: int = 8_000) -> np.ndarray:
        """Chunked BMU search to avoid out-of-memory on large datasets."""
        winners = np.empty(len(X), dtype=int)
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            winners[start:end] = cdist(X[start:end], self.codes, "sqeuclidean").argmin(1)
        return winners

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        PCA-initialise and batch-train the SOM; return per-cell BMU indices.

        Sigma schedule (Fix 1): linear decay from sigma_start to sigma_end,
        with no clamping.
            t = 0        → σ = sigma_start
            t = rlen − 1 → σ = sigma_end
        """
        X = np.asarray(X, dtype=float)
        self._pca_init(X)

        for t in range(self.rlen):
            progress = t / (self.rlen - 1) if self.rlen > 1 else 1.0
            sigma    = self.sigma_start + (self.sigma_end - self.sigma_start) * progress

            H       = self._neighbourhood(sigma)   # (n_nodes, n_nodes)
            winners = self._find_bmu(X)

            node_sums   = np.zeros_like(self.codes)
            node_counts = np.zeros(self.n_nodes)
            np.add.at(node_sums,   winners, X)
            np.add.at(node_counts, winners, 1)

            # Batch update: new_code[i] = Σ_j H[i,j]·sums[j] / Σ_j H[i,j]·counts[j]
            numerator   = H @ node_sums
            denominator = H @ node_counts
            nz          = denominator > 0
            new_codes   = self.codes.copy()
            new_codes[nz] = numerator[nz] / denominator[nz, None]
            self.codes = new_codes

        return self._find_bmu(X)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_blflowsom(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Full BL-FlowSOM pipeline: train BatchSOM → Ward meta-clustering.

    Steps
    -----
    1. BatchSOM training (PCA init + batch updates).
    2. Ward hierarchical clustering of the 100 node vectors into n_clusters.
    3. Map node meta-cluster labels back to individual cells.
    """
    som = BatchSOM(
        grid_size   = (SOM_DIM, SOM_DIM),
        sigma_start = SIGMA_START,   # None → auto = 5.0
        sigma_end   = SIGMA_END,
        rlen        = RLEN,
        pca_scale   = PCA_SCALE,
    )
    node_labels = som.fit_transform(X)           # (n_cells,) ∈ {0, …, 99}
    codes       = som.codes                      # (100, n_features)

    Z         = linkage(codes, method="ward")
    node_meta = fcluster(Z, n_clusters, criterion="maxclust") - 1  # 0-based

    return node_meta[node_labels].astype(int)


def main() -> None:
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}  |  K setting: {K_MODE}")
    print(f"SOM grid: {SOM_DIM}×{SOM_DIM} = {SOM_DIM**2} nodes")
    print(
        f"rlen={RLEN}  sigma_start={'auto=5.0' if SIGMA_START is None else SIGMA_START}"
        f"  sigma_end={SIGMA_END}  pca_scale={PCA_SCALE}"
    )
    print("Deterministic: YES (PCA init + batch learning, no randomness)")
    print("=" * 60)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, _ = datasets[name]
        k = TRUE_K[name]

        print(f"\n[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}  K_true={k}")

        t0     = time.time()
        y_pred = run_blflowsom(X, n_clusters=k)
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
            "rlen":            RLEN,
            "sigma_start":     SIGMA_START if SIGMA_START is not None else SOM_DIM / 2,
            "sigma_end":       SIGMA_END,
            "pca_scale":       PCA_SCALE,
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
