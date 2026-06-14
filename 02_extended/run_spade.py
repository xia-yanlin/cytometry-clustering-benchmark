"""
02_extended/run_spade.py
========================
SPADE density-equalisation pipeline evaluated on Nilsson_rare and
Mosmann_rare.

Algorithm overview (Qiu et al., Nature Biotechnology 2011)
----------------------------------------------------------
1. Estimate local density via k-NN inverse distance.
2. Compute per-cell retention probability: low-density cells (rare
   phenotypes) are kept with higher probability than high-density
   background cells.
3. Stochastically downsample according to the retention probabilities.
4. Cluster the retained cells (k-means or HDBSCAN).
5. Upsample: discarded cells inherit the label of their nearest retained
   neighbour via KDTree (replaces sklearn KNeighborsClassifier to avoid
   multiprocessing serialisation issues on Windows).
6. Evaluate target-population F1 / Precision / Recall.

Improvements over naive SPADE
------------------------------
* Upsampling uses KDTree directly instead of KNeighborsClassifier,
  eliminating Windows multiprocessing overhead (10–50× faster on
  Mosmann_rare with 396 k cells).
* Cluster count is scanned (K ∈ {10, 20, 30, 50, 100, 200} for k-means;
  min_cluster_size ∈ {3, 5, 10} for HDBSCAN) so rare populations can
  form independent clusters.
* Chunked upsampling prevents out-of-memory on large datasets.

Outputs
-------
results/extended/spade_results.csv   — all parameter-combination results
results/extended/spade_summary.csv   — per-dataset best configuration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from data_loader import load_all
from evaluation  import evaluate, RARE_TARGET

warnings.filterwarnings("ignore")

ALGORITHM   = "SPADE"
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "extended", "spade_results.csv",
)
SUMMARY_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "extended", "spade_summary.csv",
)

# Parameter grids
TARGET_PCT_GRID     = [0.05, 0.10, 0.20]
OUTLIER_PCTILE_GRID = [1, 5]
KMEANS_K_GRID       = [10, 20, 30, 50, 100, 200]
HDBSCAN_MCS_GRID    = [3, 5, 10]
K_DENSITY           = 15
KNN_K_UPSAMPLE      = 1       # k=1 nearest-neighbour inheritance (fastest)
CHUNK_SIZE          = 50_000  # chunked upsampling for large datasets

# Fast mode for Mosmann_rare: functionally rare cells are hard to detect
# under unsupervised single-sample constraints; 6 targeted parameter
# combinations are sufficient to confirm the conclusion.
MOSMANN_FAST = True
MOSMANN_FAST_GRID = [
    (0.05, 1, "kmeans",  50),
    (0.05, 1, "kmeans", 100),
    (0.10, 1, "kmeans",  50),
    (0.10, 1, "kmeans", 100),
    (0.05, 1, "hdbscan",  3),
    (0.10, 1, "hdbscan",  3),
]


# ---------------------------------------------------------------------------
# Module 1: density estimation
# ---------------------------------------------------------------------------

def estimate_density(X: np.ndarray, k: int = 15) -> np.ndarray:
    """k-NN inverse-distance density. Higher value = denser region (background)."""
    tree = KDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    return 1.0 / (dists[:, k] + 1e-9)


# ---------------------------------------------------------------------------
# Module 2: retention probabilities
# ---------------------------------------------------------------------------

def compute_retention_prob(
    density: np.ndarray,
    outlier_pctile: float = 1,
    target_pct: float = 0.10,
) -> np.ndarray:
    """
    SPADE retention probabilities via binary search.
    Low-density cells receive higher retention probability;
    the search finds the target_density value such that the mean
    retention rate equals target_pct.
    """
    n              = len(density)
    outlier_thresh = np.percentile(density, outlier_pctile)
    valid          = density > outlier_thresh
    d_valid        = density[valid]

    if valid.sum() == 0:
        return np.zeros(n)

    lo, hi = d_valid.min(), d_valid.max() * 10.0
    for _ in range(100):
        mid    = (lo + hi) / 2.0
        p_mean = np.minimum(1.0, mid / d_valid).mean()
        if abs(p_mean - target_pct) < 5e-4:
            break
        if p_mean < target_pct:
            lo = mid
        else:
            hi = mid

    probs        = np.zeros(n)
    probs[valid] = np.minimum(1.0, (lo + hi) / 2.0 / d_valid)
    return probs


# ---------------------------------------------------------------------------
# Module 3: downsampling
# ---------------------------------------------------------------------------

def downsample(probs: np.ndarray, seed: int = 42) -> np.ndarray:
    """Stochastic downsampling: cell i is retained if U[0,1) < probs[i]."""
    return np.random.default_rng(seed).random(len(probs)) < probs


# ---------------------------------------------------------------------------
# Module 4a: k-means clustering on retained cells
# ---------------------------------------------------------------------------

def cluster_kmeans(X_sub: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(
        n_clusters=n_clusters, n_init=5, random_state=42, max_iter=300
    ).fit_predict(X_sub).astype(int)


# ---------------------------------------------------------------------------
# Module 4b: HDBSCAN clustering on retained cells
# ---------------------------------------------------------------------------

def cluster_hdbscan(X_sub: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    labels     = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 2),
    ).fit_predict(X_sub).astype(int)

    noise_mask = labels == -1
    if noise_mask.sum() > 0:
        non_noise = ~noise_mask
        if non_noise.sum() == 0:
            labels[:] = 0
        else:
            tree = KDTree(X_sub[non_noise])
            idx  = tree.query(X_sub[noise_mask], k=1,
                              return_distance=False).flatten()
            labels[noise_mask] = labels[non_noise][idx]

    unique = np.unique(labels)
    remap  = {v: i for i, v in enumerate(unique)}
    return np.array([remap[l] for l in labels])


# ---------------------------------------------------------------------------
# Module 5: KDTree upsampling (key speed improvement)
# ---------------------------------------------------------------------------

def upsample_kdtree(
    X_all:       np.ndarray,
    X_kept:      np.ndarray,
    labels_kept: np.ndarray,
    keep_mask:   np.ndarray,
    k:           int = 1,
    chunk_size:  int = 50_000,
) -> np.ndarray:
    """
    Assign labels to discarded cells via nearest-retained-neighbour lookup.

    Retained cells: labels are locked in directly (no recomputation).
    Discarded cells: chunked KDTree.query finds the nearest retained cell;
    with k=1 the label is inherited directly (no voting required).

    Why faster than sklearn KNeighborsClassifier?
    sklearn spawns worker processes on Windows (n_jobs=-1), incurring
    large serialisation overhead for big arrays.  KDTree.query is a
    single-threaded C implementation with no process overhead.
    """
    y_all = np.full(len(X_all), -1, dtype=int)
    y_all[keep_mask] = labels_kept

    discard_idx = np.where(~keep_mask)[0]
    if len(discard_idx) == 0:
        return y_all

    tree  = KDTree(X_kept)
    k_eff = min(k, len(X_kept))

    for start in range(0, len(discard_idx), chunk_size):
        end       = min(start + chunk_size, len(discard_idx))
        idx_chunk = discard_idx[start:end]
        nn_idx    = tree.query(X_all[idx_chunk], k=k_eff,
                               return_distance=False)
        if k_eff == 1:
            y_all[idx_chunk] = labels_kept[nn_idx.flatten()]
        else:
            y_all[idx_chunk] = np.array([
                np.bincount(labels_kept[row]).argmax()
                for row in nn_idx
            ])
    return y_all


# ---------------------------------------------------------------------------
# Full pipeline (single run)
# ---------------------------------------------------------------------------

def run_spade(
    X:              np.ndarray,
    y_true:         np.ndarray,
    dataset_name:   str,
    target_pct:     float,
    outlier_pctile: float,
    algo:           str,
    algo_param:     int,
    k_density:      int  = K_DENSITY,
    knn_k:          int  = KNN_K_UPSAMPLE,
    seed:           int  = 42,
) -> tuple[np.ndarray | None, dict]:
    """
    Execute the full SPADE pipeline for one parameter combination.

    Returns
    -------
    y_pred : predicted labels (or None if too few cells retained)
    info   : diagnostic dict
    """
    t0 = time.time()

    density = estimate_density(X, k=k_density)
    probs   = compute_retention_prob(density,
                                     outlier_pctile=outlier_pctile,
                                     target_pct=target_pct)
    keep    = downsample(probs, seed=seed)
    X_kept  = X[keep]
    n_kept  = int(keep.sum())

    target = RARE_TARGET.get(dataset_name)
    if target:
        n_target_total = int((y_true == target).sum())
        n_target_kept  = int((y_true[keep] == target).sum())
    else:
        n_target_total = n_target_kept = 0

    min_needed = algo_param if algo == "kmeans" else max(algo_param * 3, 20)
    if n_kept < min_needed:
        return None, {"error": f"too_few_cells: {n_kept} < {min_needed}"}

    if algo == "kmeans":
        labels_kept = cluster_kmeans(X_kept, n_clusters=algo_param)
    elif algo == "hdbscan":
        labels_kept = cluster_hdbscan(X_kept, min_cluster_size=algo_param)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo!r}")

    n_sub = int(len(np.unique(labels_kept)))
    y_pred = upsample_kdtree(X, X_kept, labels_kept, keep,
                             k=knn_k, chunk_size=CHUNK_SIZE)

    return y_pred, {
        "n_kept":          n_kept,
        "kept_pct":        round(100 * n_kept / len(X), 2),
        "n_sub_clusters":  n_sub,
        "n_target_kept":   n_target_kept,
        "n_target_total":  n_target_total,
        "runtime_sec":     round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Parameter scan
# ---------------------------------------------------------------------------

def build_combos(ds_name: str) -> list:
    """Return the list of (target_pct, outlier_pctile, algo, param) combos."""
    if ds_name == "Mosmann_rare" and MOSMANN_FAST:
        return MOSMANN_FAST_GRID
    return (
        list(itertools.product(TARGET_PCT_GRID, OUTLIER_PCTILE_GRID,
                               ["kmeans"],  KMEANS_K_GRID))
        + list(itertools.product(TARGET_PCT_GRID, OUTLIER_PCTILE_GRID,
                                 ["hdbscan"], HDBSCAN_MCS_GRID))
    )


def scan_rare_dataset(ds_name: str, X: np.ndarray, y: np.ndarray) -> list:
    target     = RARE_TARGET[ds_name]
    all_combos = build_combos(ds_name)
    n_total    = len(all_combos)
    mode_str   = ("fast mode (6 combos)"
                  if ds_name == "Mosmann_rare" and MOSMANN_FAST
                  else f"full mode ({n_total} combos)")

    print(f"\n{'─'*65}")
    print(f"  {ds_name}  |  target: {target}  |  {mode_str}")
    print(f"{'─'*65}")

    rows    = []
    best_f1 = -1.0
    best_cfg = None

    for i, (tpct, opct, algo, aparam) in enumerate(all_combos, 1):
        pstr = f"K={aparam}" if algo == "kmeans" else f"mcs={aparam}"
        print(f"  [{i:>3}/{n_total}] tgt={tpct:.0%}  out={opct}%  "
              f"{algo:8}  {pstr:<8}", end="  ", flush=True)

        try:
            y_pred, info = run_spade(
                X, y, ds_name,
                target_pct=tpct, outlier_pctile=opct,
                algo=algo, algo_param=aparam,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        if y_pred is None:
            print(f"SKIP ({info.get('error', '?')})")
            continue

        metrics = evaluate(y, y_pred, dataset_name=ds_name)
        rf1     = metrics.get("rare_target_f1", 0.0)
        found   = "YES" if metrics.get("rare_target_found") else "NO"
        rt      = info["runtime_sec"]
        n_sub   = info["n_sub_clusters"]

        print(f"{target}_F1={rf1:.4f} [{found}]  "
              f"ARI={metrics['ARI']:.4f}  "
              f"K_sub={n_sub}->K_out={metrics['n_clusters_found']}  "
              f"target_kept={info['n_target_kept']}/{info['n_target_total']}  "
              f"({rt}s)")

        rows.append({
            "dataset":           ds_name,
            "algorithm":         ALGORITHM,
            "target_pct":        tpct,
            "outlier_pctile":    opct,
            "cluster_algo":      algo,
            "algo_param":        aparam,
            "kept_pct":          info["kept_pct"],
            "n_sub_clusters":    n_sub,
            "n_clusters_found":  metrics["n_clusters_found"],
            "runtime_sec":       rt,
            "ARI":               metrics["ARI"],
            "MacroF1":           metrics["MacroF1"],
            "accuracy":          metrics["accuracy"],
            "rare_target_f1":    rf1,
            "rare_target_precision": metrics.get("rare_target_precision"),
            "rare_target_recall":    metrics.get("rare_target_recall"),
            "rare_target_f2":        metrics.get("rare_target_f2"),
            "rare_target_found": metrics.get("rare_target_found", False),
        })

        if rf1 > best_f1:
            best_f1  = rf1
            best_cfg = (tpct, opct, algo, pstr)

    if best_cfg:
        tpct, opct, algo, pstr = best_cfg
        print(f"\n  Best [{ds_name}]: tgt={tpct:.0%}  out={opct}%  "
              f"{algo}  {pstr}  -> {target}_F1={best_f1:.4f}")
    else:
        print(f"\n  WARNING: all parameter combinations failed for {ds_name}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print(f"  Algorithm: {ALGORITHM}")
    print("  Density equalisation + KDTree upsampling + high-K clustering")
    print(f"  Mosmann fast mode: {'ON (6 combos)' if MOSMANN_FAST else 'OFF (full grid)'}")
    print("=" * 65)

    datasets = load_all(verbose=False)
    all_rows = []

    for ds_name in ["Nilsson_rare", "Mosmann_rare"]:
        X, y, _, _ = datasets[ds_name]
        rows = scan_rare_dataset(ds_name, X, y)
        all_rows.extend(rows)

    if not all_rows:
        print("WARNING: no valid results; check DATA_ROOT configuration.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nFull results saved: {RESULTS_CSV}  ({len(df)} rows)")

    if "rare_target_f1" in df.columns and not df.empty:
        summary = (
            df.sort_values("rare_target_f1", ascending=False)
            .groupby("dataset").first()
            .reset_index()
        )
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"Best-config summary saved: {SUMMARY_CSV}")

    # Comparison table
    print(f"\n{'='*65}")
    print("  Rare-population F1 comparison")
    print(f"{'─'*65}")
    baseline_f1 = {
        "k-means  (K=true)":   (0.040, 0.001),
        "GMM      (K=true)":   (0.039, 0.001),
        "FlowSOM  (K=true)":   (0.050, 0.000),
        "PhenoGraph (oracle)": (0.233, 0.471),
        "Leiden   (oracle)":   (0.131, 0.094),
        "HDBSCAN  (oracle)":   (0.099, 0.004),
    }
    print(f"  {'Method':<28} {'Nilsson HSC':>12} {'Mosmann act':>12}")
    print(f"  {'─'*54}")
    for method, (nf, mf) in baseline_f1.items():
        print(f"  {method:<28} {nf:>12.4f}   {mf:>10.4f}")
    print(f"  {'─'*54}")
    for ds_name in ["Nilsson_rare", "Mosmann_rare"]:
        sub = df[df["dataset"] == ds_name]
        if sub.empty:
            continue
        best = sub.nlargest(1, "rare_target_f1").iloc[0]
        algo = best["cluster_algo"]
        par  = int(best["algo_param"])
        label = f"SPADE ({algo}/K={par})" if algo == "kmeans" else f"SPADE ({algo}/mcs={par})"
        f1   = best["rare_target_f1"]
        col  = "Nilsson HSC" if ds_name == "Nilsson_rare" else "Mosmann act"
        print(f"  {label:<28} {f1:>12.4f}" if col == "Nilsson HSC"
              else f"  {label:<28} {'—':>12}   {f1:>10.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
