"""
02_extended/run_spade_nooracle.py
==================================
SPADE non-oracle baseline experiment.

Purpose
-------
Quantify the oracle dependency of SPADE parameter selection by comparing
the oracle-optimal configuration (chosen using ground-truth labels) against
a fixed default configuration that requires no prior knowledge.

Design
------
* Nilsson_rare : fixed k-means K = 20 (mid-range, requires no prior).
* Mosmann_rare : fixed HDBSCAN min_cluster_size = 3 (smallest sensible value).
* target_pct = 5 %, outlier_pctile = 1 % (shown to be parameter-insensitive).
* 5 independent runs (seed 0–4); mean ± std reported.

Oracle reference values are taken from the best-configuration results
produced by run_spade.py (results/extended/spade_summary.csv).

Output
------
results/extended/spade_nooracle.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import KDTree
from scipy.optimize import linear_sum_assignment

from data_loader import load_dataset
from evaluation  import RARE_TARGET

warnings.filterwarnings("ignore")

# Default (non-oracle) configurations
DEFAULT_CONFIGS = {
    "Nilsson_rare": {"algo": "kmeans",  "param": 20},
    "Mosmann_rare": {"algo": "hdbscan", "param": 3},
}
TARGET_PCT     = 0.05
OUTLIER_PCTILE = 1
N_SEEDS        = 5

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "extended", "spade_nooracle.csv",
)

# Oracle-optimal F1 values from run_spade.py (used for comparison)
ORACLE_BEST = {
    "Nilsson_rare": 0.5568,
    "Mosmann_rare": 0.7235,
}


# ---------------------------------------------------------------------------
# Reused SPADE core functions (duplicated here to avoid cross-module import)
# ---------------------------------------------------------------------------

def estimate_density(X, k=15):
    tree = KDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    return 1.0 / (dists[:, k] + 1e-9)


def compute_retention_prob(density, outlier_pctile=1, target_pct=0.05):
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
        lo, hi = (mid, hi) if p_mean < target_pct else (lo, mid)
    probs        = np.zeros(n)
    probs[valid] = np.minimum(1.0, (lo + hi) / 2.0 / d_valid)
    return probs


def upsample_kdtree(X_all, X_kept, labels_kept, keep_mask, chunk=50_000):
    y_all = np.full(len(X_all), -1, dtype=int)
    y_all[keep_mask] = labels_kept
    discard = np.where(~keep_mask)[0]
    if len(discard) == 0:
        return y_all
    tree = KDTree(X_kept)
    for s in range(0, len(discard), chunk):
        e   = min(s + chunk, len(discard))
        idx = discard[s:e]
        nn  = tree.query(X_all[idx], k=1, return_distance=False).flatten()
        y_all[idx] = labels_kept[nn]
    return y_all


def hungarian_align(y_true, y_pred):
    tl, pc = np.unique(y_true), np.unique(y_pred)
    size   = max(len(tl), len(pc))
    cost   = np.zeros((size, size), dtype=np.int64)
    for i, t in enumerate(tl):
        for j, p in enumerate(pc):
            cost[i, j] = -int(np.sum((y_true == t) & (y_pred == p)))
    ri, ci  = linear_sum_assignment(cost)
    mapping = {pc[c]: tl[r] for r, c in zip(ri, ci)
               if r < len(tl) and c < len(pc)}
    return np.array([mapping.get(p, "__unmatched__") for p in y_pred])


def target_prf(y_true, y_aligned, target):
    tp = int(np.sum((y_true == target) & (y_aligned == target)))
    fp = int(np.sum((y_true != target) & (y_aligned == target)))
    fn = int(np.sum((y_true == target) & (y_aligned != target)))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r)   if (p+r)   > 0 else 0.0
    f2 = 5*p*r / (4*p+r) if (4*p+r) > 0 else 0.0
    return round(p,4), round(r,4), round(f1,4), round(f2,4)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(X, y_true, ds_name, algo, param, seed):
    t0  = time.time()
    rng = np.random.default_rng(seed)

    density = estimate_density(X, k=15)
    probs   = compute_retention_prob(density,
                                     outlier_pctile=OUTLIER_PCTILE,
                                     target_pct=TARGET_PCT)
    keep   = rng.random(len(probs)) < probs
    X_kept = X[keep]

    if algo == "kmeans":
        labels_kept = KMeans(n_clusters=param, n_init=5,
                             random_state=seed).fit_predict(X_kept)
    else:
        model  = HDBSCAN(min_cluster_size=param,
                         min_samples=max(1, param // 2))
        labels = model.fit_predict(X_kept).astype(int)
        noise  = labels == -1
        if noise.sum() > 0 and (~noise).sum() > 0:
            tree = KDTree(X_kept[~noise])
            idx  = tree.query(X_kept[noise], k=1,
                              return_distance=False).flatten()
            labels[noise] = labels[~noise][idx]
        elif noise.sum() > 0:
            labels[:] = 0
        uniq        = np.unique(labels)
        labels_kept = np.array([{v: i for i, v in enumerate(uniq)}[l]
                                 for l in labels])

    y_pred    = upsample_kdtree(X, X_kept, labels_kept, keep)
    y_aligned = hungarian_align(y_true, y_pred)
    target    = RARE_TARGET[ds_name]
    p, r, f1, f2 = target_prf(y_true, y_aligned, target)

    return {
        "seed":         seed,
        "F1":           f1,
        "precision":    p,
        "recall":       r,
        "F2":           f2,
        "K_found":      len(np.unique(y_pred)),
        "runtime_sec":  round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print(f"  SPADE Non-Oracle Baseline")
    print(f"  Nilsson: k-means K=20  |  Mosmann: HDBSCAN mcs=3")
    print(f"  {N_SEEDS} runs each (seed 0–{N_SEEDS-1})")
    print("=" * 62)

    all_rows = []

    for ds_name, cfg in DEFAULT_CONFIGS.items():
        algo, param = cfg["algo"], cfg["param"]
        target = RARE_TARGET[ds_name]

        print(f"\n{'─'*62}")
        print(f"  {ds_name}  |  {algo}  param={param}")
        print(f"{'─'*62}")

        X, y, _, _ = load_dataset(ds_name, verbose=False)

        seed_rows = []
        for seed in range(N_SEEDS):
            print(f"  [seed={seed}]", end="  ", flush=True)
            row = run_once(X, y, ds_name, algo, param, seed)
            seed_rows.append(row)
            print(f"F1={row['F1']:.4f}  P={row['precision']:.4f}  "
                  f"R={row['recall']:.4f}  K={row['K_found']}  "
                  f"({row['runtime_sec']}s)")

        f1_arr   = np.array([r["F1"] for r in seed_rows])
        mean_f1  = float(f1_arr.mean())
        std_f1   = float(f1_arr.std())
        oracle   = ORACLE_BEST[ds_name]
        gap      = oracle - mean_f1

        print(f"\n  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Oracle best: {oracle:.4f}")
        print(f"  Oracle gain: +{gap:.4f}")

        all_rows.append({
            "dataset":           ds_name,
            "algo":              algo,
            "default_param":     param,
            "n_seeds":           N_SEEDS,
            "default_F1_mean":   round(mean_f1, 4),
            "default_F1_std":    round(std_f1,  4),
            "oracle_F1":         oracle,
            "oracle_gain":       round(gap, 4),
        })

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'='*62}")
    print("  SPADE oracle-gain summary")
    print(f"{'─'*62}")
    print(f"  {'Dataset':<16} {'Default F1':>10} {'Oracle F1':>12} {'Gain':>8}")
    print(f"  {'─'*50}")
    for _, r in df.iterrows():
        print(f"  {r['dataset']:<16} "
              f"{r['default_F1_mean']:>8.4f}±{r['default_F1_std']:.3f}"
              f"{r['oracle_F1']:>12.4f}"
              f"{r['oracle_gain']:>+9.4f}")
    print(f"\n  Results saved: {RESULTS_CSV}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
