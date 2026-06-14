"""
03_stability/run_spade_nilsson.py
=================================
Validate stability of SPADE optimal configuration on Nilsson_rare.

Optimal configuration: k-means K=50, target_pct=5%, outlier_pctile=1%
seed 0~9，共 10 次。每次约 5 秒，总计 < 1 分钟。

输出：
    results/stability/nilsson_stability_raw.csv
    results/stability/nilsson_stability_summary.csv
"""
# ── 路径修复：从子目录 import 根目录模块 ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.optimize import linear_sum_assignment
from scipy import stats as _stats

from data_loader import load_dataset

warnings.filterwarnings('ignore')

DATASET        = 'Nilsson_rare'
TARGET_POP     = 'HSCs'
TARGET_PCT     = 0.05
OUTLIER_PCTILE = 1
K_KMEANS       = 50
N_SEEDS        = 10

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RESULTS_RAW = _os.path.join(_ROOT, 'results', 'stability', 'nilsson_stability_raw.csv')
RESULTS_SUM = _os.path.join(_ROOT, 'results', 'stability', 'nilsson_stability_summary.csv')


def ci95(arr):
    n = len(arr)
    if n < 2:
        return float('nan')
    return float(_stats.t.ppf(0.975, df=n-1) * np.std(arr, ddof=1) / np.sqrt(n))


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
    probs[valid] = np.minimum(1.0, ((lo + hi) / 2.0) / d_valid)
    return probs


def upsample_kdtree(X_all, X_kept, labels_kept, keep_mask):
    y_all = np.full(len(X_all), -1, dtype=int)
    y_all[keep_mask] = labels_kept
    discard_idx = np.where(~keep_mask)[0]
    if len(discard_idx) == 0:
        return y_all
    tree   = KDTree(X_kept)
    nn_idx = tree.query(X_all[discard_idx], k=1,
                        return_distance=False).flatten()
    y_all[discard_idx] = labels_kept[nn_idx]
    return y_all


def hungarian_align(y_true, y_pred):
    tl = np.unique(y_true)
    pc = np.unique(y_pred)
    size = max(len(tl), len(pc))
    cost = np.zeros((size, size), dtype=np.int64)
    for i, t in enumerate(tl):
        for j, p in enumerate(pc):
            cost[i, j] = -np.sum((y_true == t) & (y_pred == p))
    ri, ci = linear_sum_assignment(cost)
    mapping = {pc[c]: tl[r] for r, c in zip(ri, ci)
               if r < len(tl) and c < len(pc)}
    return np.array([mapping.get(p, '__unmatched__') for p in y_pred])


def prf(y_true, y_aligned, target):
    tp = np.sum((y_true == target) & (y_aligned == target))
    fp = np.sum((y_true != target) & (y_aligned == target))
    fn = np.sum((y_true == target) & (y_aligned != target))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r)   if (p+r)   > 0 else 0.0
    f2 = 5*p*r / (4*p+r) if (4*p+r) > 0 else 0.0
    return round(p,4), round(r,4), round(f1,4), round(f2,4)


def run_once(X, y_true, seed):
    t0  = time.time()
    rng = np.random.default_rng(seed)

    density = estimate_density(X, k=15)
    probs   = compute_retention_prob(density,
                                     outlier_pctile=OUTLIER_PCTILE,
                                     target_pct=TARGET_PCT)
    keep        = rng.random(len(probs)) < probs
    X_kept      = X[keep]
    labels_kept = KMeans(n_clusters=K_KMEANS, n_init=5,
                         random_state=seed).fit_predict(X_kept)

    y_pred    = upsample_kdtree(X, X_kept, labels_kept, keep)
    y_aligned = hungarian_align(y_true, y_pred)
    p, r, f1, f2 = prf(y_true, y_aligned, TARGET_POP)

    n_target_kept = (y_true[keep] == TARGET_POP).sum()
    return {
        'seed'           : seed,
        'n_clusters_found': len(np.unique(y_pred)),
        'n_target_kept'  : int(n_target_kept),
        'pct_target_kept': round(100*n_target_kept/(y_true==TARGET_POP).sum(), 1),
        'HSCs_F1'        : f1,
        'HSCs_Precision' : p,
        'HSCs_Recall'    : r,
        'HSCs_F2'        : f2,
        'found'          : f1 > 0.1,
        'runtime_sec'    : round(time.time()-t0, 1),
    }


def main():
    print("=" * 58)
    print(f"  Nilsson_rare stability validation  |  {N_SEEDS} repeated runs")
    print(f"  参数：k-means K={K_KMEANS}  tgt={TARGET_PCT:.0%}  "
          f"out={OUTLIER_PCTILE}%  seed=0~{N_SEEDS-1}")
    print("=" * 58)

    X, y, _, _ = load_dataset(DATASET, verbose=False)
    n_target   = (y == TARGET_POP).sum()
    print(f"  cells={len(X):,}  {TARGET_POP}={n_target}  "
          f"({100*n_target/len(X):.3f}%)\n")

    rows = []
    for seed in range(N_SEEDS):
        row = run_once(X, y, seed)
        rows.append(row)
        print(f"  [seed={seed}]  "
              f"F1={row['HSCs_F1']:.4f}  P={row['HSCs_Precision']:.4f}  "
              f"R={row['HSCs_Recall']:.4f}  F2={row['HSCs_F2']:.4f}  "
              f"target_kept={row['n_target_kept']}/{n_target} "
              f"({row['pct_target_kept']}%)  ({row['runtime_sec']}s)")

    df = pd.DataFrame(rows)
    _os.makedirs(_os.path.dirname(RESULTS_RAW), exist_ok=True)
    df.to_csv(RESULTS_RAW, index=False)

    print(f"\n{'═'*58}")
    print(f"  Stability summary (with 95% CI, n={N_SEEDS})")
    print(f"{'─'*58}")
    summary_rows = []
    for col in ['HSCs_F1', 'HSCs_Precision', 'HSCs_Recall', 'HSCs_F2']:
        v    = df[col].values
        mean = v.mean()
        std  = v.std()
        ci   = ci95(v)
        print(f"  {col:<20}  {mean:.4f} ± {std:.4f}  "
              f"[CI95 ±{ci:.4f}]  [{v.min():.4f}, {v.max():.4f}]")
        summary_rows.append({'metric': col, 'mean': round(mean,4),
                              'std': round(std,4), 'CI95': round(ci,4),
                              'min': round(v.min(),4), 'max': round(v.max(),4)})

    found = df['found'].sum()
    std_f1 = df['HSCs_F1'].std()
    print(f"\n  Detected (F1>0.1): {found}/{N_SEEDS}")
    print(f"  {'Stable' if std_f1 <= 0.05 else 'UNSTABLE'}  (F1 std={std_f1:.4f})")

    pd.DataFrame(summary_rows).to_csv(RESULTS_SUM, index=False)
    print(f"\n  {RESULTS_RAW}")
    print(f"  {RESULTS_SUM}")
    print(f"{'═'*58}")


if __name__ == '__main__':
    main()