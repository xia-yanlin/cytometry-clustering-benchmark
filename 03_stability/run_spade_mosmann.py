"""
03_stability/run_spade_mosmann.py
================================
Validate stability of SPADE optimal configuration on Mosmann_rare.

Optimal configuration: HDBSCAN mcs=3, target_pct=5%, outlier_pctile=1%
seed 0~9，共 10 次。

输出：
    results/stability/mosmann_stability_raw.csv
    results/stability/mosmann_stability_summary.csv
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
from sklearn.neighbors import KDTree
from sklearn.cluster import HDBSCAN
from scipy.optimize import linear_sum_assignment
from scipy import stats as _stats

from data_loader import load_dataset
from evaluation  import evaluate, RARE_TARGET

warnings.filterwarnings('ignore')

TARGET_PCT     = 0.05
OUTLIER_PCTILE = 1
MCS            = 3
N_SEEDS        = 10
DATASET        = 'Mosmann_rare'
TARGET_POP     = 'activated'

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RESULTS_RAW = _os.path.join(_ROOT, 'results', 'stability', 'mosmann_stability_raw.csv')
RESULTS_SUM = _os.path.join(_ROOT, 'results', 'stability', 'mosmann_stability_summary.csv')


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
        if p_mean < target_pct:
            lo = mid
        else:
            hi = mid
    probs        = np.zeros(n)
    probs[valid] = np.minimum(1.0, ((lo + hi) / 2.0) / d_valid)
    return probs


def upsample_kdtree(X_all, X_kept, labels_kept, keep_mask, chunk_size=50_000):
    y_all = np.full(len(X_all), -1, dtype=int)
    y_all[keep_mask] = labels_kept
    discard_idx = np.where(~keep_mask)[0]
    if len(discard_idx) == 0:
        return y_all
    tree = KDTree(X_kept)
    for start in range(0, len(discard_idx), chunk_size):
        end      = min(start + chunk_size, len(discard_idx))
        idx_chunk = discard_idx[start:end]
        nn_idx   = tree.query(X_all[idx_chunk], k=1,
                              return_distance=False).flatten()
        y_all[idx_chunk] = labels_kept[nn_idx]
    return y_all


def precision_recall_f(y_true, y_pred_aligned, target):
    tp = np.sum((y_true == target) & (y_pred_aligned == target))
    fp = np.sum((y_true != target) & (y_pred_aligned == target))
    fn = np.sum((y_true == target) & (y_pred_aligned != target))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r)   if (p+r)   > 0 else 0.0
    f2 = 5*p*r / (4*p+r) if (4*p+r) > 0 else 0.0
    return p, r, f1, f2


def hungarian_align(y_true, y_pred):
    true_labels   = np.unique(y_true)
    pred_clusters = np.unique(y_pred)
    n_true, n_pred = len(true_labels), len(pred_clusters)
    size = max(n_true, n_pred)
    cost = np.zeros((size, size), dtype=np.int64)
    for i, tl in enumerate(true_labels):
        for j, pc in enumerate(pred_clusters):
            cost[i, j] = -np.sum((y_true == tl) & (y_pred == pc))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_true and c < n_pred:
            mapping[pred_clusters[c]] = true_labels[r]
    return np.array([mapping.get(p, '__unmatched__') for p in y_pred])


def run_once(X, y_true, seed):
    t0  = time.time()
    rng = np.random.default_rng(seed)

    density = estimate_density(X, k=15)
    probs   = compute_retention_prob(density,
                                     outlier_pctile=OUTLIER_PCTILE,
                                     target_pct=TARGET_PCT)
    keep   = rng.random(len(probs)) < probs
    X_kept = X[keep]

    model  = HDBSCAN(min_cluster_size=MCS,
                     min_samples=max(1, MCS // 2))
    labels = model.fit_predict(X_kept).astype(int)

    noise = labels == -1
    if noise.sum() > 0:
        non_noise = ~noise
        if non_noise.sum() > 0:
            tree = KDTree(X_kept[non_noise])
            idx  = tree.query(X_kept[noise], k=1,
                              return_distance=False).flatten()
            labels[noise] = labels[non_noise][idx]
        else:
            labels[:] = 0

    unique = np.unique(labels)
    remap  = {v: i for i, v in enumerate(unique)}
    labels_kept = np.array([remap[l] for l in labels])
    n_sub = len(np.unique(labels_kept))

    y_pred    = upsample_kdtree(X, X_kept, labels_kept, keep)
    metrics   = evaluate(y_true, y_pred, dataset_name=DATASET)
    y_aligned = hungarian_align(y_true, y_pred)

    tp = np.sum((y_true == TARGET_POP) & (y_aligned == TARGET_POP))
    fp = np.sum((y_true != TARGET_POP) & (y_aligned == TARGET_POP))
    fn = np.sum((y_true == TARGET_POP) & (y_aligned != TARGET_POP))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r)   if (p+r)   > 0 else 0.0
    f2 = 5*p*r / (4*p+r) if (4*p+r) > 0 else 0.0

    rt = round(time.time() - t0, 1)
    n_target_kept  = (y_true[keep] == TARGET_POP).sum()
    n_target_total = (y_true == TARGET_POP).sum()

    return {
        'seed'               : seed,
        'n_sub_clusters'     : n_sub,
        'n_clusters_found'   : metrics['n_clusters_found'],
        'n_target_kept'      : int(n_target_kept),
        'n_target_total'     : int(n_target_total),
        'pct_target_kept'    : round(100 * n_target_kept / max(n_target_total, 1), 1),
        'ARI'                : metrics['ARI'],
        'activated_F1'       : round(f1, 4),
        'activated_Precision': round(p,  4),
        'activated_Recall'   : round(r,  4),
        'activated_F2'       : round(f2, 4),
        'rare_target_found'  : bool(f1 > 0.1),
        'runtime_sec'        : rt,
    }


def main():
    print("=" * 62)
    print(f"  Mosmann_rare stability validation  |  {N_SEEDS} repeated runs")
    print(f"  参数：HDBSCAN mcs={MCS}  tgt={TARGET_PCT:.0%}  "
          f"out={OUTLIER_PCTILE}%  seed=0~{N_SEEDS-1}")
    print("=" * 62)

    print(f"\nLoading dataset...", end='', flush=True)
    X, y, _, _ = load_dataset(DATASET, verbose=False)
    n_target   = (y == TARGET_POP).sum()
    print(f" done  cells={len(X):,}  {TARGET_POP}={n_target}  "
          f"({100*n_target/len(X):.3f}%)")

    rows = []
    for seed in range(N_SEEDS):
        print(f"\n  [seed={seed}]", end='  ', flush=True)
        row = run_once(X, y, seed)
        rows.append(row)
        print(f"F1={row['activated_F1']:.4f}  "
              f"P={row['activated_Precision']:.4f}  "
              f"R={row['activated_Recall']:.4f}  "
              f"F2={row['activated_F2']:.4f}  "
              f"K={row['n_clusters_found']}  "
              f"target_kept={row['n_target_kept']}/{row['n_target_total']} "
              f"({row['pct_target_kept']}%)  "
              f"({row['runtime_sec']}s)")

    df = pd.DataFrame(rows)
    _os.makedirs(_os.path.dirname(RESULTS_RAW), exist_ok=True)
    df.to_csv(RESULTS_RAW, index=False)

    print(f"\n{'═'*62}")
    print(f"  Stability summary (with 95% CI, n={N_SEEDS})")
    print(f"{'─'*62}")

    metrics_cols = ['activated_F1', 'activated_Precision',
                    'activated_Recall', 'activated_F2', 'ARI']
    summary_rows = []
    for col in metrics_cols:
        vals = df[col].values
        mean = vals.mean()
        std  = vals.std()
        ci   = ci95(vals)
        print(f"  {col:<24}  {mean:.4f} ± {std:.4f}  "
              f"[CI95 ±{ci:.4f}]  [{vals.min():.4f}, {vals.max():.4f}]")
        summary_rows.append({'metric': col, 'mean': round(mean,4),
                              'std': round(std,4), 'CI95': round(ci,4),
                              'min': round(vals.min(),4),
                              'max': round(vals.max(),4)})

    found_count = df['rare_target_found'].sum()
    print(f"\n  Detection count (F1>0.1): {found_count}/{N_SEEDS}")

    f1_std = df['activated_F1'].std()
    if f1_std > 0.05:
        print(f"  WARNING: F1 std={f1_std:.4f} > 0.05, possible bimodal convergence")
    else:
        print(f"  Stable: F1 std={f1_std:.4f} <= 0.05")

    pd.DataFrame(summary_rows).to_csv(RESULTS_SUM, index=False)

    print(f"\n  {RESULTS_RAW}")
    print(f"  {RESULTS_SUM}")
    print(f"{'═'*62}")


if __name__ == '__main__':
    main()