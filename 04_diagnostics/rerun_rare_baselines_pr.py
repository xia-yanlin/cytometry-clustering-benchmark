"""
04_diagnostics/rerun_rare_baselines_pr.py
===========================
用更新后的 evaluation.py（含 Precision / Recall / F2）重跑
7 个基线Algorithm在两个稀有Dataset上的结果。

输出：
    results/diagnostics/rare_baselines_pr.csv
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

from data_loader import load_dataset, TRUE_K
from evaluation  import evaluate, print_summary

warnings.filterwarnings('ignore')

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RESULTS_CSV = _os.path.join(_ROOT, 'results', 'diagnostics', 'rare_baselines_pr.csv')
SPADE_SUM   = _os.path.join(_ROOT, 'results', 'extended',    'spade_summary.csv')

RUN_PHENOGRAPH_MOSMANN = False

BEST_PARAMS = {
    'PhenoGraph': {
        'Nilsson_rare': {'k': 15},
        'Mosmann_rare': {'k': 15},
    },
    'Leiden': {
        'Nilsson_rare': {'resolution': 0.3},
        'Mosmann_rare': {'resolution': 0.3},
    },
    'HDBSCAN': {
        'Nilsson_rare': {'min_cluster_size': 5},
        'Mosmann_rare': {'min_cluster_size': 50},
    },
}


def run_kmeans(X, k):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)


def run_gmm(X, k):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type='full',
                           n_init=5, random_state=42).fit_predict(X)


def run_ward(X, k):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)


def run_flowsom(X, marker_cols, k):
    import anndata as ad
    import flowsom as fs
    df    = pd.DataFrame(X, columns=marker_cols)
    adata = ad.AnnData(X=df)
    fsom  = fs.FlowSOM(adata, n_clusters=k, cols_to_use=marker_cols,
                       xdim=10, ydim=10, seed=42)
    raw   = np.array(fsom.get_cell_data().obs["metaclustering"])
    uniq  = sorted(set(raw))
    lmap  = {v: i for i, v in enumerate(uniq)}
    return np.array([lmap[v] for v in raw])


def run_phenograph(X, k):
    import phenograph
    labels, _, _ = phenograph.cluster(X, k=k, seed=42, n_jobs=1)
    return labels.astype(int)


def run_leiden(X, resolution):
    import scanpy as sc
    import anndata as ad
    adata = ad.AnnData(X=X.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution, random_state=42)
    return adata.obs['leiden'].astype(int).values


def run_hdbscan(X, min_cluster_size):
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import KDTree
    model  = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X).astype(int)
    noise  = labels == -1
    if noise.sum() > 0 and (~noise).sum() > 0:
        tree = KDTree(X[~noise])
        idx  = tree.query(X[noise], k=1, return_distance=False).flatten()
        labels[noise] = labels[~noise][idx]
    elif noise.sum() > 0:
        labels[:] = 0
    uniq = np.unique(labels)
    rmap = {v: i for i, v in enumerate(uniq)}
    return np.array([rmap[l] for l in labels])


def run_and_eval(algo_name, ds_name, X, y, markers, k_true,
                 skip=False, skip_reason=''):
    tag = f"[{algo_name} | {ds_name}]"
    if skip:
        print(f"  {tag}  ⚠️  SKIP ({skip_reason})")
        return {
            'dataset': ds_name, 'algorithm': algo_name,
            'K_mode': 'N/A', 'n_clusters_true': k_true,
            'runtime_sec': None, 'ARI': None,
            'MacroF1': None, 'accuracy': None,
            'n_clusters_found': None,
            'rare_target_f1': None,
            'rare_target_precision': None,
            'rare_target_recall': None,
            'rare_target_f2': None,
            'rare_target_found': None,
            'note': skip_reason,
        }

    print(f"  {tag}  Running...", end='', flush=True)
    t0 = time.time()

    try:
        if algo_name == 'k-means':
            y_pred = run_kmeans(X, k_true)
        elif algo_name == 'GMM':
            y_pred = run_gmm(X, k_true)
        elif algo_name == 'Ward':
            y_pred = run_ward(X, k_true)
        elif algo_name == 'FlowSOM':
            y_pred = run_flowsom(X, markers, k_true)
        elif algo_name == 'PhenoGraph':
            k = BEST_PARAMS['PhenoGraph'][ds_name]['k']
            y_pred = run_phenograph(X, k)
        elif algo_name == 'Leiden':
            res = BEST_PARAMS['Leiden'][ds_name]['resolution']
            y_pred = run_leiden(X, res)
        elif algo_name == 'HDBSCAN':
            mcs = BEST_PARAMS['HDBSCAN'][ds_name]['min_cluster_size']
            y_pred = run_hdbscan(X, mcs)
        else:
            raise ValueError(f"未知Algorithm: {algo_name}")
    except Exception as e:
        rt = round(time.time() - t0, 1)
        print(f" ❌ ERROR ({rt}s): {e}")
        return {'dataset': ds_name, 'algorithm': algo_name,
                'runtime_sec': rt, 'note': f'ERROR: {e}'}

    rt      = round(time.time() - t0, 1)
    metrics = evaluate(y, y_pred, dataset_name=ds_name)
    print_summary(metrics, ds_name, algo_name)

    return {
        'dataset'              : ds_name,
        'algorithm'            : algo_name,
        'K_mode'               : ('K=true' if algo_name in
                                   ('k-means','GMM','Ward','FlowSOM')
                                   else 'K=oracle'),
        'n_clusters_true'      : k_true,
        'runtime_sec'          : rt,
        'ARI'                  : metrics['ARI'],
        'MacroF1'              : metrics['MacroF1'],
        'accuracy'             : metrics['accuracy'],
        'n_clusters_found'     : metrics['n_clusters_found'],
        'rare_target_f1'       : metrics.get('rare_target_f1'),
        'rare_target_precision': metrics.get('rare_target_precision'),
        'rare_target_recall'   : metrics.get('rare_target_recall'),
        'rare_target_f2'       : metrics.get('rare_target_f2'),
        'rare_target_found'    : metrics.get('rare_target_found'),
    }


ALGORITHMS = ['k-means', 'GMM', 'Ward', 'FlowSOM',
              'PhenoGraph', 'Leiden', 'HDBSCAN']
DATASETS   = ['Nilsson_rare', 'Mosmann_rare']


def main():
    print("=" * 65)
    print("  Rare-dataset baseline re-run (with Precision / Recall / F2)")
    print("=" * 65)

    all_rows = []

    for ds_name in DATASETS:
        print(f"\n{'─'*65}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─'*65}")

        X, y, markers, _ = load_dataset(ds_name, verbose=True)
        k_true = TRUE_K[ds_name]
        n      = len(X)

        for algo in ALGORITHMS:
            skip, reason = False, ''
            if algo == 'Ward' and n > 60_000:
                skip, reason = True, f'cells={n:,} > 60,000 cell limit'
            elif (algo == 'PhenoGraph' and ds_name == 'Mosmann_rare'
                  and not RUN_PHENOGRAPH_MOSMANN):
                skip, reason = True, '~3h required; set RUN_PHENOGRAPH_MOSMANN=True to enable'

            row = run_and_eval(algo, ds_name, X, y, markers,
                               k_true, skip=skip, skip_reason=reason)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    _os.makedirs(_os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n\nResults saved: {RESULTS_CSV}  ({len(df)} rows)")

    print(f"\n{'═'*72}")
    print(f"  Rare-population detection full comparison (P / R / F2)")
    print(f"{'─'*72}")
    print(f"  {'Algorithm':<18} {'Dataset':<16} "
          f"{'F1':>7} {'P':>7} {'R':>7} {'F2':>7} {'Found':>5}")
    print(f"  {'─'*68}")

    for _, row in df.iterrows():
        if row.get('rare_target_f1') is None:
            continue
        f1   = row['rare_target_f1']
        prec = row.get('rare_target_precision', float('nan'))
        rec  = row.get('rare_target_recall',    float('nan'))
        f2   = row.get('rare_target_f2',        float('nan'))
        found = '✅' if row.get('rare_target_found') else '❌'
        print(f"  {row['algorithm']:<18} {row['dataset']:<16} "
              f"{f1:>7.4f} {prec:>7.4f} {rec:>7.4f} {f2:>7.4f} {found:>5}")

    # SPADE comparison
    try:
        spade = pd.read_csv(SPADE_SUM)
        print(f"  {'─'*68}")
        for _, row in spade.iterrows():
            ds = row.get('dataset', '')
            if ds not in ('Nilsson_rare', 'Mosmann_rare'):
                continue
            f1   = row.get('rare_target_f1',       float('nan'))
            prec = row.get('rare_target_precision', float('nan'))
            rec  = row.get('rare_target_recall',    float('nan'))
            f2   = row.get('rare_target_f2',        float('nan'))
            found = '✅' if (not np.isnan(f1) and f1 > 0.1) else '❌'
            print(f"  {'SPADE (best)':<18} {ds:<16} "
                  f"{f1:>7.4f} {prec:>7.4f} {rec:>7.4f} {f2:>7.4f} {found:>5}")
    except FileNotFoundError:
        print(f"  （SPADE summary not found: {SPADE_SUM})")

    print(f"{'═'*72}")


if __name__ == '__main__':
    main()