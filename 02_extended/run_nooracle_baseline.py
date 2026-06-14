"""
02_extended/run_nooracle_baseline.py
=================================
用固定默认参数运行 PhenoGraph、Leiden、HDBSCAN，
不依赖真实标签选参（即去掉 oracle 超参数选择）。

目的：
    当前论文中 K=auto 方法的结果是对扫描范围内所有参数组合
    选 ARI 最高者报告，属于 oracle 评估（需要地面真值）。
    本脚本提供"真实场景"下的性能基线：
    实践中用户不知道哪个参数最优，只能用默认参数。

默认参数选择依据：
    PhenoGraph : k=30（原始论文推荐）
    Leiden     : resolution=1.0（scanpy 默认）
    HDBSCAN    : min_cluster_size=50（稳健默认，适合多数细胞规模）

输出：
    results/nooracle_results.csv

论文用途：
    在附录或补充材料中，与 oracle 结果并列，
    展示"oracle 增益"= oracle_ARI - default_ARI。
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
import scanpy as sc
import anndata as ad
import phenograph

from sklearn.cluster import HDBSCAN

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────────────
RESULTS_CSV = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'results', 'diagnostics', 'nooracle_results.csv'
)

# Fixed default parameters (no scan, no ground-truth label usage)
DEFAULT_PARAMS = {
    'PhenoGraph': {'k': 30},
    'Leiden':     {'resolution': 1.0},
    'HDBSCAN':    {'min_cluster_size': 50},
}

# Mosmann_rare 太大（39万细胞），PhenoGraph 默认 k=30 约需 40 分钟
# 设置跳过标志，可按需改为 False
SKIP_PHENOGRAPH_MOSMANN = True


# ── 算法运行函数 ───────────────────────────────────────────────

def run_phenograph(X, k=30):
    communities, _, _ = phenograph.cluster(X, k=k, n_jobs=1)
    return np.array(communities)


def run_leiden(X, resolution=1.0):
    adata = ad.AnnData(X=X.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution, random_state=42)
    labels = adata.obs['leiden'].astype(int).values
    return labels


def run_hdbscan(X, min_cluster_size=50):
    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    # 噪声点（label=-1）单独视为一个簇，编号重映射为最大簇号+1
    if -1 in labels:
        noise_label = labels.max() + 1
        labels = np.where(labels == -1, noise_label, labels)
    return labels


# ── 主流程 ────────────────────────────────────────────────────

ALGORITHMS = {
    'PhenoGraph': run_phenograph,
    'Leiden':     run_leiden,
    'HDBSCAN':    run_hdbscan,
}


def main():
    print("=" * 65)
    print("K=auto algorithms: non-oracle baseline (fixed default parameters)")
    print("=" * 65)
    print("Default parameters:")
    for algo, params in DEFAULT_PARAMS.items():
        print(f"  {algo:<12} {params}")
    print()

    datasets = load_all(verbose=False)
    all_rows = []

    for algo_name, algo_fn in ALGORITHMS.items():
        params = DEFAULT_PARAMS[algo_name]
        param_key = list(params.keys())[0]
        param_val = list(params.values())[0]

        print(f"\n{'─' * 65}")
        print(f"  {algo_name}  ({param_key}={param_val}，固定不扫描)")
        print(f"{'─' * 65}")

        for ds_name in DATASETS:
            # PhenoGraph × Mosmann_rare 可选跳过
            if (algo_name == 'PhenoGraph'
                    and ds_name == 'Mosmann_rare'
                    and SKIP_PHENOGRAPH_MOSMANN):
                print(f"  [{algo_name} | {ds_name}]  SKIP (SKIP_PHENOGRAPH_MOSMANN=True)")
                row = {
                    'dataset'      : ds_name,
                    'algorithm'    : f'{algo_name}_default',
                    'K_mode'       : 'K=auto_default',
                    'default_param': f'{param_key}={param_val}',
                    'n_clusters_true': TRUE_K[ds_name],
                    'runtime_sec'  : None,
                    'ARI'          : None,
                    'MacroF1'      : None,
                    'n_clusters_found': None,
                    'note'         : 'skipped',
                }
                all_rows.append(row)
                continue

            X, y, markers, pop_names = datasets[ds_name]

            print(f"  [{algo_name} | {ds_name}]  cells={X.shape[0]:,} ...",
                  end='', flush=True)

            t0     = time.time()
            y_pred = algo_fn(X, **params)
            rt     = round(time.time() - t0, 1)

            metrics = evaluate(y, y_pred, dataset_name=ds_name)

            ari = metrics['ARI']
            f1  = metrics['MacroF1']
            k_found = metrics['n_clusters_found']
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"  ARI={ari:.4f}  MacroF1={f1_str}"
                  f"  K_found={k_found}  ({rt}s)")

            row = {
                'dataset'        : ds_name,
                'algorithm'      : f'{algo_name}_default',
                'K_mode'         : 'K=auto_default',
                'default_param'  : f'{param_key}={param_val}',
                'n_clusters_true': TRUE_K[ds_name],
                'runtime_sec'    : rt,
                'note'           : '',
            }
            for k2, v in metrics.items():
                if k2 not in ('per_pop_f1', 'per_pop_pr'):
                    row[k2] = v
            row['per_pop_f1'] = metrics.get('per_pop_f1', {})
            row['per_pop_pr'] = metrics.get('per_pop_pr', {})
            all_rows.append(row)

    save_results(all_rows, RESULTS_CSV)
    print(f"\nResults saved: {RESULTS_CSV}")

    # ── 打印 oracle 增益对比 ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  Oracle 增益摘要")
    print("  （oracle ARI 来自工作记录，default ARI 来自本次运行）")
    print("=" * 65)

    # 原始 oracle 结果（来自工作记录）
    ORACLE_ARI = {
        'PhenoGraph': {'Levine_13dim': 0.9385, 'Levine_32dim': 0.7071,
                       'Samusik_01': 0.9493, 'Nilsson_rare': None,
                       'Mosmann_rare': None},
        'Leiden':     {'Levine_13dim': 0.9395, 'Levine_32dim': 0.9697,
                       'Samusik_01': 0.9444, 'Nilsson_rare': None,
                       'Mosmann_rare': None},
        'HDBSCAN':    {'Levine_13dim': 0.4793, 'Levine_32dim': 0.5058,
                       'Samusik_01': 0.3829, 'Nilsson_rare': None,
                       'Mosmann_rare': None},
    }

    import pandas as pd
    df = pd.DataFrame(all_rows)

    for algo_name in ALGORITHMS:
        oracle_dict = ORACLE_ARI[algo_name]
        print(f"\n  {algo_name}")
        for ds_name in DATASETS:
            sub = df[(df['algorithm'] == f'{algo_name}_default') &
                     (df['dataset']   == ds_name)]
            if sub.empty or sub['ARI'].isna().all():
                continue
            default_ari = sub['ARI'].values[0]
            oracle_ari  = oracle_dict.get(ds_name)
            if oracle_ari is None or default_ari is None:
                continue
            gain = oracle_ari - default_ari
            flag = ' ← 增益显著' if abs(gain) > 0.05 else ''
            print(f"    {ds_name:<15}  "
                  f"oracle={oracle_ari:.4f}  "
                  f"default={default_ari:.4f}  "
                  f"gain={gain:+.4f}{flag}")

    print("\n" + "=" * 65)
    print("  完成")
    print("=" * 65)


if __name__ == '__main__':
    main()
