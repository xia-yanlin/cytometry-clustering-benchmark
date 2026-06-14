"""
03_stability/run_stochastic_multirun.py
===================================
对三个随机初始化算法（k-means、GMM、FlowSOM）各运行 N_RUNS 次，
报告 ARI 和 Macro F1 的均值 ± 标准差及 95% CI，评估结果稳定性。

输出：
    results/stability/multirun_raw.csv      每次单独运行的原始结果
    results/stability/multirun_summary.csv  均值 ± std ± 95%CI 汇总

运行时间估算：
    k-means  × 30 次 ≈ 30 分钟
    GMM      × 30 次 ≈ 90 分钟
    FlowSOM  × 30 次 ≈ 30 分钟
    总计约 150 分钟
"""
# ── 路径修复：从子目录 import 根目录模块 ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────

import time
import numpy as np
import pandas as pd
import anndata as ad
import flowsom as fs
from scipy import stats as _stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate

# ── 配置 ──────────────────────────────────────────────────────
N_RUNS = 30
SEEDS  = list(range(N_RUNS))

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RAW_CSV     = _os.path.join(_ROOT, 'results', 'stability', 'multirun_raw.csv')
SUMMARY_CSV = _os.path.join(_ROOT, 'results', 'stability', 'multirun_summary.csv')


# ── 各算法运行函数 ─────────────────────────────────────────────

def run_kmeans(X, marker_cols, n_clusters, seed):
    model = KMeans(n_clusters=n_clusters, n_init=20,
                   max_iter=300, random_state=seed)
    return model.fit_predict(X)


def run_gmm(X, marker_cols, n_clusters, seed):
    model = GaussianMixture(n_components=n_clusters,
                            covariance_type='full',
                            n_init=5, max_iter=200, random_state=seed)
    model.fit(X)
    return model.predict(X)


def run_flowsom(X, marker_cols, n_clusters, seed):
    df_tmp = pd.DataFrame(X, columns=marker_cols)
    adata  = ad.AnnData(X=df_tmp)
    fsom   = fs.FlowSOM(adata, n_clusters=n_clusters,
                        cols_to_use=marker_cols,
                        xdim=10, ydim=10, seed=seed)
    raw_labels = np.array(fsom.get_cell_data().obs["metaclustering"])
    unique     = sorted(set(raw_labels))
    label_map  = {v: i for i, v in enumerate(unique)}
    return np.array([label_map[v] for v in raw_labels])


ALGORITHMS = {
    'k-means': run_kmeans,
    'GMM'    : run_gmm,
    'FlowSOM': run_flowsom,
}


def ci95(arr):
    """95% 置信区间半宽（t分布，df=n-1）。"""
    n = len(arr)
    if n < 2:
        return float('nan')
    return float(_stats.t.ppf(0.975, df=n-1) * np.std(arr, ddof=1) / np.sqrt(n))


def main():
    print("=" * 65)
    print(f"Stochastic algorithm stability  |  {N_RUNS} runs each  |  seeds={SEEDS}")
    print("=" * 65)

    datasets = load_all(verbose=False)
    raw_rows = []

    for algo_name, algo_fn in ALGORITHMS.items():
        print(f"\n{'─' * 65}")
        print(f"  {algo_name}")
        print(f"{'─' * 65}")

        for ds_name in DATASETS:
            X, y, markers, pop_names = datasets[ds_name]
            k = TRUE_K[ds_name]

            ari_list = []
            f1_list  = []

            for seed in SEEDS:
                t0     = time.time()
                y_pred = algo_fn(X, markers, n_clusters=k, seed=seed)
                rt     = round(time.time() - t0, 1)

                metrics = evaluate(y, y_pred, dataset_name=ds_name)
                ari = metrics['ARI']
                f1  = metrics['MacroF1']

                ari_list.append(ari)
                if f1 is not None:
                    f1_list.append(f1)

                row = {
                    'algorithm'  : algo_name,
                    'dataset'    : ds_name,
                    'seed'       : seed,
                    'ARI'        : ari,
                    'MacroF1'    : f1,
                    'n_clusters' : metrics['n_clusters_found'],
                    'runtime_sec': rt,
                }
                if ds_name in ('Nilsson_rare', 'Mosmann_rare'):
                    row['rare_target_f1']    = metrics.get('rare_target_f1')
                    row['rare_target_found'] = metrics.get('rare_target_found')

                raw_rows.append(row)

                f1_str = f"{f1:.4f}" if f1 is not None else "N/A "
                print(f"  [{algo_name} | {ds_name} | seed={seed:02d}]  "
                      f"ARI={ari:.4f}  MacroF1={f1_str}  ({rt}s)")

            ari_m, ari_s = np.mean(ari_list), np.std(ari_list)
            flag = " ⚠️" if ari_s > 0.02 else ""
            print(f"  → ARI  {ari_m:.4f} ± {ari_s:.4f}"
                  f"  [{min(ari_list):.4f}, {max(ari_list):.4f}]{flag}")
            if f1_list:
                f1_m, f1_s = np.mean(f1_list), np.std(f1_list)
                flag2 = " ⚠️" if f1_s > 0.02 else ""
                print(f"  → F1   {f1_m:.4f} ± {f1_s:.4f}{flag2}")

    df_raw = pd.DataFrame(raw_rows)
    _os.makedirs(_os.path.dirname(RAW_CSV), exist_ok=True)
    df_raw.to_csv(RAW_CSV, index=False)
    print(f"\n原始数据已保存：{RAW_CSV}  ({len(df_raw)} 行)")

    # ── 汇总统计（含 95% CI）─────────────────────────────────
    summary_rows = []
    for algo_name in ALGORITHMS:
        for ds_name in DATASETS:
            sub = df_raw[(df_raw['algorithm'] == algo_name) &
                         (df_raw['dataset']   == ds_name)]
            if sub.empty:
                continue

            ari_v = sub['ARI'].dropna().values
            f1_v  = sub['MacroF1'].dropna().values

            row = {
                'algorithm'   : algo_name,
                'dataset'     : ds_name,
                'n_runs'      : len(sub),
                'ARI_mean'    : round(ari_v.mean(), 4),
                'ARI_std'     : round(ari_v.std(),  4),
                'ARI_CI95'    : round(ci95(ari_v),  4),
                'ARI_min'     : round(ari_v.min(),  4),
                'ARI_max'     : round(ari_v.max(),  4),
                'MacroF1_mean': round(f1_v.mean(), 4) if len(f1_v) else None,
                'MacroF1_std' : round(f1_v.std(),  4) if len(f1_v) else None,
                'MacroF1_CI95': round(ci95(f1_v),  4) if len(f1_v) else None,
                'MacroF1_min' : round(f1_v.min(),  4) if len(f1_v) else None,
                'MacroF1_max' : round(f1_v.max(),  4) if len(f1_v) else None,
            }

            if 'rare_target_f1' in sub.columns:
                rf1 = sub['rare_target_f1'].dropna().values
                if len(rf1):
                    row['rare_f1_mean'] = round(rf1.mean(), 4)
                    row['rare_f1_std']  = round(rf1.std(),  4)
                    row['rare_f1_CI95'] = round(ci95(rf1),  4)

            summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary saved: {SUMMARY_CSV}")

    print("\n" + "=" * 65)
    print("  稳定性摘要（std > 0.02 标记 ⚠️，含 95% CI）")
    print("=" * 65)
    for algo_name in ALGORITHMS:
        print(f"\n  {algo_name}")
        sub = df_sum[df_sum['algorithm'] == algo_name]
        for _, r in sub.iterrows():
            a_flag = ' ⚠️' if r['ARI_std'] > 0.02 else ''
            mf1_std = r.get('MacroF1_std')
            f_flag  = ' ⚠️' if (mf1_std is not None and
                                 not pd.isna(mf1_std) and
                                 mf1_std > 0.02) else ''
            mf1_str = (f"{r['MacroF1_mean']:.4f}±{mf1_std:.4f}"
                       f" [CI95±{r['MacroF1_CI95']:.4f}]"
                       if (mf1_std is not None and not pd.isna(mf1_std))
                       else 'N/A')
            print(f"    {r['dataset']:<15}  "
                  f"ARI={r['ARI_mean']:.4f}±{r['ARI_std']:.4f}"
                  f" [CI95±{r['ARI_CI95']:.4f}]{a_flag}  "
                  f"MacroF1={mf1_str}{f_flag}")

    print("\n" + "=" * 65)
    print("  完成")
    print("=" * 65)


if __name__ == '__main__':
    main()