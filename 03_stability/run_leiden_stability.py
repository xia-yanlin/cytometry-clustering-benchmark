"""
03_stability/run_leiden_stability.py
=================================
对 Leiden 运行 N_RUNS 次（seed=0~9），
报告 ARI、Macro F1、K_found 的均值 ± 标准差及 95% CI，评估结果稳定性。

仅在三个多群体数据集上运行：
    Levine_13dim / Levine_32dim / Samusik_01

设计说明：
    固定 n_neighbors=15 和各数据集的 ARI-oracle 最优 resolution，
    只改变 seed，隔离 Leiden 算法本身的随机性。
    Leiden 的随机性来源：
        1. scanpy kNN 图构建（random_state）
        2. Leiden 社区检测本身（random_state）
    两处均由同一个 seed 控制。

    ★ 填写说明：
    运行前请从 leiden_results.csv 查出各数据集的 best_resolution，
    填入下方 BEST_RESOLUTION 字典。
    例：打开 CSV 找到每个数据集对应的 best_resolution 列的值。

崩溃恢复：
    每跑完一次立即追加写入 RAW_CSV，重启自动跳过已完成行。

输出：
    results/stability/leiden_stability_raw.csv
    results/stability/leiden_stability_summary.csv
"""

# ── 路径修复：从子目录 import 根目录模块 ──────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from scipy import stats as _stats

from data_loader import load_all, TRUE_K
from evaluation  import evaluate

# ── 配置 ──────────────────────────────────────────────────────
N_RUNS = 10
SEEDS  = list(range(N_RUNS))

MULTI_POP_DATASETS = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']

N_NEIGHBORS = 15   # 与基准脚本一致

# ★★★ 从 leiden_results.csv 的 best_resolution 列填入 ★★★
# 示例：打开 results/baseline/leiden_results.csv，
#        找到每行 dataset 对应的 best_resolution 值填入此处。
BEST_RESOLUTION = {
    'Levine_13dim': 1.0,
    'Levine_32dim': 0.3,
    'Samusik_01'  : 0.5,
}

# seed=42 单次参考值（来自 leiden_results.csv）
SEED42_REF = {
    'Levine_13dim': {'ARI': 0.9395, 'MacroF1': 0.4639, 'K_found': 15},
    'Levine_32dim': {'ARI': 0.9697, 'MacroF1': 0.5087, 'K_found': 11},
    'Samusik_01'  : {'ARI': 0.9444, 'MacroF1': 0.5530, 'K_found': 16},
}

ALGORITHM = 'Leiden'

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RAW_CSV     = _os.path.join(_ROOT, 'results', 'stability',
                            'leiden_stability_raw.csv')
SUMMARY_CSV = _os.path.join(_ROOT, 'results', 'stability',
                            'leiden_stability_summary.csv')


# ── 工具函数 ──────────────────────────────────────────────────

def ci95(arr):
    """95% 置信区间半宽（t 分布，df=n-1）。"""
    n = len(arr)
    if n < 2:
        return float('nan')
    return float(_stats.t.ppf(0.975, df=n - 1) * np.std(arr, ddof=1) / np.sqrt(n))


def run_leiden(X, resolution, seed, n_neighbors=N_NEIGHBORS):
    """
    构建 kNN 图并运行 Leiden 社区检测，返回 0-based 整数标签数组。
    与基准脚本 flow_gating_Leiden.py 的实现完全一致。
    """
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(X=X.copy())
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X',
                    random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    return adata.obs['leiden'].values.astype(int)


def load_completed(raw_csv):
    """读取已完成的 (dataset, seed) 集合，文件不存在则返回空集合。"""
    if not _os.path.exists(raw_csv):
        return set()
    try:
        df = pd.read_csv(raw_csv)
        return set(zip(df['dataset'], df['seed']))
    except Exception:
        return set()


def append_row(raw_csv, row: dict):
    """立即追加一行到 CSV（不存在则新建带表头的文件）。"""
    _os.makedirs(_os.path.dirname(raw_csv), exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not _os.path.exists(raw_csv)
    df_row.to_csv(raw_csv, mode='a', header=write_header, index=False)


def build_summary(raw_csv, summary_csv):
    """从 raw CSV 重新计算并写入 summary。"""
    df_raw = pd.read_csv(raw_csv)
    summary_rows = []

    for ds_name in MULTI_POP_DATASETS:
        sub = df_raw[df_raw['dataset'] == ds_name]
        if sub.empty:
            continue

        ari_v = sub['ARI'].dropna().values
        f1_v  = sub['MacroF1'].dropna().values
        kf_v  = sub['K_found'].dropna().values
        ref   = SEED42_REF[ds_name]

        summary_rows.append({
            'algorithm'             : ALGORITHM,
            'dataset'               : ds_name,
            'resolution_fixed'      : BEST_RESOLUTION[ds_name],
            'n_neighbors'           : N_NEIGHBORS,
            'n_runs'                : len(sub),
            # ARI
            'ARI_mean'              : round(ari_v.mean(), 4),
            'ARI_std'               : round(ari_v.std(),  4),
            'ARI_CI95'              : round(ci95(ari_v),  4),
            'ARI_min'               : round(ari_v.min(),  4),
            'ARI_max'               : round(ari_v.max(),  4),
            'ARI_seed42_ref'        : ref['ARI'],
            'ARI_mean_vs_seed42'    : round(ari_v.mean() - ref['ARI'], 4)
                                      if ref['ARI'] is not None else None,
            # Macro F1
            'MacroF1_mean'          : round(f1_v.mean(), 4),
            'MacroF1_std'           : round(f1_v.std(),  4),
            'MacroF1_CI95'          : round(ci95(f1_v),  4),
            'MacroF1_min'           : round(f1_v.min(),  4),
            'MacroF1_max'           : round(f1_v.max(),  4),
            'MacroF1_seed42_ref'    : ref['MacroF1'],
            # K_found
            'Kfound_mean'           : round(kf_v.mean(), 2),
            'Kfound_std'            : round(kf_v.std(),  2),
            'Kfound_min'            : int(kf_v.min()),
            'Kfound_max'            : int(kf_v.max()),
            'Kfound_seed42_ref'     : ref['K_found'],
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(summary_csv, index=False)
    return df_sum


# ── 启动前检查 ────────────────────────────────────────────────

def _check_config():
    missing = [ds for ds, v in BEST_RESOLUTION.items() if v is None]
    if missing:
        print("=" * 65)
        print("  ❌ 启动失败：请先填写 BEST_RESOLUTION 字典")
        print("     打开 results/baseline/leiden_results.csv，")
        print("     找到以下数据集的 best_resolution 列的值：")
        for ds in missing:
            print(f"       {ds}")
        print("  然后修改本脚本顶部的 BEST_RESOLUTION 和 SEED42_REF。")
        print("=" * 65)
        return False
    return True


# ── 主流程 ────────────────────────────────────────────────────

def main():
    if not _check_config():
        return

    completed  = load_completed(RAW_CSV)
    total_todo = len(MULTI_POP_DATASETS) * N_RUNS
    already_done = len(completed)

    print("=" * 65)
    print(f"Leiden stability assessment  |  {N_RUNS} runs  |  seeds={SEEDS}")
    print(f"n_neighbors={N_NEIGHBORS}  固定 resolution：{BEST_RESOLUTION}")
    if already_done:
        print(f"⚡ Resuming: {already_done}/{total_todo} complete, skipping done")
    print("=" * 65)

    datasets = load_all(verbose=False)

    for ds_name in MULTI_POP_DATASETS:
        X, y, markers, _ = datasets[ds_name]
        res = BEST_RESOLUTION[ds_name]
        ref = SEED42_REF[ds_name]

        pending_seeds = [s for s in SEEDS if (ds_name, s) not in completed]
        if not pending_seeds:
            print(f"\n[{ds_name}]  All {N_RUNS} runs complete, skipping.")
            continue

        ref_ari_str = f"{ref['ARI']}" if ref['ARI'] is not None else 'N/A'
        ref_f1_str  = f"{ref['MacroF1']}" if ref['MacroF1'] is not None else 'N/A'
        ref_k_str   = f"{ref['K_found']}" if ref['K_found'] is not None else 'N/A'

        print(f"\n{'─' * 65}")
        print(f"  {ds_name}  |  cells={X.shape[0]:,}  markers={X.shape[1]}"
              f"  resolution={res}  K_true={TRUE_K[ds_name]}")
        print(f"  seed=42 参考：ARI={ref_ari_str}  "
              f"MacroF1={ref_f1_str}  K_found={ref_k_str}")
        print(f"  待运行 seeds：{pending_seeds}")
        print(f"{'─' * 65}")

        for seed in pending_seeds:
            t0     = time.time()
            y_pred = run_leiden(X, resolution=res, seed=seed)
            rt     = round(time.time() - t0, 1)

            metrics = evaluate(y, y_pred, dataset_name=ds_name)
            ari     = metrics['ARI']
            f1      = metrics['MacroF1']
            k_found = metrics['n_clusters_found']

            row = {
                'algorithm'       : ALGORITHM,
                'dataset'         : ds_name,
                'resolution_fixed': res,
                'n_neighbors'     : N_NEIGHBORS,
                'seed'            : seed,
                'ARI'             : ari,
                'MacroF1'         : f1,
                'K_found'         : k_found,
                'runtime_sec'     : rt,
            }

            append_row(RAW_CSV, row)
            completed.add((ds_name, seed))

            print(f"  [{ALGORITHM} | {ds_name} | seed={seed:02d}]  "
                  f"ARI={ari:.4f}  MacroF1={f1:.4f}  "
                  f"K_found={k_found}  ({rt}s)  → 已写入")

    # ── 生成 summary ──────────────────────────────────────────
    print(f"\n原始数据：{RAW_CSV}")
    df_sum = build_summary(RAW_CSV, SUMMARY_CSV)
    print(f"Summary saved: {SUMMARY_CSV}")

    # ── 最终摘要打印 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Leiden stability summary (std > 0.02 flagged, with 95% CI)")
    print("=" * 65)

    for _, r in df_sum.iterrows():
        ari_flag = ' ⚠️' if r['ARI_std'] > 0.02 else ''
        f1_flag  = ' ⚠️' if r['MacroF1_std'] > 0.02 else ''
        delta_str = ''
        if r.get('ARI_mean_vs_seed42') is not None:
            delta_str = f"  Δseed42={r['ARI_mean_vs_seed42']:+.4f}"

        print(f"\n  {r['dataset']}  (resolution={r['resolution_fixed']})")
        print(f"    ARI      {r['ARI_mean']:.4f} ± {r['ARI_std']:.4f}"
              f"  [CI95 ±{r['ARI_CI95']:.4f}]"
              f"  [{r['ARI_min']:.4f}, {r['ARI_max']:.4f}]"
              f"{delta_str}{ari_flag}")
        print(f"    MacroF1  {r['MacroF1_mean']:.4f} ± {r['MacroF1_std']:.4f}"
              f"  [CI95 ±{r['MacroF1_CI95']:.4f}]{f1_flag}")
        print(f"    K_found  {r['Kfound_mean']:.1f} ± {r['Kfound_std']:.1f}"
              f"  [{r['Kfound_min']}, {r['Kfound_max']}]"
              f"  (seed42 ref={r['Kfound_seed42_ref']})")

    print("\n" + "=" * 65)
    print("  Done")
    print("=" * 65)


if __name__ == '__main__':
    main()
