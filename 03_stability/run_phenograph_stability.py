"""
03_stability/run_phenograph_stability.py
=====================================
对 PhenoGraph 运行 N_RUNS 次（seed=0~9），
报告 ARI、Macro F1、K_found 的均值 ± 标准差及 95% CI，评估结果稳定性。

仅在三个多群体数据集上运行（稀有群体数据集 ARI≈0，无参数区分能力）：
    Levine_13dim / Levine_32dim / Samusik_01

设计说明：
    各数据集固定使用 seed=42 跑出的最优 k（ARI-oracle），
    只改变 seed，目的是隔离 PhenoGraph 内部随机性（Louvain 社区检测）
    与参数选择 variance 的贡献，得到对算法本身稳定性的干净估计。
    如需同时评估参数选择的不确定性，应另建全扫描版本。

    背景：seed=42 下 Levine_32dim ARI 从旧无 seed 结果的 0.7071
    降至 0.6362（差距 0.071），本实验量化该差距的统计意义。

崩溃恢复：
    每跑完一次（一个 dataset × 一个 seed）就立即追加写入 RAW_CSV。
    重新启动时自动跳过已完成的行，从断点继续。
    全部完成后单独生成 SUMMARY_CSV。

输出：
    results/stability/phenograph_stability_raw.csv      每次单独运行的原始结果
    results/stability/phenograph_stability_summary.csv  均值 ± std ± 95%CI 汇总

运行时间估算（n_jobs=1，Windows 安全模式）：
    Levine_13dim  ≈ 5-10 min × 10 次
    Levine_32dim  ≈ 10-20 min × 10 次
    Samusik_01    ≈ 5-10 min × 10 次
    总计约 3-5 小时

Windows multiprocessing 说明：
    PhenoGraph 内部即使设置 n_jobs=1，某些版本仍会在 kNN 构建阶段
    尝试 spawn 子进程。本脚本通过以下方式规避：
    1. multiprocessing.freeze_support()
    2. 环境变量强制单线程（OMP / MKL / OpenBLAS）
    3. if __name__ == '__main__': 保护
    必须从文件路径直接运行（python path/to/script.py），
    不能在 Jupyter cell 或交互式 REPL 中运行。
"""

# ── multiprocessing 保护（必须在所有其他 import 之前）──────────
import multiprocessing
multiprocessing.freeze_support()

# ── 强制单线程，抑制 phenograph 内部的并行尝试 ─────────────────
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS',   '1')
_os.environ.setdefault('MKL_NUM_THREADS',   '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

# ── 路径修复：从子目录 import 根目录模块 ──────────────────────
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ─────────────────────────────────────────────────────────────

import time
import numpy as np
import pandas as pd
from scipy import stats as _stats

from data_loader import load_all, TRUE_K
from evaluation  import evaluate

# ── 配置 ──────────────────────────────────────────────────────
N_RUNS = 10
SEEDS  = list(range(N_RUNS))

# 仅多群体数据集（稀有数据集 ARI 在所有参数下均≈0，不适合稳定性实验）
MULTI_POP_DATASETS = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']

# 各数据集固定 k：来自 seed=42 的 ARI-oracle 扫描结果（见实验记录第三节）
# Levine_13dim: best_k=45  ARI_ref=0.9361
# Levine_32dim: best_k=60  ARI_ref=0.6362  ← 稳定性核心关注点
# Samusik_01:   best_k=45  ARI_ref=0.9541
BEST_K = {
    'Levine_13dim': 45,
    'Levine_32dim': 60,
    'Samusik_01'  : 45,
}

# seed=42 单次参考值（用于 summary 对比，验证多次均值是否偏离单次结果）
SEED42_REF = {
    'Levine_13dim': {'ARI': 0.9361, 'MacroF1': 0.4804, 'K_found': 15},
    'Levine_32dim': {'ARI': 0.6362, 'MacroF1': 0.5722, 'K_found': 20},
    'Samusik_01'  : {'ARI': 0.9541, 'MacroF1': 0.7359, 'K_found': 21},
}

ALGORITHM = 'PhenoGraph'

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RAW_CSV     = _os.path.join(_ROOT, 'results', 'stability',
                            'phenograph_stability_raw.csv')
SUMMARY_CSV = _os.path.join(_ROOT, 'results', 'stability',
                            'phenograph_stability_summary.csv')


# ── 工具函数 ──────────────────────────────────────────────────

def ci95(arr):
    """95% 置信区间半宽（t 分布，df=n-1）。与 stochastic_multirun.py 相同。"""
    n = len(arr)
    if n < 2:
        return float('nan')
    return float(_stats.t.ppf(0.975, df=n - 1) * np.std(arr, ddof=1) / np.sqrt(n))


def run_phenograph(X, k, seed):
    """
    运行 PhenoGraph，返回预测簇编号数组（0-based）。

    n_jobs=1：Windows 下尽量避免 multiprocessing 嵌套死锁。
    -1 噪声点（若有）统一映射到独立簇。
    """
    import phenograph
    labels, graph, q = phenograph.cluster(X, k=k, seed=seed, n_jobs=1)
    if -1 in labels:
        labels = np.where(labels == -1, labels.max() + 1, labels)
    return labels.astype(int)


def load_completed(raw_csv):
    """
    读取已完成的运行记录，返回 (dataset, seed) 集合。
    文件不存在时返回空集合。
    """
    if not _os.path.exists(raw_csv):
        return set()
    try:
        df = pd.read_csv(raw_csv)
        return set(zip(df['dataset'], df['seed']))
    except Exception:
        return set()


def append_row(raw_csv, row: dict):
    """
    将单行结果立即追加到 CSV（文件不存在则新建带表头的文件）。
    每次写一行，崩溃后已完成的行不丢失。
    """
    _os.makedirs(_os.path.dirname(raw_csv), exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not _os.path.exists(raw_csv)
    df_row.to_csv(raw_csv, mode='a', header=write_header, index=False)


def build_summary(raw_csv, summary_csv):
    """从 raw CSV 重新计算并写入 summary（含 95% CI 和 seed=42 对比）。"""
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
            'algorithm'            : ALGORITHM,
            'dataset'              : ds_name,
            'k_fixed'              : BEST_K[ds_name],
            'n_runs'               : len(sub),
            # ARI
            'ARI_mean'             : round(ari_v.mean(), 4),
            'ARI_std'              : round(ari_v.std(),  4),
            'ARI_CI95'             : round(ci95(ari_v),  4),
            'ARI_min'              : round(ari_v.min(),  4),
            'ARI_max'              : round(ari_v.max(),  4),
            # seed=42 单次参考及偏差（核心诊断列）
            'ARI_seed42_ref'       : ref['ARI'],
            'ARI_mean_vs_seed42'   : round(ari_v.mean() - ref['ARI'], 4),
            # Macro F1
            'MacroF1_mean'         : round(f1_v.mean(), 4),
            'MacroF1_std'          : round(f1_v.std(),  4),
            'MacroF1_CI95'         : round(ci95(f1_v),  4),
            'MacroF1_min'          : round(f1_v.min(),  4),
            'MacroF1_max'          : round(f1_v.max(),  4),
            'MacroF1_seed42_ref'   : ref['MacroF1'],
            # K_found
            'Kfound_mean'          : round(kf_v.mean(), 2),
            'Kfound_std'           : round(kf_v.std(),  2),
            'Kfound_min'           : int(kf_v.min()),
            'Kfound_max'           : int(kf_v.max()),
            'Kfound_seed42_ref'    : ref['K_found'],
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(summary_csv, index=False)
    return df_sum


# ── 主流程 ────────────────────────────────────────────────────

def main():
    # 读取已完成记录，支持断点续跑
    completed = load_completed(RAW_CSV)
    total_todo = len(MULTI_POP_DATASETS) * N_RUNS
    already_done = len(completed)

    print("=" * 65)
    print(f"PhenoGraph stability assessment  |  {N_RUNS} runs  |  seeds={SEEDS}")
    print(f"固定 k（ARI-oracle，seed=42）：{BEST_K}")
    print(f"目的：隔离 Louvain 随机性，量化 Levine_32dim ARI 波动（0.071）")
    if already_done:
        print(f"⚡ 断点续跑：已完成 {already_done}/{total_todo}，"
              f"跳过已有结果")
    print("=" * 65)

    datasets = load_all(verbose=False)

    for ds_name in MULTI_POP_DATASETS:
        X, y, markers, pop_names = datasets[ds_name]
        k   = BEST_K[ds_name]
        ref = SEED42_REF[ds_name]

        # 检查该数据集是否还有待完成的 seed
        pending_seeds = [s for s in SEEDS if (ds_name, s) not in completed]
        if not pending_seeds:
            print(f"\n[{ds_name}]  All {N_RUNS} runs complete, skipping.")
            continue

        print(f"\n{'─' * 65}")
        print(f"  {ds_name}  |  cells={X.shape[0]:,}  markers={X.shape[1]}"
              f"  k={k}  K_true={TRUE_K[ds_name]}")
        print(f"  seed=42 参考：ARI={ref['ARI']}  "
              f"MacroF1={ref['MacroF1']}  K_found={ref['K_found']}")
        print(f"  待运行 seeds：{pending_seeds}")
        print(f"{'─' * 65}")

        for seed in pending_seeds:
            t0     = time.time()
            y_pred = run_phenograph(X, k=k, seed=seed)
            rt     = round(time.time() - t0, 1)

            metrics = evaluate(y, y_pred, dataset_name=ds_name)
            ari     = metrics['ARI']
            f1      = metrics['MacroF1']
            k_found = metrics['n_clusters_found']

            row = {
                'algorithm'  : ALGORITHM,
                'dataset'    : ds_name,
                'k_fixed'    : k,
                'seed'       : seed,
                'ARI'        : ari,
                'MacroF1'    : f1,
                'K_found'    : k_found,
                'runtime_sec': rt,
            }

            # ── 立即写入，崩溃不丢数据 ───────────────────────
            append_row(RAW_CSV, row)
            completed.add((ds_name, seed))

            print(f"  [{ALGORITHM} | {ds_name} | seed={seed:02d}]  "
                  f"ARI={ari:.4f}  MacroF1={f1:.4f}  "
                  f"K_found={k_found}  ({rt}s)  -> saved")

    # ── 所有运行完成后生成 summary ────────────────────────────
    print(f"\n原始数据：{RAW_CSV}")
    df_sum = build_summary(RAW_CSV, SUMMARY_CSV)
    print(f"Summary saved: {SUMMARY_CSV}")

    # ── 最终摘要打印 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PhenoGraph stability summary (std > 0.02 flagged, with 95% CI)")
    print("=" * 65)

    for _, r in df_sum.iterrows():
        ari_flag = ' ⚠️' if r['ARI_std'] > 0.02 else ''
        f1_flag  = ' ⚠️' if r['MacroF1_std'] > 0.02 else ''
        delta    = r['ARI_mean_vs_seed42']
        print(f"\n  {r['dataset']}  (k={r['k_fixed']})")
        print(f"    ARI      {r['ARI_mean']:.4f} ± {r['ARI_std']:.4f}"
              f"  [CI95 ±{r['ARI_CI95']:.4f}]"
              f"  [{r['ARI_min']:.4f}, {r['ARI_max']:.4f}]"
              f"  Δseed42={delta:+.4f}{ari_flag}")
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
