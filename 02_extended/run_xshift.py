"""
02_extended/run_xshift.py
======================
X-shift clustering algorithm evaluated on all five benchmark datasets.

算法原理（Vargas et al., EBioMedicine 2016）：
    1. 对每个细胞计算 k 近邻密度（KNN density）
    2. 每个细胞"跟随"其 k 近邻中密度最高的邻居
    3. 沿密度梯度方向反复跟随，直到收敛到局部密度极大值（峰值）
    4. 收敛到同一峰值的细胞归为一个簇
    5. 扫描 k 值（{5,10,15,20,30,45,60}），按 ARI 取最优

与 Weber 2016 的对应：
    - Weber 使用 FlowJo 插件实现；本脚本为忠实的 Python 复现
    - 参数范围与 Weber 2016 supplementary 对齐
    - 结果与 Weber 2016 Table 3 对比，验证实现正确性

输出：
    results/xshift_results.csv
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

from data_loader import load_all, DATASETS, TRUE_K
from evaluation  import evaluate, save_results, print_summary

warnings.filterwarnings('ignore')

ALGORITHM   = 'X-shift'
RESULTS_CSV = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'results', 'extended', 'xshift_results.csv'
)

# k scan range aligned with Weber 2016 supplementary
K_GRID_STANDARD = [5, 10, 15, 20, 30, 45, 60]
K_GRID_LARGE    = [15, 20, 30, 45, 60]   # 大数据集（>100k）节省时间

# Weber 2016 Table 3 reference values for validation
WEBER_REFERENCE = {
    'Levine_13dim': {'MacroF1': 0.45, 'ARI': None},
    'Levine_32dim': {'MacroF1': 0.58, 'ARI': None},
    'Samusik_01'  : {'MacroF1': 0.71, 'ARI': None},
}


# ════════════════════════════════════════════════════════════
# X-shift 核心实现
# ════════════════════════════════════════════════════════════

def knn_density(X, k):
    """
    KNN 密度估计：第 k 近邻距离的倒数。
    密度越高 = 细胞越密集 = 背景区域。
    """
    tree        = KDTree(X)
    dists, idxs = tree.query(X, k=k + 1)   # k+1 因为包含自身
    kth_dist    = dists[:, k] + 1e-9
    density     = 1.0 / kth_dist
    return density, idxs[:, 1:]             # 去掉自身，返回 k 近邻索引


def xshift_cluster(X, k, max_iter=100):
    """
    X-shift 聚类核心算法。

    步骤：
    1. 计算每个细胞的 KNN 密度
    2. 每个细胞跟随 k 近邻中密度最高的邻居（爬山）
    3. 重复直到所有细胞的跟随对象不再改变（收敛）
    4. 找到最终的密度峰值（没有更高密度邻居的细胞）
    5. 回溯每个细胞的跟随链，确定其所属峰值（簇）

    参数
    ----
    X       : (n, d) 特征矩阵
    k       : 近邻数
    max_iter: 最大迭代次数（防止死循环）

    返回
    ----
    labels : (n,) 整数数组，0-based 簇编号
    """
    n        = len(X)
    density, neighbor_idx = knn_density(X, k)

    # 每个细胞的"跟随目标"：k 近邻中密度最高的那个
    # follow[i] = j 表示细胞 i 跟随细胞 j
    follow = np.array([
        neighbor_idx[i, np.argmax(density[neighbor_idx[i]])]
        for i in range(n)
    ])

    # 如果自身密度已经是局部最大值，指向自身
    for i in range(n):
        if density[i] >= density[follow[i]]:
            follow[i] = i

    # 迭代：沿跟随链传播，直到收敛
    for _ in range(max_iter):
        new_follow = follow[follow]   # 跟随两步
        if np.all(new_follow == follow):
            break
        follow = new_follow

    # 找密度峰值（自指节点）
    peaks      = np.where(follow == np.arange(n))[0]
    peak_set   = set(peaks)

    # 如果没有自指节点，取密度最高的细胞作为唯一峰值
    if len(peaks) == 0:
        peaks    = np.array([np.argmax(density)])
        peak_set = set(peaks)

    # 为每个峰值分配簇编号
    peak_label = {p: i for i, p in enumerate(peaks)}

    # 每个细胞溯源到其峰值
    labels = np.full(n, -1, dtype=int)
    for p in peaks:
        labels[p] = peak_label[p]

    # 对非峰值细胞，沿 follow 链找到峰值
    # 用迭代传播而非递归，避免 Python 递归深度限制
    for _ in range(max_iter):
        unresolved = labels == -1
        if not unresolved.any():
            break
        # 如果 follow[i] 已经有标签，继承
        can_resolve = unresolved & (labels[follow] != -1)
        labels[can_resolve] = labels[follow[can_resolve]]

    # 兜底：仍未解析的细胞（极罕见）分配到最近峰值
    still_unresolved = labels == -1
    if still_unresolved.sum() > 0:
        peak_X   = X[peaks]
        tree_p   = KDTree(peak_X)
        idx_near = tree_p.query(X[still_unresolved], k=1,
                                return_distance=False).flatten()
        labels[still_unresolved] = idx_near

    # 重映射为连续 0-based（去除可能的空簇）
    unique = np.unique(labels)
    remap  = {v: i for i, v in enumerate(unique)}
    return np.array([remap[l] for l in labels])


# ════════════════════════════════════════════════════════════
# 单数据集运行（k 扫描取最优）
# ════════════════════════════════════════════════════════════

def run_xshift_scan(X, y, dataset_name, k_grid):
    """
    扫描 k 值，取 ARI 最高的结果（oracle 模式，与 Weber 2016 一致）。
    """
    best_ari     = -np.inf
    best_metrics = None
    best_k       = None
    best_pred    = None

    for k in k_grid:
        try:
            y_pred   = xshift_cluster(X, k=k)
            metrics  = evaluate(y, y_pred, dataset_name=dataset_name)
            ari      = metrics['ARI']
            k_found  = metrics['n_clusters_found']
            rf1      = metrics.get('rare_target_f1', None)

            status = ''
            if rf1 is not None:
                status = f"  rare_F1={rf1:.4f}"
            print(f"    k={k:<4}  ARI={ari:.4f}  K_found={k_found}{status}")

            if ari > best_ari:
                best_ari     = ari
                best_metrics = metrics
                best_k       = k
                best_pred    = y_pred
        except Exception as e:
            print(f"    k={k}  ❌ {e}")

    return best_metrics, best_k, best_pred


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print(f"  Algorithm: {ALGORITHM}  |  K mode: oracle (ARI-optimal)")
    print("=" * 62)

    datasets = load_all(verbose=False)
    all_rows = []

    for name in DATASETS:
        X, y, markers, pop_names = datasets[name]
        n = len(X)

        k_grid = K_GRID_LARGE if n > 100_000 else K_GRID_STANDARD

        print(f"\n[{name}]  cells={n:,}  markers={X.shape[1]}  "
              f"k_grid={k_grid}")
        print("  Scanning...")

        t0 = time.time()
        metrics, best_k, _ = run_xshift_scan(X, y, name, k_grid)
        rt = round(time.time() - t0, 1)

        if metrics is None:
            print(f"  ❌ 所有 k 值均失败")
            continue

        print(f"\n  Best k={best_k}  (total {rt}s)")
        print_summary(metrics, name, ALGORITHM)

        row = {
            'dataset'        : name,
            'algorithm'      : ALGORITHM,
            'K_mode'         : 'K=oracle',
            'best_k'         : best_k,
            'n_clusters_true': TRUE_K[name],
            'runtime_sec'    : rt,
            'ARI'            : metrics['ARI'],
            'MacroF1'        : metrics['MacroF1'],
            'accuracy'       : metrics['accuracy'],
            'n_clusters_found': metrics['n_clusters_found'],
            'rare_target_f1' : metrics.get('rare_target_f1'),
            'rare_target_precision': metrics.get('rare_target_precision'),
            'rare_target_recall'   : metrics.get('rare_target_recall'),
            'rare_target_f2'       : metrics.get('rare_target_f2'),
            'rare_target_found'    : metrics.get('rare_target_found'),
        }
        all_rows.append(row)

    # ── 保存结果 ─────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n\nResults saved: {RESULTS_CSV}  ({len(df)} rows)")

    # ── 验收：与 Weber 2016 对比 ──────────────────────────────
    print(f"\n{'═'*62}")
    print("  Validation: comparison with Weber 2016 Table 3")
    print(f"{'─'*62}")
    print(f"  {'数据集':<18} {'本次 MacroF1':>14} {'Weber F1':>10} "
          f"{'差异':>8} {'状态':>6}")
    print(f"  {'─'*56}")

    for _, row in df.iterrows():
        ds   = row['dataset']
        ref  = WEBER_REFERENCE.get(ds)
        if ref is None or row['MacroF1'] is None:
            continue
        f1   = row['MacroF1']
        wf1  = ref['MacroF1']
        diff = f1 - wf1
        ok   = '✅' if abs(diff) <= 0.07 else '⚠️ '
        print(f"  {ds:<18} {f1:>14.4f} {wf1:>10.2f} "
              f"{diff:>+8.4f} {ok:>6}")

    # ── 稀有群体汇总 ──────────────────────────────────────────
    print(f"\n  稀有群体检测结果：")
    print(f"  {'─'*56}")
    for _, row in df.iterrows():
        ds  = row['dataset']
        rf1 = row.get('rare_target_f1')
        if rf1 is None:
            continue
        rp  = row.get('rare_target_precision', float('nan'))
        rr  = row.get('rare_target_recall',    float('nan'))
        found = '✅' if row.get('rare_target_found') else '❌'
        print(f"  {ds:<20}  F1={rf1:.4f}  "
              f"P={rp:.4f}  R={rr:.4f}  {found}")

    print(f"{'═'*62}")


if __name__ == '__main__':
    main()
