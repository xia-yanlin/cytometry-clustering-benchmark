"""
03_stability/run_blflowsom_stability.py  ── 修正版（2026-05-27）
====================================
对 BL-FlowSOM 运行 N_RUNS 次，验证确定性（std 应为 0）。

修正内容（同 flow_gating_BLFlowSOM_fixed.py）：
─────────────────────────────────────────────
Bug 1 ✅ sigma 衰减：max(1.0,...) 夹断 → 线性衰减至 sigma_end
Bug 2 ✅ rlen: 10 → 100
Bug 3 ✅ PCA 展布: ±1σ → ±2σ
─────────────────────────────────────────────
"""

# ── 路径修复 ──────────────────────────────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ─────────────────────────────────────────────────────────────

import time
import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA

from data_loader import load_all, TRUE_K
from evaluation  import evaluate

# ── 配置 ──────────────────────────────────────────────────────
N_RUNS = 10
SEEDS  = list(range(N_RUNS))

MULTI_POP_DATASETS = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']

# FlowSOM 10次结果（来自 multirun_raw.csv），用于对比
FLOWSOM_REF = {
    'Levine_13dim': {'ARI_mean': 0.8797, 'ARI_std': 0.0177},
    'Levine_32dim': {'ARI_mean': 0.8146, 'ARI_std': 0.1285},
    'Samusik_01'  : {'ARI_mean': 0.7836, 'ARI_std': 0.1542},
}

ALGORITHM = 'BL-FlowSOM'

# ── 修正后的超参数（与 flow_gating_BLFlowSOM_fixed.py 保持一致）──
SOM_DIM     = 10
RLEN        = 100   # ✅ Bug 2 修正（原 10）
SIGMA_START = 5.0   # ✅ Bug 1 修正：max(xdim,ydim)/2（原 3.3）
SIGMA_END   = 0.3   # ✅ Bug 1 修正：不再 max(1.0,...) 夹断
PCA_SCALE   = 2.0   # ✅ Bug 3 修正：±2σ（原 ±1σ）
# ─────────────────────────────────────────────────────────────

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RAW_CSV     = _os.path.join(_ROOT, 'results', 'stability',
                            'blflowsom_stability_raw.csv')
SUMMARY_CSV = _os.path.join(_ROOT, 'results', 'stability',
                            'blflowsom_stability_summary.csv')


def ci95(arr):
    n = len(arr)
    if n < 2:
        return float('nan')
    return float(_stats.t.ppf(0.975, df=n-1) * np.std(arr, ddof=1) / np.sqrt(n))


def run_blflowsom(X, n_clusters):
    """
    独立实现（避免循环 import）。
    已修正三个 Bug，与 flow_gating_BLFlowSOM_fixed.py 保持完全一致。
    """
    n_nodes = SOM_DIM * SOM_DIM

    # ── PCA 初始化（Bug 3 修正：±PCA_SCALE σ）────────────────
    pca = PCA(n_components=min(2, X.shape[1]))
    pca.fit(X)
    mean = X.mean(axis=0)
    rs = np.linspace(-PCA_SCALE, PCA_SCALE, SOM_DIM)
    cs = np.linspace(-PCA_SCALE, PCA_SCALE, SOM_DIM)
    codes = np.zeros((n_nodes, X.shape[1]))
    idx = 0
    for r in rs:
        for c in cs:
            codes[idx] = mean.copy()
            codes[idx] += r * np.sqrt(pca.explained_variance_[0]) * pca.components_[0]
            if pca.n_components_ > 1:
                codes[idx] += c * np.sqrt(pca.explained_variance_[1]) * pca.components_[1]
            idx += 1

    positions = np.array(
        [(i, j) for i in range(SOM_DIM) for j in range(SOM_DIM)], dtype=float
    )

    def find_winners(codes_):
        winners = np.empty(len(X), dtype=int)
        for s in range(0, len(X), 8000):
            e = min(s + 8000, len(X))
            d = cdist(X[s:e], codes_, 'sqeuclidean')
            winners[s:e] = d.argmin(axis=1)
        return winners

    # ── 批量训练（Bug 1+2 修正）───────────────────────────────
    for t in range(RLEN):
        # ✅ 修正 Bug 1：线性衰减，不再 max(1.0,...) 夹断
        progress = t / max(1, RLEN - 1)
        sigma    = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        sq_d = cdist(positions, positions, 'sqeuclidean')
        H    = np.exp(-sq_d / (2.0 * sigma ** 2))

        winners     = find_winners(codes)
        node_sums   = np.zeros_like(codes)
        node_counts = np.zeros(n_nodes)
        np.add.at(node_sums,   winners, X)
        np.add.at(node_counts, winners, 1)

        num = H @ node_sums
        den = H @ node_counts
        nz  = den > 0
        new = codes.copy()
        new[nz] = num[nz] / den[nz, None]
        codes = new

    # ── Meta-clustering（不变）───────────────────────────────
    winners_final = find_winners(codes)
    Z     = linkage(codes, method='ward')
    meta  = fcluster(Z, n_clusters, criterion='maxclust') - 1
    return meta[winners_final].astype(int)


def append_row(raw_csv, row):
    _os.makedirs(_os.path.dirname(raw_csv), exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not _os.path.exists(raw_csv)
    df.to_csv(raw_csv, mode='a', header=write_header, index=False)


def build_summary(raw_csv):
    df_raw = pd.read_csv(raw_csv)
    rows = []
    for ds in MULTI_POP_DATASETS:
        sub = df_raw[df_raw['dataset'] == ds]
        if sub.empty:
            continue
        ari_v = sub['ARI'].dropna().values
        f1_v  = sub['MacroF1'].dropna().values
        ref   = FLOWSOM_REF[ds]

        rows.append({
            'algorithm'           : ALGORITHM,
            'dataset'             : ds,
            'n_runs'              : len(sub),
            'rlen'                : RLEN,
            'sigma_start'         : SIGMA_START,
            'sigma_end'           : SIGMA_END,
            'pca_scale'           : PCA_SCALE,
            'ARI_mean'            : round(ari_v.mean(), 4),
            'ARI_std'             : round(ari_v.std(),  6),
            'ARI_CI95'            : round(ci95(ari_v),  6),
            'ARI_min'             : round(ari_v.min(),  4),
            'ARI_max'             : round(ari_v.max(),  4),
            'MacroF1_mean'        : round(f1_v.mean(), 4),
            'MacroF1_std'         : round(f1_v.std(),  6),
            'FlowSOM_ARI_mean'    : ref['ARI_mean'],
            'FlowSOM_ARI_std'     : ref['ARI_std'],
            'stability_gain_x'    : round(ref['ARI_std'] / max(ari_v.std(), 1e-8), 1),
        })

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(SUMMARY_CSV, index=False)
    return df_sum


def main():
    print("=" * 65)
    print(f"BL-FlowSOM stability validation (corrected)  |  {N_RUNS} runs")
    print(f"rlen={RLEN}  sigma={SIGMA_START}→{SIGMA_END}  pca_scale=±{PCA_SCALE}σ")
    print("预期：ARI std ≈ 0（确定性算法，PCA初始化+批量学习）")
    print("=" * 65)

    datasets  = load_all(verbose=False)
    completed = set()

    if _os.path.exists(RAW_CSV):
        df_done = pd.read_csv(RAW_CSV)
        # 注意：修正后的参数与旧结果不兼容，旧 CSV 应删除或重命名
        completed = set(zip(df_done['dataset'], df_done['run_id']))
        print(f"Resuming: {len(completed)} runs already complete, skipping")
        print("  ⚠️  NOTE: if existing CSV used old parameters (rlen=10), delete it and re-run")

    for ds in MULTI_POP_DATASETS:
        X, y, markers, _ = datasets[ds]
        k = TRUE_K[ds]
        ref = FLOWSOM_REF[ds]

        pending = [i for i in range(N_RUNS) if (ds, i) not in completed]
        if not pending:
            print(f"\n[{ds}]  全部完成，跳过。")
            continue

        print(f"\n{'─'*65}")
        print(f"  {ds}  cells={X.shape[0]:,}  K_true={k}")
        print(f"  FlowSOM 参考：ARI={ref['ARI_mean']:.4f} ± {ref['ARI_std']:.4f}")
        print(f"{'─'*65}")

        ari_list, f1_list = [], []

        for run_id in pending:
            t0     = time.time()
            y_pred = run_blflowsom(X, n_clusters=k)
            rt     = round(time.time() - t0, 1)

            metrics = evaluate(y, y_pred, dataset_name=ds)
            ari, f1 = metrics['ARI'], metrics['MacroF1']
            ari_list.append(ari)
            f1_list.append(f1)

            row = {
                'algorithm'  : ALGORITHM,
                'dataset'    : ds,
                'run_id'     : run_id,
                'rlen'       : RLEN,
                'sigma_start': SIGMA_START,
                'sigma_end'  : SIGMA_END,
                'pca_scale'  : PCA_SCALE,
                'ARI'        : ari,
                'MacroF1'    : f1,
                'K_found'    : metrics['n_clusters_found'],
                'runtime_sec': rt,
            }
            append_row(RAW_CSV, row)
            completed.add((ds, run_id))
            print(f"  [{ALGORITHM} | {ds} | run={run_id:02d}]  "
                  f"ARI={ari:.4f}  MacroF1={f1:.4f}  ({rt}s)  → 已写入")

        ari_arr = np.array(ari_list)
        print(f"  → ARI  {ari_arr.mean():.4f} ± {ari_arr.std():.6f}"
              f"  [{ari_arr.min():.4f}, {ari_arr.max():.4f}]"
              f"  （FlowSOM: {ref['ARI_std']:.4f}）")

    print(f"\n原始数据：{RAW_CSV}")
    df_sum = build_summary(RAW_CSV)
    print(f"Summary: {SUMMARY_CSV}")

    print("\n" + "=" * 65)
    print("  BL-FlowSOM vs FlowSOM stability comparison")
    print("=" * 65)
    for _, r in df_sum.iterrows():
        flag = "✅ 完全确定性" if r['ARI_std'] < 1e-6 else \
               "✅ 高度稳定"   if r['ARI_std'] < 0.01  else "⚠️ 存在波动"
        print(f"\n  {r['dataset']}")
        print(f"    BL-FlowSOM  ARI={r['ARI_mean']:.4f} ± {r['ARI_std']:.6f}  {flag}")
        print(f"    FlowSOM     ARI={r['FlowSOM_ARI_mean']:.4f} ± {r['FlowSOM_ARI_std']:.4f}  "
              f"stability gain {r['stability_gain_x']:.0f}x")

    print("\n" + "=" * 65)
    print("  Done")
    print("=" * 65)


if __name__ == '__main__':
    main()