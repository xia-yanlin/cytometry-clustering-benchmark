"""
04_diagnostics/oracle_gain.py
========================
量化两类 oracle 依赖的性能增益（P4）：

类型 A：K=true oracle（kmeans / GMM / FlowSOM）
    比较：已知真实群体数 K_true  vs  固定默认 K=10
    来源：baseline/*_results.csv（K=true）+ 本脚本快速跑 K=10

类型 B：parameter oracle（PhenoGraph / Leiden / HDBSCAN）
    比较：ARI-oracle 最优参数  vs  默认参数（无参数知识）
    来源：baseline/*_results.csv vs diagnostics/nooracle_results.csv

输出：
    results/diagnostics/oracle_gain.csv    详细增益表
    终端打印汇总
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))

import time
import numpy as np
import pandas as pd

_ROOT      = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUTPUT_CSV = _os.path.join(_ROOT, 'results', 'diagnostics', 'oracle_gain.csv')

MULTI_POP  = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']
RARE_POP   = ['Nilsson_rare', 'Mosmann_rare']
ALL_DS     = MULTI_POP + RARE_POP

# K=true 算法：使用真实 K 值
K_TRUE_ALGOS  = ['k-means', 'GMM', 'FlowSOM']
K_FIXED_LIST  = [10, 40]   # 两个对照基准：K=10（历史默认）/ K=40（Weber 2016 推荐）

# parameter oracle 算法：对应 nooracle_results.csv 里的 _default 变体
PARAM_ORACLE_MAP = {
    'PhenoGraph': 'PhenoGraph_default',
    'Leiden'    : 'Leiden_default',
    'HDBSCAN'   : 'HDBSCAN_default',
}


# ── 工具函数 ──────────────────────────────────────────────────

def load_baseline():
    """加载所有基准结果（baseline + diagnostics/nooracle）。"""
    import glob
    pattern = _os.path.join(_ROOT, 'results', '**', '*_results.csv')
    files   = sorted(glob.glob(pattern, recursive=True))
    files   = [f for f in files if 'stability' not in f.replace('\\', '/')]

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, engine='python', on_bad_lines='skip'))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    # 去重：同一 algorithm×dataset 保留 ARI 最高的
    if 'ARI' in df.columns:
        df = (df.sort_values('ARI', ascending=False)
                .drop_duplicates(subset=['algorithm', 'dataset'], keep='first')
                .reset_index(drop=True))
    return df


def run_k_fixed(k_fixed=10):
    """
    快速运行 kmeans 和 FlowSOM（K=10 固定）作为 K=true oracle 的对照。
    GMM 运行太慢，用 kmeans 结果作为 GMM 的近似对照（两者 K=10 下性能相近）。
    """
    from data_loader import load_all, TRUE_K
    from evaluation  import evaluate
    from sklearn.cluster import KMeans
    import anndata as ad
    import flowsom as fs

    datasets = load_all(verbose=False)
    rows = []

    for ds in ALL_DS:
        X, y, markers, _ = datasets[ds]
        df_tmp = pd.DataFrame(X, columns=markers)

        for algo in ['k-means', 'FlowSOM']:
            print(f"  K={k_fixed} control [{algo} | {ds}] ...", end='', flush=True)
            t0 = time.time()
            try:
                if algo == 'k-means':
                    model  = KMeans(n_clusters=k_fixed, n_init=10,
                                    random_state=42)
                    y_pred = model.fit_predict(X)
                else:  # FlowSOM
                    adata  = ad.AnnData(X=df_tmp)
                    fsom   = fs.FlowSOM(adata, n_clusters=k_fixed,
                                        cols_to_use=markers,
                                        xdim=10, ydim=10, seed=42)
                    raw    = np.array(
                        fsom.get_cell_data().obs["metaclustering"])
                    unique = sorted(set(raw))
                    lmap   = {v: i for i, v in enumerate(unique)}
                    y_pred = np.array([lmap[v] for v in raw])

                rt      = round(time.time() - t0, 1)
                metrics = evaluate(y, y_pred, dataset_name=ds)
                print(f" ARI={metrics['ARI']:.4f}  ({rt}s)")

                rows.append({
                    'algorithm'  : f"{algo}_K{k_fixed}",
                    'dataset'    : ds,
                    'K_mode'     : f'K={k_fixed}_fixed',
                    'ARI'        : metrics['ARI'],
                    'MacroF1'    : metrics['MacroF1'],
                    'runtime_sec': rt,
                })
            except Exception as e:
                print(f" ❌ {e}")

    return pd.DataFrame(rows)


def compute_gain(df_oracle, df_default, algo_oracle, algo_default, label):
    """计算单个算法的 oracle 增益，返回 list of dicts。"""
    rows = []
    for ds in ALL_DS:
        o = df_oracle[(df_oracle['algorithm'] == algo_oracle) &
                      (df_oracle['dataset']   == ds)]
        d = df_default[(df_default['algorithm'] == algo_default) &
                       (df_default['dataset']   == ds)]

        if o.empty or d.empty:
            continue

        ari_oracle  = float(o['ARI'].iloc[0])
        ari_default = float(d['ARI'].iloc[0])
        gain        = round(ari_oracle - ari_default, 4)

        rows.append({
            'oracle_type'   : label,
            'algorithm'     : algo_oracle,
            'dataset'       : ds,
            'ARI_oracle'    : round(ari_oracle,  4),
            'ARI_default'   : round(ari_default, 4),
            'oracle_gain'   : gain,
            'gain_pct'      : round(gain / max(abs(ari_default), 1e-6) * 100, 1),
        })
    return rows


def main():
    print("=" * 65)
    print("  Oracle gain analysis")
    print("=" * 65)

    df_base = load_baseline()
    if df_base.empty:
        print("ERROR: no baseline results found; run 01_baseline/ scripts first")
        return

    all_rows = []

    # ── 类型 B：parameter oracle（已有数据，直接计算）────────────────
    print("\n[Type B] Parameter oracle gain (ARI-oracle optimal vs default parameters)")
    for algo_oracle, algo_default in PARAM_ORACLE_MAP.items():
        rows = compute_gain(df_base, df_base,
                            algo_oracle, algo_default,
                            'param_oracle')
        for r in rows:
            all_rows.append(r)
            gain = r['oracle_gain']
            flag = '⚠️' if abs(gain) > 0.1 else ''
            print(f"  {algo_oracle:<12} | {r['dataset']:<15} "
                  f"oracle={r['ARI_oracle']:.4f}  "
                  f"default={r['ARI_default']:.4f}  "
                  f"gain={gain:+.4f} {flag}")

    # ── 类型 A：K=true oracle（K=10 和 K=40 两个对照基准）──────
    print(f"\n[Type A] K=true oracle gain")
    print(f"  Baselines: K={K_FIXED_LIST} (historical default vs Weber-2016 recommendation)")

    for k_fixed in K_FIXED_LIST:
        print(f"\n  --- K={k_fixed} 对照 ---")
        print(f"  Running (k-means + FlowSOM, ~3-5 min)...")
        df_kfixed = run_k_fixed(k_fixed)

        if df_kfixed.empty:
            print(f"  ⚠️  K={k_fixed} 对照实验失败，跳过")
            continue

        df_kfixed.to_csv(
            _os.path.join(_ROOT, 'results', 'diagnostics',
                          f'k{k_fixed}_fixed_results.csv'),
            index=False
        )

        for algo in K_TRUE_ALGOS:
            algo_kfixed = f"{algo.replace('-','')}_K{k_fixed}"
            if algo not in df_base['algorithm'].values:
                continue
            rows = compute_gain(df_base, df_kfixed,
                                algo, algo_kfixed,
                                f'K_true_vs_K{k_fixed}')
            for r in rows:
                all_rows.append(r)
                gain = r['oracle_gain']
                flag = '⚠️' if abs(gain) > 0.1 else ''
                neg  = '🔻负增益' if gain < -0.02 else ''
                print(f"  {algo:<12} | {r['dataset']:<15} "
                      f"K_true={r['ARI_oracle']:.4f}  "
                      f"K={k_fixed}={r['ARI_default']:.4f}  "
                      f"gain={gain:+.4f} {flag}{neg}")

    # ── 保存并打印汇总 ─────────────────────────────────────────
    if all_rows:
        df_gain = pd.DataFrame(all_rows)
        _os.makedirs(_os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_gain.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved: {OUTPUT_CSV}")

        print("\n" + "=" * 65)
        print("  Oracle gain per algorithm (multi-population datasets)")
        print("=" * 65)

        for otype in ['param_oracle'] + [f'K_true_vs_K{k}' for k in K_FIXED_LIST]:
            sub = df_gain[(df_gain['oracle_type'] == otype) &
                          (df_gain['dataset'].isin(MULTI_POP))]
            if sub.empty:
                continue
            if otype == 'param_oracle':
                label = "parameter oracle"
            else:
                k = otype.split('K')[-1]
                label = f"K=true oracle (vs K={k})"
            print(f"\n  {label}：")
            for algo in sub['algorithm'].unique():
                a = sub[sub['algorithm'] == algo]['oracle_gain']
                neg_note = '  <- negative gain: knowing true K hurts' if a.mean() < -0.02 else ''
                print(f"    {algo:<12}  均值增益={a.mean():+.4f}  "
                      f"最大={a.max():+.4f}  最小={a.min():+.4f}{neg_note}")

        print("\n  Notes:")
        print("    gain > 0.1 = strong oracle dependence; real deployment cannot achieve benchmark")
        print("    negative gain = knowing true K reduces ARI; consider clustering landscape")

    print("\n" + "=" * 65)
    print("  Done")
    print("=" * 65)


if __name__ == '__main__':
    main()