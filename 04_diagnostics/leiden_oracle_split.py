"""
04_diagnostics/leiden_oracle_split.py
================================
量化 Leiden 算法在 Mosmann_rare 上
"ARI-optimal 参数 ≠ rare-group-optimal 参数"的分裂现象。

扫描所有 resolution 值，同时记录 ARI 和 activated_F1，
直接可视化两套 oracle 标准给出的不同结论。

输出：
    results/leiden_oracle_split.csv
    figures/leiden_oracle_split.png
"""
# ── 路径修复：从子目录 import 根目录模块 ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_dataset
from evaluation  import evaluate

warnings.filterwarnings('ignore')

DATASET     = 'Mosmann_rare'
TARGET      = 'activated'
RESOLUTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
RESULTS_CSV = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'results', 'diagnostics', 'leiden_oracle_split.csv'
)
FIGURE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', 'leiden_oracle_split.png')

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)


def run_leiden(X, resolution):
    import scanpy as sc
    import anndata as ad
    adata = ad.AnnData(X=X.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution, random_state=42)
    return adata.obs['leiden'].astype(int).values


def main():
    print("=" * 58)
    print("  Leiden oracle-split experiment")
    print(f"  Dataset: {DATASET}  resolution scan: {RESOLUTIONS}")
    print("=" * 58)

    X, y, _, _ = load_dataset(DATASET, verbose=True)
    rows = []

    for res in RESOLUTIONS:
        print(f"  resolution={res:<5}", end='  ', flush=True)
        try:
            y_pred  = run_leiden(X, res)
            metrics = evaluate(y, y_pred, dataset_name=DATASET)
            ari     = metrics['ARI']
            rf1     = metrics['rare_target_f1']
            rp      = metrics.get('rare_target_precision', float('nan'))
            rr      = metrics.get('rare_target_recall',    float('nan'))
            k       = metrics['n_clusters_found']
            print(f"ARI={ari:.4f}  activated_F1={rf1:.4f}  "
                  f"P={rp:.4f}  R={rr:.4f}  K={k}")
            rows.append({'resolution':res,'ARI':ari,
                         'activated_F1':rf1,
                         'activated_P':rp,'activated_R':rr,
                         'K':k})
        except Exception as e:
            print(f"❌ {e}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved: {RESULTS_CSV}")

    # ── 可视化 ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        'Leiden on Mosmann_rare: ARI vs. Rare-group F1\n'
        'ARI-optimal resolution ≠ Rare-F1-optimal resolution',
        fontsize=12, fontweight='bold'
    )

    res_vals = df['resolution'].values

    # 上图：ARI
    ax1.plot(res_vals, df['ARI'], 'o-', color='#1976D2',
             linewidth=2, markersize=7, label='ARI')
    best_ari_idx = df['ARI'].idxmax()
    ax1.axvline(df.loc[best_ari_idx, 'resolution'],
                color='#1976D2', linestyle='--', alpha=0.6,
                label=f"ARI-optimal res="
                      f"{df.loc[best_ari_idx,'resolution']}")
    ax1.set_ylabel('ARI', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.01, max(df['ARI'].max()*1.15, 0.05))

    # 下图：activated F1
    ax2.plot(res_vals, df['activated_F1'], 's-', color='#E53935',
             linewidth=2, markersize=7, label='activated F1')
    best_f1_idx = df['activated_F1'].idxmax()
    ax2.axvline(df.loc[best_f1_idx, 'resolution'],
                color='#E53935', linestyle='--', alpha=0.6,
                label=f"F1-optimal res="
                      f"{df.loc[best_f1_idx,'resolution']}")
    ax2.fill_between(res_vals, df['activated_R'], df['activated_P'],
                     alpha=0.15, color='#E53935',
                     label='P–R range')
    ax2.set_xlabel('Leiden resolution', fontsize=10)
    ax2.set_ylabel('activated F1 / P / R', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.01, 1.05)

    # 标注最优点
    ax1.annotate(f"ARI={df.loc[best_ari_idx,'ARI']:.4f}",
                 xy=(df.loc[best_ari_idx,'resolution'],
                     df.loc[best_ari_idx,'ARI']),
                 xytext=(10, -15), textcoords='offset points',
                 fontsize=8, color='#1976D2',
                 arrowprops=dict(arrowstyle='->', color='#1976D2'))

    ax2.annotate(f"F1={df.loc[best_f1_idx,'activated_F1']:.4f}",
                 xy=(df.loc[best_f1_idx,'resolution'],
                     df.loc[best_f1_idx,'activated_F1']),
                 xytext=(10, -15), textcoords='offset points',
                 fontsize=8, color='#E53935',
                 arrowprops=dict(arrowstyle='->', color='#E53935'))

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {FIGURE_PATH}")

    # ── 终端汇总 ──────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  Oracle-split quantification")
    print(f"{'─'*58}")
    best_ari_row = df.loc[df['ARI'].idxmax()]
    best_f1_row  = df.loc[df['activated_F1'].idxmax()]
    print(f"  ARI-optimal:  res={best_ari_row['resolution']}  "
          f"ARI={best_ari_row['ARI']:.4f}  "
          f"F1={best_ari_row['activated_F1']:.4f}  "
          f"K={int(best_ari_row['K'])}")
    print(f"  F1-optimal:   res={best_f1_row['resolution']}  "
          f"ARI={best_f1_row['ARI']:.4f}  "
          f"F1={best_f1_row['activated_F1']:.4f}  "
          f"K={int(best_f1_row['K'])}")
    gap_f1  = best_f1_row['activated_F1'] - best_ari_row['activated_F1']
    gap_ari = best_ari_row['ARI'] - best_f1_row['ARI']
    print(f"\n  F1 gap (F1-oracle vs ARI-oracle): +{gap_f1:.4f}")
    print(f"  ARI gap (ARI-oracle vs F1-oracle): +{gap_ari:.4f}")
    print(f"{'═'*58}")


if __name__ == '__main__':
    main()
