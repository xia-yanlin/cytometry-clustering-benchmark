"""
04_diagnostics/separability.py
============================
Separability-frequency diagnostic analysis (Spearman + stratified)

可分性定义：
    sep(p) = min_dist_to_other_centroid / intra_class_std
    其中 intra_class_std = 该群体所有标记物方差的均值开根号
    值越大 = 该群体与最近邻群体越容易区分

输入：
    results/summary/summary_all.csv
    5 个数据集原始数据（通过 data_loader 加载）

输出：
    results/diagnostics/separability_metrics_v2.csv
    figures/separability_scatter_v2_*.png
"""
# ── 路径修复：从子目录 import 根目录模块 ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from data_loader import load_dataset

warnings.filterwarnings('ignore')

_ROOT        = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SUMMARY_CSV  = _os.path.join(_ROOT, 'results', 'summary',      'summary_all.csv')
METRICS_CSV  = _os.path.join(_ROOT, 'results', 'diagnostics',  'separability_metrics_v2.csv')
FIGURES_DIR  = _os.path.join(_ROOT, 'figures')

_os.makedirs(FIGURES_DIR, exist_ok=True)

MULTI_POP_DS = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']
COLORS = {
    'Levine_13dim': '#2196F3',
    'Levine_32dim': '#FF9800',
    'Samusik_01'  : '#4CAF50',
}
RARE_COLOR   = '#E53935'
COMMON_ALPHA = 0.55
RARE_ALPHA   = 0.85


def compute_separability(X, y, pop):
    """
    可分性 = 到最近邻群体质心的距离 / 类内标准差均值
    这是论文中 sep 指标的完整定义。
    """
    mask = y == pop
    if mask.sum() < 2:
        return np.nan

    centroid  = X[mask].mean(axis=0)
    intra_std = X[mask].std(axis=0).mean() + 1e-9

    other_pops = [p for p in np.unique(y) if p != pop]
    if not other_pops:
        return np.nan

    min_dist = min(
        np.linalg.norm(centroid - X[y == p].mean(axis=0))
        for p in other_pops
    )
    return round(min_dist / intra_std, 4)


def extract_best_f1(summary_csv, ds_name):
    df  = pd.read_csv(summary_csv)
    sub = df[df['dataset'] == ds_name]
    f1_cols = [c for c in df.columns if c.startswith('f1_')]
    if sub.empty or not f1_cols:
        return {}
    best = {}
    for col in f1_cols:
        pop  = col[3:]
        vals = sub[col].dropna()
        if not vals.empty:
            best[pop] = round(float(vals.max()), 4)
    return best


def build_metrics():
    rows = []
    for ds_name in MULTI_POP_DS:
        print(f"  Processing {ds_name}...", end='', flush=True)
        X, y, _, _ = load_dataset(ds_name, verbose=False)
        best_f1    = extract_best_f1(SUMMARY_CSV, ds_name)
        n_total    = len(y)

        for pop in np.unique(y):
            n_pop = (y == pop).sum()
            freq  = round(100 * n_pop / n_total, 4)
            sep   = compute_separability(X, y, pop)
            bf1   = best_f1.get(str(pop), np.nan)
            rows.append({'dataset': ds_name, 'population': pop,
                         'n_cells': int(n_pop), 'frequency': freq,
                         'separability': sep, 'best_F1': bf1})

        print(f" {len(np.unique(y))} 个群体")

    df = pd.DataFrame(rows)
    _os.makedirs(_os.path.dirname(METRICS_CSV), exist_ok=True)
    df.to_csv(METRICS_CSV, index=False)
    print(f"  → saved: {METRICS_CSV}  ({len(df)} rows)")
    return df


def spearman_analysis(df):
    valid = df.dropna(subset=['frequency', 'separability', 'best_F1'])

    print(f"\n  {'─'*55}")
    print(f"  Spearman correlation analysis (n={len(valid)} populations)")
    print(f"  Note: populations within a dataset are not independent; interpret conservatively")
    print(f"  {'─'*55}")

    r_freq, p_freq = stats.spearmanr(valid['frequency'],    valid['best_F1'])
    r_sep,  p_sep  = stats.spearmanr(valid['separability'], valid['best_F1'])
    print(f"  Overall: frequency vs best_F1    ρ={r_freq:.3f}  p={p_freq:.4f}")
    print(f"  Overall: separability vs best_F1  ρ={r_sep:.3f}  p={p_sep:.4f}")

    rare   = valid[valid['frequency'] < 1.0]
    common = valid[valid['frequency'] >= 1.0]

    print(f"\n  Low-frequency (< 1%): n={len(rare)}")
    if len(rare) >= 3:
        r, p = stats.spearmanr(rare['frequency'],    rare['best_F1'])
        print(f"    频率 vs best_F1    ρ={r:.3f}  p={p:.4f}")
        r, p = stats.spearmanr(rare['separability'], rare['best_F1'])
        print(f"    可分性 vs best_F1  ρ={r:.3f}  p={p:.4f}  ← dominant factor for rare populations")

    print(f"\n  High-frequency (>= 1%): n={len(common)}")
    if len(common) >= 3:
        r, p = stats.spearmanr(common['frequency'],    common['best_F1'])
        print(f"    频率 vs best_F1    ρ={r:.3f}  p={p:.4f}  ← dominant factor for common populations")
        r, p = stats.spearmanr(common['separability'], common['best_F1'])
        print(f"    可分性 vs best_F1  ρ={r:.3f}  p={p:.4f}")

    print(f"  {'─'*55}")


def _scatter_ax(ax, sub, x_col, xlabel, title, show_spearman=True):
    valid       = sub.dropna(subset=[x_col, 'best_F1'])
    rare_mask   = valid['frequency'] < 1.0
    common_mask = ~rare_mask

    ax.scatter(valid.loc[common_mask, x_col],
               valid.loc[common_mask, 'best_F1'],
               c='#90A4AE', s=40, alpha=COMMON_ALPHA,
               label='频率 ≥ 1%', zorder=2)
    ax.scatter(valid.loc[rare_mask, x_col],
               valid.loc[rare_mask, 'best_F1'],
               c=RARE_COLOR, s=55, alpha=RARE_ALPHA,
               label='频率 < 1%', zorder=3)

    if show_spearman and len(valid) >= 3:
        r, p = stats.spearmanr(valid[x_col], valid['best_F1'])
        sig  = '***' if p < 0.001 else ('**' if p < 0.01 else
               ('*' if p < 0.05 else 'ns'))
        ax.text(0.05, 0.92, f'ρ = {r:.3f} {sig}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Best F1 across algorithms', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_all(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Separability & Frequency vs. Best F1\n'
                 '(Spearman ρ, red = frequency < 1%)',
                 fontsize=13, fontweight='bold')

    valid = df.dropna(subset=['frequency', 'separability', 'best_F1'])

    _scatter_ax(axes[0, 0], valid, 'frequency',
                'Cell Frequency (%)', 'All datasets — Frequency vs Best F1')
    _scatter_ax(axes[0, 1], valid, 'separability',
                'Separability (centroid dist / intra-std)',
                'All datasets — Separability vs Best F1')

    rare   = valid[valid['frequency'] < 1.0]['best_F1']
    common = valid[valid['frequency'] >= 1.0]['best_F1']
    axes[1, 0].boxplot([rare.values, common.values],
                       labels=[f'< 1%\n(n={len(rare)})',
                               f'≥ 1%\n(n={len(common)})'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#FFCDD2', color=RARE_COLOR),
                       medianprops=dict(color='black', linewidth=2))
    axes[1, 0].set_ylabel('Best F1', fontsize=10)
    axes[1, 0].set_title('Best F1 by Frequency Stratum',
                          fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    ds_rhos = []
    for ds in MULTI_POP_DS:
        sub = valid[valid['dataset'] == ds]
        if len(sub) >= 3:
            rf, _ = stats.spearmanr(sub['frequency'],    sub['best_F1'])
            rs, _ = stats.spearmanr(sub['separability'], sub['best_F1'])
            ds_rhos.append({'dataset': ds, 'freq_rho': rf, 'sep_rho': rs})

    if ds_rhos:
        rho_df = pd.DataFrame(ds_rhos)
        x, w   = np.arange(len(rho_df)), 0.35
        axes[1, 1].bar(x - w/2, rho_df['freq_rho'], w,
                       label='Frequency', color='#1976D2', alpha=0.8)
        axes[1, 1].bar(x + w/2, rho_df['sep_rho'],  w,
                       label='Separability', color='#388E3C', alpha=0.8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(
            [d.replace('_', '\n') for d in rho_df['dataset']], fontsize=8)
        axes[1, 1].set_ylabel('Spearman ρ', fontsize=10)
        axes[1, 1].set_title('Spearman ρ per Dataset',
                              fontsize=11, fontweight='bold')
        axes[1, 1].axhline(0, color='black', linewidth=0.8)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = _os.path.join(FIGURES_DIR, 'separability_scatter_v2_all.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_single(df, ds_name):
    sub   = df[df['dataset'] == ds_name].copy()
    valid = sub.dropna(subset=['frequency', 'separability', 'best_F1'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f'{ds_name}  (n={len(valid)} populations)',
                 fontsize=12, fontweight='bold')

    _scatter_ax(ax1, valid, 'frequency',
                'Cell Frequency (%)', 'Frequency vs Best F1')
    _scatter_ax(ax2, valid, 'separability',
                'Separability (centroid dist / intra-std)',
                'Separability vs Best F1')

    if not valid.empty:
        low3 = valid.nsmallest(3, 'frequency')
        for _, row in low3.iterrows():
            ax1.annotate(str(row['population']),
                         (row['frequency'], row['best_F1']),
                         textcoords='offset points', xytext=(5, 3),
                         fontsize=7, color=RARE_COLOR)

    plt.tight_layout()
    path = _os.path.join(FIGURES_DIR, f'separability_scatter_v2_{ds_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def main():
    print("=" * 58)
    print("  可分性-频率诊断分析 v2（Spearman + 分层）")
    print("=" * 58)

    if not _os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found.  Run summarize_results.py first.")
        return

    print("\n  Computing per-population separability...")
    df = build_metrics()
    spearman_analysis(df)

    print("\n  Generating figures...")
    plot_all(df)
    for ds in MULTI_POP_DS:
        plot_single(df, ds)

    print(f"\n  完成。")
    print(f"    {METRICS_CSV}")
    print(f"    {FIGURES_DIR}/separability_scatter_v2_*.png")
    print("=" * 58)


if __name__ == '__main__':
    main()