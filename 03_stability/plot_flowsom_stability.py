"""
03_stability/plot_flowsom_stability.py
==============================
FlowSOM bimodal convergence visualisation

读取 multirun_raw.csv（来自 flow_gating_stochastic_multirun.py），
对 FlowSOM 在 Levine_32dim 和 Samusik_01 两个数据集上的
10 次 ARI 分布绘制核密度估计图（KDE），直观展示双峰现象。

同时绘制三算法（k-means / GMM / FlowSOM）的稳定性对比图，
突出 FlowSOM 的高方差问题。

输入：
    results/multirun_raw.csv        (150行，3×5×10)
    results/multirun_summary.csv    (均值±std，可选)

输出：
    figures/flowsom_bimodal_kde.png      双峰 KDE 核心图（论文主图）
    figures/flowsom_stability_compare.png 三算法稳定性对比
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
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

RAW_CSV     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'stability', 'multirun_raw.csv')
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 颜色方案
COLORS = {
    'k-means' : '#2196F3',
    'GMM'     : '#FF9800',
    'FlowSOM' : '#E53935',
}
DS_LABELS = {
    'Levine_13dim': 'Levine 13-dim',
    'Levine_32dim': 'Levine 32-dim',
    'Samusik_01'  : 'Samusik 01',
    'Nilsson_rare': 'Nilsson rare',
    'Mosmann_rare': 'Mosmann rare',
}


def load_data():
    if not os.path.exists(RAW_CSV):
        print(f"ERROR: file not found: {RAW_CSV}")
        print("   Run 03_stability/run_stochastic_multirun.py first")
        return None
    df = pd.read_csv(RAW_CSV)
    print(f"  Loaded {RAW_CSV}: {len(df)} rows")
    print(f"  Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    return df


# ════════════════════════════════════════════════════════════
# 图1：FlowSOM 双峰 KDE（论文主图）
# ════════════════════════════════════════════════════════════

def plot_bimodal_kde(df):
    """
    专门展示 FlowSOM 在 Levine_32dim 和 Samusik_01 上的
    ARI 分布双峰现象，与 k-means 的单峰稳定分布对比。
    """
    focus_ds    = ['Levine_32dim', 'Samusik_01']
    focus_algos = ['FlowSOM', 'k-means']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        'FlowSOM Bimodal Convergence vs. k-means Stability\n'
        '(10 random seeds per condition)',
        fontsize=13, fontweight='bold'
    )

    for ax, ds in zip(axes, focus_ds):
        ax.set_title(DS_LABELS.get(ds, ds), fontsize=11, fontweight='bold')

        for algo in focus_algos:
            sub  = df[(df['algorithm'] == algo) & (df['dataset'] == ds)]
            if sub.empty:
                continue
            vals = sub['ARI'].dropna().values
            c    = COLORS.get(algo, '#607D8B')
            mean = vals.mean()
            std  = vals.std()

            # 散点抖动（jitter）
            jitter = np.random.default_rng(42).uniform(-0.03, 0.03, len(vals))
            y_pos  = 0.15 if algo == 'FlowSOM' else 0.05
            ax.scatter(vals, np.full(len(vals), y_pos) + jitter,
                       c=c, s=60, alpha=0.7, zorder=4,
                       label=f'{algo} (μ={mean:.3f}, σ={std:.3f})')

            # KDE 曲线
            if len(vals) >= 4:
                try:
                    kde  = gaussian_kde(vals, bw_method=0.3)
                    x_g  = np.linspace(max(0, vals.min()-0.05),
                                       min(1, vals.max()+0.05), 300)
                    y_g  = kde(x_g)
                    # 归一化到 [0, 0.6] 区间以便叠加显示
                    y_g  = y_g / y_g.max() * 0.55
                    ax.plot(x_g, y_g + 0.2, color=c, linewidth=2.2,
                            alpha=0.9)
                    ax.fill_between(x_g, 0.2, y_g + 0.2,
                                    color=c, alpha=0.12)
                except Exception:
                    pass

            # 均值虚线
            ax.axvline(mean, color=c, linestyle='--',
                       linewidth=1.5, alpha=0.7)

        # 双峰区域标注（仅 FlowSOM）
        flowsom_sub = df[(df['algorithm'] == 'FlowSOM') &
                         (df['dataset'] == ds)]
        if not flowsom_sub.empty:
            vals_fs = flowsom_sub['ARI'].dropna().values
            # 简单双峰判断：检查是否有超过 0.08 ARI 的间距
            sorted_v = np.sort(vals_fs)
            gaps     = np.diff(sorted_v)
            if gaps.max() > 0.08:
                gap_idx = gaps.argmax()
                split   = (sorted_v[gap_idx] + sorted_v[gap_idx+1]) / 2
                lo_mean = sorted_v[:gap_idx+1].mean()
                hi_mean = sorted_v[gap_idx+1:].mean()
                n_lo    = gap_idx + 1
                n_hi    = len(sorted_v) - n_lo

                ax.axvspan(sorted_v[0] - 0.01, split,
                           alpha=0.07, color='#E53935',
                           label=f'Low mode (n={n_lo})')
                ax.axvspan(split, sorted_v[-1] + 0.01,
                           alpha=0.07, color='#B71C1C',
                           label=f'High mode (n={n_hi})')
                ax.annotate(
                    f'Low\nμ={lo_mean:.3f}',
                    xy=(lo_mean, 0.75), ha='center', fontsize=8,
                    color='#E53935',
                    arrowprops=dict(arrowstyle='->', color='#E53935'),
                    xytext=(lo_mean - 0.07, 0.85)
                )
                ax.annotate(
                    f'High\nμ={hi_mean:.3f}',
                    xy=(hi_mean, 0.75), ha='center', fontsize=8,
                    color='#B71C1C',
                    arrowprops=dict(arrowstyle='->', color='#B71C1C'),
                    xytext=(hi_mean + 0.05, 0.85)
                )

        ax.set_xlabel('ARI', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, axis='x')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'flowsom_bimodal_kde.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════
# 图2：三算法 × 三数据集 稳定性矩阵（ARI 分布条带图）
# ════════════════════════════════════════════════════════════

def plot_stability_matrix(df):
    """
    三算法 × 三多群体数据集的 ARI 分布，
    用小提琴图 + 散点叠加展示，方差对比一目了然。
    """
    algos      = ['k-means', 'GMM', 'FlowSOM']
    multi_ds   = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']
    sub        = df[df['dataset'].isin(multi_ds) &
                    df['algorithm'].isin(algos)]

    if sub.empty:
        print("  WARNING: no multi-population data in multirun_raw.csv, skipping stability matrix")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        'Clustering Stability: ARI Distribution over 10 Random Seeds',
        fontsize=12, fontweight='bold'
    )

    for ax, ds in zip(axes, multi_ds):
        data_list  = []
        tick_labels = []
        colors_list = []

        for algo in algos:
            vals = sub[(sub['algorithm'] == algo) &
                       (sub['dataset'] == ds)]['ARI'].dropna().values
            if len(vals) > 0:
                data_list.append(vals)
                tick_labels.append(algo)
                colors_list.append(COLORS.get(algo, '#607D8B'))

        if not data_list:
            ax.set_visible(False)
            continue

        # 小提琴图
        parts = ax.violinplot(data_list, positions=range(len(data_list)),
                              showmeans=True, showmedians=False,
                              widths=0.6)
        for pc, c in zip(parts['bodies'], colors_list):
            pc.set_facecolor(c)
            pc.set_alpha(0.45)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.8)

        # 散点叠加
        for i, (vals, c) in enumerate(zip(data_list, colors_list)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08,
                                                        len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       c=c, s=35, alpha=0.85, zorder=3)

        # std 标注
        for i, vals in enumerate(data_list):
            std = vals.std()
            clr = '#E53935' if std > 0.05 else '#388E3C'
            ax.text(i, vals.max() + 0.02, f'σ={std:.3f}',
                    ha='center', fontsize=8, color=clr, fontweight='bold')

        ax.set_title(DS_LABELS.get(ds, ds), fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel('ARI' if ds == 'Levine_13dim' else '', fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3, axis='y')

        # FlowSOM 高方差警告框
        fs_vals = sub[(sub['algorithm'] == 'FlowSOM') &
                      (sub['dataset'] == ds)]['ARI'].dropna().values
        if len(fs_vals) > 0 and fs_vals.std() > 0.05:
            ax.text(0.97, 0.03, '⚠ Bimodal',
                    transform=ax.transAxes, ha='right', fontsize=8,
                    color='#E53935',
                    bbox=dict(boxstyle='round', facecolor='#FFEBEE',
                              edgecolor='#E53935', alpha=0.9))

    # 图例
    patches = [mpatches.Patch(color=COLORS[a], label=a, alpha=0.7)
               for a in algos]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'flowsom_stability_compare.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════
# 图3：单次运行 vs 多次均值——"幸运运行"量化
# ════════════════════════════════════════════════════════════

def plot_lucky_run_gap(df):
    """
    柱状图对比：原始单次运行的 ARI 与 10 次均值的差距，
    直接量化"幸运运行"偏差。
    """
    # 原始单次结果（来自实验记录）
    single_run = {
        ('FlowSOM', 'Levine_13dim'): 0.8936,
        ('FlowSOM', 'Levine_32dim'): 0.9240,
        ('FlowSOM', 'Samusik_01')  : 0.9032,
        ('k-means', 'Levine_13dim'): 0.5622,
        ('k-means', 'Levine_32dim'): 0.5725,
        ('k-means', 'Samusik_01')  : 0.4979,
    }

    multi_ds = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']
    algos    = ['k-means', 'FlowSOM']

    rows = []
    for algo in algos:
        for ds in multi_ds:
            vals = df[(df['algorithm'] == algo) &
                      (df['dataset'] == ds)]['ARI'].dropna().values
            if len(vals) == 0:
                continue
            mean = vals.mean()
            std  = vals.std()
            single = single_run.get((algo, ds), mean)
            rows.append({
                'algo'  : algo,
                'ds'    : ds,
                'single': single,
                'mean'  : mean,
                'std'   : std,
                'gap'   : single - mean,
            })

    if not rows:
        return

    gap_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(
        '"Lucky Run" Bias: Single-Run ARI vs. 10-Run Mean\n'
        '(positive gap = single run overestimates true performance)',
        fontsize=11, fontweight='bold'
    )

    x      = np.arange(len(gap_df))
    colors = [COLORS.get(r['algo'], '#607D8B') for _, r in gap_df.iterrows()]

    bars = ax.bar(x, gap_df['gap'], color=colors, alpha=0.75, width=0.6)

    # 误差棒（±std）
    ax.errorbar(x, gap_df['gap'], yerr=gap_df['std'],
                fmt='none', color='black', capsize=4, linewidth=1.5)

    # 零线
    ax.axhline(0, color='black', linewidth=1)

    # 数值标注
    for i, (_, row) in enumerate(gap_df.iterrows()):
        ax.text(i, row['gap'] + (0.01 if row['gap'] >= 0 else -0.03),
                f"{row['gap']:+.3f}", ha='center', fontsize=8,
                fontweight='bold',
                color='#E53935' if row['gap'] > 0.05 else '#333333')

    tick_labels = [f"{r['algo']}\n{DS_LABELS.get(r['ds'], r['ds'])}"
                   for _, r in gap_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel('ARI gap (single − mean)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    patches = [mpatches.Patch(color=COLORS[a], label=a, alpha=0.75)
               for a in algos]
    ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'flowsom_lucky_run_gap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  FlowSOM bimodal convergence visualisation")
    print("=" * 58)

    df = load_data()
    if df is None:
        return

    print("\n  Generating figure 1: bimodal KDE...")
    plot_bimodal_kde(df)

    print("  Generating figure 2: three-algorithm stability matrix...")
    plot_stability_matrix(df)

    print("  Generating figure 3: lucky-run bias quantification...")
    plot_lucky_run_gap(df)

    print(f"\n  Done. Output files:")
    print(f"    figures/flowsom_bimodal_kde.png")
    print(f"    figures/flowsom_stability_compare.png")
    print(f"    figures/flowsom_lucky_run_gap.png")
    print("=" * 58)

    # 终端打印关键数字
    print("\n  Key numbers:")
    fs_datasets = ['Levine_32dim', 'Samusik_01']
    for ds in fs_datasets:
        sub = df[(df['algorithm'] == 'FlowSOM') & (df['dataset'] == ds)]
        if sub.empty:
            continue
        vals = sub['ARI'].dropna().values
        print(f"    FlowSOM | {ds}")
        print(f"      n={len(vals)}  mean={vals.mean():.4f}  "
              f"std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")
        # 双峰判断
        sorted_v = np.sort(vals)
        gaps     = np.diff(sorted_v)
        if len(gaps) > 0 and gaps.max() > 0.08:
            idx   = gaps.argmax()
            split = (sorted_v[idx] + sorted_v[idx+1]) / 2
            lo    = sorted_v[:idx+1]
            hi    = sorted_v[idx+1:]
            print(f"      ⚠️  双峰检测：split≈{split:.3f}  "
                  f"低峰(n={len(lo)})={lo.mean():.3f}  "
                  f"high-peak(n={len(hi)})={hi.mean():.3f}")
        else:
            print(f"      No clear bimodal (max gap={gaps.max():.3f})")


if __name__ == '__main__':
    main()
