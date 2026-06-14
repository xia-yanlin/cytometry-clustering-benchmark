"""
04_diagnostics/perpop_f1_heatmap.py  (v2 - fixed paths)
==================================================
读取 results/summary/summary_all.csv，绘制 per-population F1 热力图。

输出：
    figures/heatmap_Levine_13dim.png
    figures/heatmap_Levine_32dim.png
    figures/heatmap_Samusik_01.png
"""
# ── 路径修复：从子目录 import 根目录模块 ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

_ROOT       = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SUMMARY_CSV = _os.path.join(_ROOT, 'results', 'summary', 'summary_all.csv')
FIGURES_DIR = _os.path.join(_ROOT, 'figures')
_os.makedirs(FIGURES_DIR, exist_ok=True)

MULTI_POP_DS = ['Levine_13dim', 'Levine_32dim', 'Samusik_01']

ALGO_ORDER = ['k-means', 'GMM', 'Ward', 'FlowSOM',
              'PhenoGraph', 'Leiden', 'HDBSCAN']

ALGO_LABELS = {
    'k-means'   : 'k-means',
    'GMM'       : 'GMM',
    'Ward'      : 'Ward',
    'FlowSOM'   : 'FlowSOM',
    'PhenoGraph': 'PhenoGraph',
    'Leiden'    : 'Leiden',
    'HDBSCAN'   : 'HDBSCAN',
}

CMAP = mcolors.LinearSegmentedColormap.from_list(
    'f1_cmap',
    [(0.0, '#d73027'), (0.3, '#fee08b'),
     (0.6, '#91cf60'), (1.0, '#1a7837')],
)


def extract_perpop_f1(df, dataset_name):
    sub = df[df['dataset'] == dataset_name].copy()
    if sub.empty:
        print(f"  WARNING: no rows for {dataset_name}")
        return None, None

    all_f1_cols = [c for c in sub.columns if c.startswith('f1_')]
    if not all_f1_cols:
        print(f"  WARNING: no f1_ columns in {dataset_name}")
        return None, None

    # 只保留当前数据集有实际值的列（排除其他数据集的群体列）
    valid_f1_cols = [c for c in all_f1_cols if sub[c].notna().any()]

    sub = sub.set_index('algorithm')
    present = [a for a in ALGO_ORDER if a in sub.index]
    matrix  = sub.loc[present, valid_f1_cols].astype(float).copy()
    matrix.columns = [c[3:] for c in matrix.columns]
    return matrix, present


def get_pop_sizes(dataset_name):
    try:
        from data_loader import load_dataset
        _, y, _, _ = load_dataset(dataset_name, verbose=False)
        return {str(p): int(np.sum(y == p)) for p in np.unique(y)}
    except Exception as e:
        print(f"  WARNING: could not load sizes for {dataset_name}: {e}")
        return {}


def plot_heatmap(matrix, pop_sizes, dataset_name, out_path):
    pops = list(matrix.columns)

    if pop_sizes:
        sizes = [pop_sizes.get(str(p), 0) for p in pops]
        order = np.argsort(sizes)
        pops   = [pops[i] for i in order]
        matrix = matrix[pops]

    n_algo = len(matrix)
    n_pop  = len(pops)

    fig_w = max(12, n_pop * 0.75)
    fig_h = n_algo * 0.65 + 2.8
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs    = GridSpec(2, 1, height_ratios=[n_algo, 2.0],
                     hspace=0.06, figure=fig)

    ax_heat = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    data = matrix.values.astype(float)
    im   = ax_heat.imshow(data, aspect='auto', cmap=CMAP,
                          vmin=0, vmax=1, interpolation='nearest')

    ax_heat.set_yticks(range(n_algo))
    ax_heat.set_yticklabels(
        [ALGO_LABELS.get(a, a) for a in matrix.index],
        fontsize=9, fontweight='bold'
    )
    ax_heat.set_xticks([])

    for i in range(n_algo):
        for j in range(n_pop):
            val = data[i, j]
            if np.isnan(val):
                txt, col = 'N/A', '#aaaaaa'
            else:
                txt = f'{val:.2f}'
                col = 'white' if (val > 0.62 or val < 0.15) else 'black'
            ax_heat.text(j, i, txt, ha='center', va='center',
                         fontsize=6.5, color=col)

    ax_heat.set_title(
        f'Per-population F1  —  {dataset_name}  '
        f'(sorted by cell count, rarest on left)',
        fontsize=11, pad=8
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)
    cbar.set_label('F1 score', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    heights = [pop_sizes.get(str(p), 0) for p in pops]
    colors  = ['#e74c3c' if h < 1000 else '#3498db' for h in heights]

    ax_bar.bar(range(n_pop), heights, color=colors,
               width=0.7, edgecolor='white', linewidth=0.3)
    ax_bar.set_xticks(range(n_pop))
    ax_bar.set_xticklabels(pops, rotation=45, ha='right', fontsize=7.5)
    ax_bar.set_ylabel('Cell count', fontsize=8)
    ax_bar.set_xlim(-0.5, n_pop - 0.5)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}')
    )
    ax_bar.tick_params(axis='y', labelsize=7)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[Patch(facecolor='#e74c3c', label='< 1,000 cells (rare)'),
                 Patch(facecolor='#3498db', label='>= 1,000 cells')],
        fontsize=7, loc='upper right', framealpha=0.8
    )

    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_zero_f1_summary(matrix, dataset_name):
    print(f"\n  [{dataset_name}] populations with F1=0 (completely missed):")
    for algo in matrix.index:
        zeros = [p for p in matrix.columns
                 if not np.isnan(matrix.loc[algo, p])
                 and matrix.loc[algo, p] == 0.0]
        label = ALGO_LABELS.get(algo, algo)
        # 注意：若行全为 NaN（如 Ward 跳过），不计入统计，标注 N/A
        all_nan = matrix.loc[algo].isna().all()
        if all_nan:
            print(f"    {label:<12}  N/A (dataset skipped, not included in stats)")
        elif zeros:
            print(f"    {label:<12}  {len(zeros)} populations")
        else:
            print(f"    {label:<12}  0 populations")


def main():
    print("=" * 65)
    print("  Per-population F1 Heatmap Generator  (v2)")
    print("=" * 65)

    if not _os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found.  Run summarize_results.py first.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {SUMMARY_CSV}: {len(df)} rows")

    for ds_name in MULTI_POP_DS:
        print(f"\n{'─' * 65}")
        print(f"  Dataset: {ds_name}")

        matrix, present = extract_perpop_f1(df, ds_name)
        if matrix is None:
            continue

        print(f"  Algorithms: {len(present)}   "
              f"Populations: {len(matrix.columns)}")

        pop_sizes = get_pop_sizes(ds_name)
        if pop_sizes:
            smallest_pop = min(pop_sizes, key=pop_sizes.get)
            pct = pop_sizes[smallest_pop] / sum(pop_sizes.values()) * 100
            print(f"  Smallest: {smallest_pop}  "
                  f"({pop_sizes[smallest_pop]:,} cells, {pct:.2f}%)")

        print_zero_f1_summary(matrix, ds_name)

        out_path = _os.path.join(FIGURES_DIR, f'heatmap_{ds_name}.png')
        plot_heatmap(matrix, pop_sizes, ds_name, out_path)

    print(f"\n  Done. Figures saved to {FIGURES_DIR}/")
    print("=" * 65)


if __name__ == '__main__':
    main()