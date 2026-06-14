"""
04_diagnostics/flowsom_dip_test.py
=============================
对 Formal statistical test for FlowSOM bimodal convergence。

检验方法：
  1. Hartigan Dip Test（单峰 vs 多峰）
  2. 双组分 GMM vs 单组分 GMM（BIC 比较）

Multiple-testing note:
  本脚本对 3 个dataset分别做 Dip Test，存在多重检验问题。
  Bonferroni 校正后显著性阈值为 0.05/3 = 0.017。
  若某dataset Dip Test 在校正后不显著，但 ΔBIC >> 10（贝叶斯强证据），
  结论仍成立。论文中以 ΔBIC 为主要证据，Dip Test p 值作辅助，并注明校正情况。

Input: results/stability/multirun_raw.csv

输出：
    results/diagnostics/flowsom_dip_test.csv
    figures/flowsom_dip_annotated.png
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
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

_ROOT   = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
RAW_CSV = _os.path.join(_ROOT, 'results', 'stability',    'multirun_raw.csv')
OUT_CSV = _os.path.join(_ROOT, 'results', 'diagnostics',  'flowsom_dip_test.csv')
OUT_FIG = _os.path.join(_ROOT, 'figures', 'flowsom_dip_annotated.png')

# Bonferroni 校正阈值（3个dataset）
BONFERRONI_ALPHA = 0.05 / 3   # = 0.0167

_os.makedirs(_os.path.join(_ROOT, 'figures'), exist_ok=True)


def hartigan_dip(x):
    try:
        import diptest
        dip, pval = diptest.diptest(np.asarray(x, dtype=float))
        return float(dip), float(pval)
    except ImportError:
        raise ImportError(
            "\n\n❌  diptest 未安装，无法计算 Hartigan Dip Test。\n"
            "   原备用实现计算的是 KS 统计量而非 Dip 统计量，p 值完全不可信。\n"
            "   请先安装：pip install diptest\n"
            "   然后重新运行本脚本。\n"
        )


def gmm_bic(x, n_components):
    x_arr = np.asarray(x).reshape(-1, 1)
    gm    = GaussianMixture(n_components=n_components,
                             random_state=42, n_init=5)
    gm.fit(x_arr)
    return gm.bic(x_arr)


def analyze_dataset(algo, ds, vals):
    n    = len(vals)
    dip, p_dip = hartigan_dip(vals)

    bic1      = gmm_bic(vals, 1)
    bic2      = gmm_bic(vals, 2) if n >= 4 else np.nan
    delta_bic = bic1 - bic2 if not np.isnan(bic2) else np.nan

    sorted_v = np.sort(vals)
    gaps     = np.diff(sorted_v)
    max_gap  = gaps.max() if len(gaps) > 0 else 0

    if len(gaps) > 0 and max_gap > 0.08:
        idx     = gaps.argmax()
        split   = (sorted_v[idx] + sorted_v[idx + 1]) / 2
        lo_mean = sorted_v[:idx+1].mean()
        hi_mean = sorted_v[idx+1:].mean()
        n_lo    = idx + 1
        n_hi    = n - n_lo
    else:
        split = lo_mean = hi_mean = np.nan
        n_lo  = n_hi = 0

    # 双峰判断：Bonferroni 校正后的 Dip Test 或强 BIC 证据
    p_corrected  = p_dip * 3   # Bonferroni 校正后等效 p 值
    bimodal_stat = bool(
        p_dip < BONFERRONI_ALPHA or
        (not np.isnan(delta_bic) and delta_bic > 10)
    )

    return {
        'algorithm'        : algo,
        'dataset'          : ds,
        'n_runs'           : n,
        'mean'             : round(vals.mean(), 4),
        'std'              : round(vals.std(),  4),
        'dip_stat'         : round(dip,   4),
        'dip_pvalue'       : round(p_dip, 4),
        'dip_pvalue_bonf'  : round(min(p_corrected, 1.0), 4),
        'bonferroni_alpha' : round(BONFERRONI_ALPHA, 4),
        'bic_1comp'        : round(bic1,  2),
        'bic_2comp'        : round(bic2,  2) if not np.isnan(bic2) else np.nan,
        'delta_bic'        : round(delta_bic, 2) if not np.isnan(delta_bic) else np.nan,
        'max_gap'          : round(max_gap, 4),
        'bimodal_stat'     : bimodal_stat,
        'split'            : round(split,   4) if not np.isnan(split)   else np.nan,
        'lo_mean'          : round(lo_mean, 4) if not np.isnan(lo_mean) else np.nan,
        'hi_mean'          : round(hi_mean, 4) if not np.isnan(hi_mean) else np.nan,
        'n_lo'             : n_lo,
        'n_hi'             : n_hi,
    }


def plot_annotated(df_raw, df_results):
    focus_ds    = ['Levine_32dim', 'Samusik_01']
    focus_algos = ['FlowSOM', 'k-means']
    COLORS      = {'FlowSOM': '#E53935', 'k-means': '#1976D2'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'FlowSOM Bimodal Convergence — Statistical Validation\n'
        '(Hartigan Dip Test + GMM BIC, Bonferroni-corrected α=0.017)',
        fontsize=13, fontweight='bold'
    )

    for ax, ds in zip(axes, focus_ds):
        ax.set_title(ds.replace('_', ' '), fontsize=11, fontweight='bold')

        for algo in focus_algos:
            sub  = df_raw[(df_raw['algorithm'] == algo) &
                          (df_raw['dataset']   == ds)]
            if sub.empty and algo == 'k-means':
                sub = df_raw[(df_raw['algorithm'] == 'kmeans') &
                             (df_raw['dataset']   == ds)]
            vals = sub['ARI'].dropna().values
            if len(vals) == 0:
                continue

            c    = COLORS.get(algo, '#607D8B')
            mean = vals.mean()
            std  = vals.std()

            jitter = np.random.default_rng(42).uniform(-0.025, 0.025, len(vals))
            y_base = 0.12 if algo == 'FlowSOM' else 0.04
            ax.scatter(vals,
                       np.full(len(vals), y_base) + jitter,
                       c=c, s=65, alpha=0.75, zorder=4,
                       label=f'{algo} μ={mean:.3f} σ={std:.3f}')

            if len(vals) >= 4:
                try:
                    kde = gaussian_kde(vals, bw_method=0.3)
                    xg  = np.linspace(max(0, vals.min()-0.05),
                                      min(1, vals.max()+0.05), 300)
                    yg  = kde(xg) / kde(xg).max() * 0.55
                    ax.plot(xg, yg + 0.2, color=c, linewidth=2.2)
                    ax.fill_between(xg, 0.2, yg + 0.2, color=c, alpha=0.12)
                except Exception:
                    pass

            ax.axvline(mean, color=c, linestyle='--',
                       linewidth=1.4, alpha=0.6)

        res_row = df_results[
            (df_results['algorithm'] == 'FlowSOM') &
            (df_results['dataset']   == ds)
        ]
        if not res_row.empty:
            r       = res_row.iloc[0]
            p_raw   = r['dip_pvalue']
            p_bonf  = r['dip_pvalue_bonf']
            dbic    = r['delta_bic']
            bimodal = r['bimodal_stat']

            sig_raw  = f'p={p_raw:.3f}'
            sig_bonf = f'Bonf: p={p_bonf:.3f}'
            bic_str  = (f'ΔBIC={dbic:.1f} ✓' if (not np.isnan(dbic) and dbic > 10)
                        else f'ΔBIC={dbic:.1f}' if not np.isnan(dbic) else '')
            conclusion  = '双峰 ✅' if bimodal else '单峰'
            box_color   = '#FFEBEE' if bimodal else '#E8F5E9'
            edge_color  = '#E53935' if bimodal else '#388E3C'

            ax.text(0.97, 0.97,
                    f'Dip: {sig_raw}\n{sig_bonf}\n{bic_str}\n→ {conclusion}',
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor=box_color,
                              edgecolor=edge_color, alpha=0.95))

            if not np.isnan(r['split']):
                ax.axvline(r['split'], color='#E53935',
                           linestyle=':', linewidth=1.5, alpha=0.5)

        ax.set_xlabel('ARI', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
        ax.grid(True, alpha=0.25, axis='x')

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUT_FIG}")


def main():
    print("=" * 62)
    print("  FlowSOM 双峰统计检验（Dip Test + GMM BIC）")
    print(f"  Bonferroni 校正阈值：{BONFERRONI_ALPHA:.4f}（0.05/3）")
    print("=" * 62)

    if not _os.path.exists(RAW_CSV):
        print(f"❌ 找不到 {RAW_CSV}")
        return

    df = pd.read_csv(RAW_CSV)
    df['algorithm'] = df['algorithm'].replace({'kmeans': 'k-means'})

    focus = [
        ('FlowSOM', 'Levine_13dim'),
        ('FlowSOM', 'Levine_32dim'),
        ('FlowSOM', 'Samusik_01'),
        ('k-means', 'Levine_13dim'),
        ('k-means', 'Levine_32dim'),
        ('k-means', 'Samusik_01'),
        ('GMM',     'Levine_13dim'),
        ('GMM',     'Levine_32dim'),
        ('GMM',     'Samusik_01'),
    ]

    rows = []
    print(f"\n  {'algorithm':<12} {'dataset':<18} {'Dip p':>8} "
          f"{'Bonf p':>8} {'ΔBIC':>8} {'bimodal':>6} {'split':>7}")
    print(f"  {'─'*66}")

    for algo, ds in focus:
        sub  = df[(df['algorithm'] == algo) & (df['dataset'] == ds)]
        vals = sub['ARI'].dropna().values
        if len(vals) < 3:
            continue

        r = analyze_dataset(algo, ds, vals)
        rows.append(r)

        bimodal_str = '✅ 双峰' if r['bimodal_stat'] else '  单峰'
        split_str   = f"{r['split']:.3f}" if not np.isnan(r['split']) else '  —  '
        dbic_str    = (f"{r['delta_bic']:>8.2f}"
                       if not np.isnan(r['delta_bic']) else '     nan')
        print(f"  {algo:<12} {ds:<18} "
              f"{r['dip_pvalue']:>8.4f} "
              f"{r['dip_pvalue_bonf']:>8.4f} "
              f"{dbic_str} "
              f"{bimodal_str:>8}  {split_str:>7}")

    df_results = pd.DataFrame(rows)
    _os.makedirs(_os.path.dirname(OUT_CSV), exist_ok=True)
    df_results.to_csv(OUT_CSV, index=False)
    print(f"\n  Results saved: {OUT_CSV}")
    print(f"\n  Generating annotated KDE plot...")
    plot_annotated(df, df_results)

    print(f"\n{'═'*62}")
    print("  Statistical conclusions (Bonferroni-corrected):")
    for _, r in df_results[df_results['algorithm'] == 'FlowSOM'].iterrows():
        ds      = r['dataset']
        bimodal = r['bimodal_stat']
        p_raw   = r['dip_pvalue']
        p_bonf  = r['dip_pvalue_bonf']
        dbic    = r['delta_bic']
        if bimodal:
            print(f"  FlowSOM | {ds:<18} → Bimodal  "
                  f"(Dip p={p_raw:.4f}, Bonf p={p_bonf:.4f}, "
                  f"ΔBIC={dbic:.1f})")
        else:
            print(f"  FlowSOM | {ds:<18} → Insufficient evidence  "
                  f"(Dip p={p_raw:.4f}, Bonf p={p_bonf:.4f}, "
                  f"ΔBIC={dbic:.1f})")

    # 动态输出 Levine_32dim 的补充说明（不再硬编码 ΔBIC 数值）
    lev32 = df_results[
        (df_results['algorithm'] == 'FlowSOM') &
        (df_results['dataset']   == 'Levine_32dim')
    ]
    if not lev32.empty:
        r32 = lev32.iloc[0]
        p_bonf32 = r32['dip_pvalue_bonf']
        dbic32   = r32['delta_bic']
        if p_bonf32 > BONFERRONI_ALPHA:
            print(f"\n  Note: Levine_32dim Dip p(Bonf)={p_bonf32:.4f}"
                  f" not significant after Bonferroni correction (threshold {BONFERRONI_ALPHA:.4f}),")
            if not np.isnan(dbic32) and dbic32 > 10:
                print(f"      但 ΔBIC={dbic32:.2f} >> 10，"
                      f"贝叶斯证据非常强，双峰结论仍成立。")
            print(f"      ΔBIC is the primary evidence in the paper; Bonferroni correction is reported.")
        else:
            print(f"\n  Note: Levine_32dim Dip p(Bonf)={p_bonf32:.4f}"
                  f"（n={r32['n_runs']} runs), significant after Bonferroni correction,"
                  f"ΔBIC={dbic32:.2f}。")
    print(f"{'═'*62}")

    try:
        import diptest
        print("\n  Using diptest package")
    except ImportError:
        print("\n  ERROR: diptest not installed. Install with: pip install diptest")


if __name__ == '__main__':
    main()