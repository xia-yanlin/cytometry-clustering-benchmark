"""
04_diagnostics/verify_ward_f1.py
==================
查验 Ward 在 Levine_13dim / Levine_32dim 上
per-pop F1 热力图里 "F1=0 群体数=0 ✅" 的矛盾。

不跑新实验，只读现有 CSV。
"""
# ── 路径修复：从子目录 import 根目录模块 ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────

import pandas as pd
import numpy as np

_ROOT        = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SUMMARY_CSV  = _os.path.join(_ROOT, 'results', 'summary',   'summary_all.csv')
WARD_CSV     = _os.path.join(_ROOT, 'results', 'baseline',  'ward_results.csv')
HEATMAP_SCRIPT = _os.path.join(_ROOT, '04_diagnostics',
                                'analysis_perpop_f1_heatmap.py')


def check_summary():
    print("=" * 58)
    print("  Ward F1=0 consistency check")
    print("=" * 58)

    if not _os.path.exists(SUMMARY_CSV):
        print(f"❌ 找不到 {SUMMARY_CSV}")
        return

    df   = pd.read_csv(SUMMARY_CSV)
    ward = df[df['algorithm'].str.lower().str.contains('ward', na=False)]

    print(f"\n  Ward rows in summary_all.csv: total={len(ward)}")

    if ward.empty:
        print("  -> Ward has no rows in summary_all.csv")
    else:
        print(ward[['dataset', 'algorithm', 'ARI', 'MacroF1',
                    'n_clusters_found']].to_string(index=False))

    f1_cols = [c for c in df.columns if c.startswith('f1_')]
    print(f"\n  per-pop F1 columns: {len(f1_cols)}")

    for ds in ['Levine_13dim', 'Levine_32dim']:
        sub = ward[ward['dataset'] == ds]
        if sub.empty:
            print(f"\n  {ds}: Ward has no record (N/A as expected)")
        else:
            row    = sub.iloc[0]
            f1_vals = [v for v in [row.get(c, np.nan) for c in f1_cols]
                       if not pd.isna(v)]
            n_zero  = sum(1 for v in f1_vals if v == 0.0)
            print(f"\n  {ds}: Ward has a record (unexpected)")
            print(f"    ARI={row.get('ARI')}  MacroF1={row.get('MacroF1')}")
            print(f"    有效 F1 列数：{len(f1_vals)}  F1=0 群体数：{n_zero}")


def check_ward_csv():
    print(f"\n{'─'*58}")
    if not _os.path.exists(WARD_CSV):
        print(f"  {WARD_CSV} 不存在")
        return

    df = pd.read_csv(WARD_CSV)
    print(f"  ward_results.csv: {len(df)} rows")
    print(df[['dataset', 'algorithm', 'ARI', 'MacroF1',
              'n_clusters_found', 'runtime_sec']].to_string(index=False))

    for ds in ['Levine_13dim', 'Levine_32dim']:
        sub = df[df['dataset'] == ds]
        if sub.empty:
            print(f"\n  {ds}: no record in ward_results.csv (skipped as expected)")
        else:
            row = sub.iloc[0]
            f1_cols = [c for c in df.columns if c.startswith('f1_')]
            f1_vals = {c: row.get(c) for c in f1_cols
                       if not pd.isna(row.get(c, np.nan))}
            n_zero  = sum(1 for v in f1_vals.values() if v == 0.0)
            print(f"\n  {ds}：有记录 ⚠️  ARI={row.get('ARI')}  "
                  f"F1=0群体数：{n_zero}")


def check_heatmap_script():
    print(f"\n{'─'*58}")
    if not _os.path.exists(HEATMAP_SCRIPT):
        print(f"  Heatmap script not found: {HEATMAP_SCRIPT}")
        return

    with open(HEATMAP_SCRIPT, 'r', encoding='utf-8', errors='replace') as f:
        code = f.read()

    suspects = ['fillna(0)', 'fill_value=0', 'fillna(0.0)',
                'replace(nan', 'nan_to_num']
    found = [s for s in suspects if s in code]

    if found:
        print(f"  ⚠️  Heatmap script has suspect NaN->0 fill: {found}")
    else:
        print(f"  Heatmap script: no NaN->0 fill detected")

    # 检查是否有 N/A 单独处理逻辑
    if 'all_nan' in code or 'isna().all()' in code:
        print(f"  Heatmap script: N/A rows handled correctly")
    else:
        print(f"  WARNING: heatmap script may not handle all-NaN rows（Ward 跳过的数据集）")


def main():
    check_summary()
    check_ward_csv()
    check_heatmap_script()

    print(f"\n{'═'*58}")
    print("  Conclusion:")
    print("  Ward results for Levine_13dim/32dim should be N/A (skipped).")
    print("  If the heatmap reports '0 populations with F1=0', that is a logic error:")
    print("  N/A rows must be annotated separately and excluded from all comparisons.")
    print(f"{'═'*58}")


if __name__ == '__main__':
    main()