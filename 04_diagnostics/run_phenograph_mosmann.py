"""
04_diagnostics/run_phenograph_mosmann.py
==================================
补跑 PhenoGraph 在 Mosmann_rare 上的结果。

参数选择说明：
    k=15 来自主脚本（flow_gating_PhenoGraph.py）在 Mosmann_rare 上
    ARI-oracle 扫描的最优近邻数，因此 K_mode 标注为 'K=auto(k=15,ARI-oracle)'。
    这与"手动指定 K"的 oracle 方式不同，是自动扫描后选出的参数。

预计Runtime：约 2-3 小时

输出：
    results/diagnostics/phenograph_mosmann.csv

修改记录（v1.1）：
    - 修正 K_mode 标注（原为误导性的 'K=oracle'）
"""

# ── 路径修复：从子目录 import 根目录模块 ──────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..'
))
# ────────────────────────────────────────────────────

import time
import warnings
import numpy as np
import pandas as pd

from data_loader import load_dataset
from evaluation  import evaluate, print_summary

warnings.filterwarnings('ignore')

DATASET     = 'Mosmann_rare'
K_BEST      = 15     # ARI-oracle 扫描最优近邻数
SEED        = 42
RESULTS_CSV = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'results', 'diagnostics', 'phenograph_mosmann.csv'
)


def main():
    print("=" * 58)
    print(f"  PhenoGraph | {DATASET}  k={K_BEST}  seed={SEED}")
    print(f"  K_mode: K=auto(k={K_BEST}, ARI-oracle)")
    print("=" * 58)

    X, y, _, _ = load_dataset(DATASET, verbose=True)

    print(f"\n  Running (estimated 2-3 hours)...", flush=True)
    t0 = time.time()

    import phenograph
    labels, _, _ = phenograph.cluster(X, k=K_BEST, seed=SEED, n_jobs=1)
    y_pred = labels.astype(int)
    rt     = round(time.time() - t0, 1)

    metrics = evaluate(y, y_pred, dataset_name=DATASET)
    print_summary(metrics, DATASET, 'PhenoGraph')

    row = {
        'dataset'              : DATASET,
        'algorithm'            : 'PhenoGraph',
        # K_mode 说明：k=15 是主脚本 ARI-oracle 扫描的结果，
        # 不是手动指定的 oracle 参数。
        'K_mode'               : f'K=auto(k={K_BEST},ARI-oracle)',
        'best_k'               : K_BEST,
        'seed'                 : SEED,
        'runtime_sec'          : rt,
        'ARI'                  : metrics['ARI'],
        'MacroF1'              : metrics['MacroF1'],
        'accuracy'             : metrics['accuracy'],
        'n_unmatched_cells'    : metrics.get('n_unmatched_cells', None),
        'n_clusters_found'     : metrics['n_clusters_found'],
        'rare_target_f1'       : metrics.get('rare_target_f1'),
        'rare_target_precision': metrics.get('rare_target_precision'),
        'rare_target_recall'   : metrics.get('rare_target_recall'),
        'rare_target_f2'       : metrics.get('rare_target_f2'),
        'rare_target_found'    : metrics.get('rare_target_found'),
    }

    pd.DataFrame([row]).to_csv(RESULTS_CSV, index=False)
    print(f"\n  Results saved: {RESULTS_CSV}")

    print(f"\n{'═'*58}")
    print(f"  activated_F1       = {row['rare_target_f1']}")
    print(f"  activated_Precision= {row['rare_target_precision']}")
    print(f"  activated_Recall   = {row['rare_target_recall']}")
    print(f"  activated_F2       = {row['rare_target_f2']}")
    print(f"  K_found            = {row['n_clusters_found']}")
    print(f"  Runtime            = {rt}s")
    print(f"{'═'*58}")


if __name__ == '__main__':
    main()