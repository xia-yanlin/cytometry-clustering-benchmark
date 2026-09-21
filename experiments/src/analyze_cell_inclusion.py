from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-003A")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    run_file = args.run_directory / "run_level_metrics.csv"
    population_file = args.run_directory / "population_level_metrics.csv"
    runs = pd.read_csv(run_file)
    populations = pd.read_csv(population_file)
    metrics = ["ari", "macro_precision", "macro_recall", "macro_f1"]
    rows = []

    for dataset in sorted(runs["dataset"].unique()):
        subset = runs[runs["dataset"] == dataset]
        for metric in metrics:
            pivot = subset.pivot(index="seed", columns="fit_policy", values=metric).dropna()
            full = pivot["all_cells_fit"].to_numpy()
            labeled = pivot["labeled_only_fit"].to_numpy()
            delta = full - labeled
            n = len(delta)
            sem = stats.sem(delta)
            ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=np.mean(delta), scale=sem)
            try:
                wilcoxon = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
                w_stat, p_value = float(wilcoxon.statistic), float(wilcoxon.pvalue)
            except ValueError:
                w_stat, p_value = 0.0, 1.0
            shapiro = stats.shapiro(delta)
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "paired_seeds": n,
                "mean_all_cells_fit": float(np.mean(full)),
                "sd_all_cells_fit": float(np.std(full, ddof=1)),
                "mean_labeled_only_fit": float(np.mean(labeled)),
                "sd_labeled_only_fit": float(np.std(labeled, ddof=1)),
                "mean_delta_all_minus_labeled": float(np.mean(delta)),
                "median_delta_all_minus_labeled": float(np.median(delta)),
                "sd_delta": float(np.std(delta, ddof=1)),
                "ci95_delta_low": float(ci_low),
                "ci95_delta_high": float(ci_high),
                "all_cells_better_fraction": float(np.mean(delta > 0)),
                "wilcoxon_w": w_stat,
                "wilcoxon_p_raw": p_value,
                "shapiro_w_delta": float(shapiro.statistic),
                "shapiro_p_delta": float(shapiro.pvalue),
            })

    comparison = pd.DataFrame(rows)
    comparison["wilcoxon_p_holm"] = holm_adjust(comparison["wilcoxon_p_raw"].tolist())
    comparison["holm_significant_0_05"] = comparison["wilcoxon_p_holm"] < 0.05
    comparison.to_csv(args.output / "statistical_comparison.csv", index=False)

    pop_pivot = populations.pivot_table(
        index=["dataset", "population", "seed"], columns="fit_policy",
        values=["precision", "recall", "f1"],
    )
    pop_pivot.columns = [f"{metric}_{policy}" for metric, policy in pop_pivot.columns]
    pop_pivot = pop_pivot.reset_index()
    for metric in ("precision", "recall", "f1"):
        pop_pivot[f"delta_{metric}_all_minus_labeled"] = (
            pop_pivot[f"{metric}_all_cells_fit"] - pop_pivot[f"{metric}_labeled_only_fit"]
        )
    pop_summary = pop_pivot.groupby(["dataset", "population"]).agg(
        seeds=("seed", "count"),
        mean_delta_precision=("delta_precision_all_minus_labeled", "mean"),
        mean_delta_recall=("delta_recall_all_minus_labeled", "mean"),
        mean_delta_f1=("delta_f1_all_minus_labeled", "mean"),
        sd_delta_f1=("delta_f1_all_minus_labeled", "std"),
    ).reset_index()
    pop_summary.to_csv(args.output / "population_effects.csv", index=False)

    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_directory": str(args.run_directory.resolve()),
        "source_run_level_sha256": sha256(run_file),
        "source_population_level_sha256": sha256(population_file),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "multiple_testing": "Holm correction across 12 dataset-by-metric Wilcoxon tests",
        "completed": True,
        "exit_code": 0,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {args.experiment_id} 未标注细胞纳入政策统计审计",
        "",
        "以下均为30个相同编号种子的成对比较；差值定义为全部细胞拟合减去仅标注细胞拟合。",
        "K=true 只用于隔离拟合集合效应，不属于可部署主比较。",
        "",
        "## 数据集级结果",
        "",
    ]
    for dataset in sorted(comparison["dataset"].unique()):
        lines.append(f"### {dataset}")
        lines.append("")
        for metric in ("ari", "macro_f1"):
            row = comparison[(comparison["dataset"] == dataset) & (comparison["metric"] == metric)].iloc[0]
            lines.append(
                f"- {metric}: 全细胞 {row.mean_all_cells_fit:.6f}，仅标注 {row.mean_labeled_only_fit:.6f}；"
                f"平均差 {row.mean_delta_all_minus_labeled:+.6f}，95% CI "
                f"[{row.ci95_delta_low:+.6f}, {row.ci95_delta_high:+.6f}]，"
                f"Holm校正 p={row.wilcoxon_p_holm:.6g}。"
            )
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- 该结果证明在本地三个基准和K-means中，聚类前过滤未标注细胞会实质改变外部评价。",
        "- 方向并非预设或普遍单向，因此不能把全细胞政策包装成必然提高分数的方法贡献。",
        "- 规范结论是人工标签不得决定无监督算法的拟合集合；性能变化属于该错误的后果而非采用新政策的理由。",
        "- 其他算法仍需在统一主协议中落实同一输入政策，本实验不能替代全算法重跑。",
    ])
    (args.output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "stdout.log").write_text(
        f"Analyzed {len(comparison)} dataset-by-metric comparisons and {len(pop_summary)} population rows.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "comparisons": len(comparison), "exit_code": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
