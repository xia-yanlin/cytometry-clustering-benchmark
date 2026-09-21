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
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def contingency(y_true: np.ndarray, y_pred: np.ndarray):
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            matrix[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))
    return true_labels, pred_labels, matrix


def rectangular_alignment(y_true: np.ndarray, y_pred: np.ndarray):
    true_labels, pred_labels, matrix = contingency(y_true, y_pred)
    rows, cols = linear_sum_assignment(-matrix)
    pred_to_true = {pred_labels[col]: true_labels[row] for row, col in zip(rows, cols)}
    aligned = np.array([pred_to_true.get(value, "__extra_cluster__") for value in y_pred])
    return aligned, matrix, pred_to_true


def prf(y_true: np.ndarray, y_aligned: np.ndarray, label: str) -> dict:
    positive = y_true == label
    predicted = y_aligned == label
    tp = int(np.sum(positive & predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "f2": f2}


def boundary_tests() -> list[dict]:
    tests = []

    y_true = np.array(["A", "A", "B", "B", "C", "C"])
    y_pred = np.array(["p1", "p1", "p2", "p2", "p2", "p2"])
    aligned, matrix, mapping = rectangular_alignment(y_true, y_pred)
    scores = [prf(y_true, aligned, label)["f1"] for label in np.unique(y_true)]
    tests.append({
        "case": "K_pred_lt_K_true",
        "k_true": 3,
        "k_pred": 2,
        "matrix_shape": str(matrix.shape),
        "mapping_size": len(mapping),
        "unmatched_true_populations": 3 - len(set(mapping.values())),
        "macro_f1": float(np.mean(scores)),
        "pass": len(mapping) == 2 and sum(score == 0 for score in scores) == 1,
    })

    y_true = np.array(["A", "A", "B", "B"])
    y_pred = np.array(["p1", "p2", "p3", "p3"])
    aligned, matrix, mapping = rectangular_alignment(y_true, y_pred)
    scores = [prf(y_true, aligned, label)["f1"] for label in np.unique(y_true)]
    tests.append({
        "case": "K_pred_gt_K_true",
        "k_true": 2,
        "k_pred": 3,
        "matrix_shape": str(matrix.shape),
        "mapping_size": len(mapping),
        "extra_predicted_clusters": 3 - len(mapping),
        "macro_f1": float(np.mean(scores)),
        "pass": len(mapping) == 2 and int(np.sum(aligned == "__extra_cluster__")) == 1,
    })

    y_true = np.array(["target"] * 10 + ["other"] * 90)
    y_pred = np.array(["cluster_target"] * 10 + ["cluster_other"] * 90)
    aligned, _, _ = rectangular_alignment(y_true, y_pred)
    score = prf(y_true, aligned, "target")
    tests.append({
        "case": "rare_target_perfect",
        "k_true": 2,
        "k_pred": 2,
        "matrix_shape": "(2, 2)",
        "mapping_size": 2,
        "target_f1": score["f1"],
        "pass": score["precision"] == score["recall"] == score["f1"] == 1.0,
    })
    return tests


def background_split_simulation() -> list[dict]:
    rows = []
    n_target = 100
    n_other = 9900
    y_true = np.array(["target"] * n_target + ["other"] * n_other)
    for splits in [1, 2, 4, 8, 16, 32, 64]:
        background = np.arange(n_other) % splits
        y_pred = np.concatenate([
            np.array(["target_cluster"] * n_target),
            np.array([f"other_{value}" for value in background]),
        ])
        target_pred = np.array(["target" if value == "target_cluster" else "other" for value in y_pred])
        score = prf(y_true, target_pred, "target")
        rows.append({
            "background_clusters": splits,
            "target_prevalence": n_target / (n_target + n_other),
            "target_precision": score["precision"],
            "target_recall": score["recall"],
            "target_f1": score["f1"],
            "target_f2": score["f2"],
            "binary_ari_using_raw_clusters": adjusted_rand_score(y_true, y_pred),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    tests = boundary_tests()
    simulation = background_split_simulation()
    pd.DataFrame(tests).to_csv(args.output / "hungarian_boundary_tests.csv", index=False)
    pd.DataFrame(simulation).to_csv(args.output / "rare_background_split_simulation.csv", index=False)
    failures = [row["case"] for row in tests if not row["pass"]]
    manifest = {
        "experiment_id": "EXP-002",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__)),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
        },
        "failures": failures,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    first = simulation[0]
    last = simulation[-1]
    report = [
        "# EXP-002 指标与匹配规则验证",
        "",
        "对应审稿问题：12、15、16、17。",
        "",
        f"边界测试：{len(tests) - len(failures)}/{len(tests)} 通过。",
        "",
        "## 背景拆分模拟",
        "",
        f"目标群体分类始终完美（precision=recall=F1={first['target_f1']:.3f}），",
        f"但把同一背景从 1 个簇拆成 64 个簇后，原始聚类 ARI 从 {first['binary_ari_using_raw_clusters']:.6f} 变为 {last['binary_ari_using_raw_clusters']:.6f}。",
        "",
        "这证明在稀有群体检测任务中，ARI 会同时惩罚背景内部的异质性划分，不能替代目标群体 precision/recall/F1/F2。",
        "",
        "## 完善度边界",
        "",
        "- 问题 15 的边界计分规则可在单元层面达标，但仍需接入真实算法流水线。",
        "- 问题 12、16、17 目前只完成机制和指标实现验证；必须在两个真实稀有数据集上运行后再评估是否达标。",
    ]
    (args.output / "audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    exit_code = int(bool(failures))
    summary = {"output": str(args.output), "failures": failures, "exit_code": exit_code}
    (args.output / "stdout.log").write_text(
        json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "run_status.json").write_text(
        json.dumps({"exit_code": exit_code, "completed": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
