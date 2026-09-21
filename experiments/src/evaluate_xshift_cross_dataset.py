from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


EXPECTED = {
    "Levine_32dim": {
        "mode": "multiclass",
        "total_events": 265627,
        "evaluable_events": 104184,
        "true_populations": 14,
    },
    "Samusik_01": {
        "mode": "multiclass",
        "total_events": 86864,
        "evaluable_events": 53173,
        "true_populations": 24,
    },
    "Nilsson_rare": {
        "mode": "rare_target",
        "total_events": 44140,
        "evaluable_events": 44140,
        "target": "HSCs",
        "target_events": 358,
    },
    "Mosmann_rare": {
        "mode": "rare_target",
        "total_events": 396460,
        "evaluable_events": 396460,
        "target": "activated",
        "target_events": 109,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denominator if denominator else 0.0


def contingency_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels, true_codes = np.unique(y_true.astype(str), return_inverse=True)
    pred_labels, pred_codes = np.unique(y_pred.astype(np.int64), return_inverse=True)
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    np.add.at(matrix, (true_codes, pred_codes), 1)
    return true_labels, pred_labels, matrix


def effective_categories(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    proportions = counts[counts > 0] / total
    return float(1.0 / np.sum(proportions * proportions))


def evaluate_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray, prediction_all: np.ndarray
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    true_labels, pred_labels, matrix = contingency_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {int(pred_labels[col]): str(true_labels[row]) for row, col in zip(rows, cols)}
    inverse_mapping = {population: cluster for cluster, population in mapping.items()}
    aligned = np.array(
        [mapping.get(int(cluster), "__extra_cluster__") for cluster in y_pred], dtype=object
    )

    population_rows: list[dict] = []
    population_supports: list[int] = []
    for row_index, population in enumerate(true_labels):
        support = int(matrix[row_index].sum())
        population_supports.append(support)
        assigned_cluster = inverse_mapping.get(str(population))
        predicted = aligned == str(population)
        actual = y_true.astype(str) == str(population)
        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        fn = int(np.sum(~predicted & actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        proportions = matrix[row_index] / support
        dominant_col = int(np.argmax(matrix[row_index]))
        population_rows.append(
            {
                "population": str(population),
                "support": support,
                "hungarian_cluster": assigned_cluster,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f_score(precision, recall),
                "overlapping_predicted_clusters": int(np.sum(matrix[row_index] > 0)),
                "split_clusters_ge_1pct_population": int(np.sum(proportions >= 0.01)),
                "effective_predicted_clusters": effective_categories(matrix[row_index]),
                "dominant_predicted_cluster": int(pred_labels[dominant_col]),
                "dominant_cluster_capture": float(matrix[row_index, dominant_col] / support),
            }
        )

    all_clusters, all_counts = np.unique(prediction_all, return_counts=True)
    all_count_by_cluster = {
        int(cluster): int(count) for cluster, count in zip(all_clusters, all_counts)
    }
    eval_col_by_cluster = {int(cluster): index for index, cluster in enumerate(pred_labels)}
    cluster_rows: list[dict] = []
    for cluster in all_clusters:
        cluster_int = int(cluster)
        all_count = all_count_by_cluster[cluster_int]
        if cluster_int in eval_col_by_cluster:
            col = matrix[:, eval_col_by_cluster[cluster_int]]
            evaluable_count = int(col.sum())
            proportions = col / evaluable_count
            dominant_row = int(np.argmax(col))
            cluster_rows.append(
                {
                    "predicted_cluster": cluster_int,
                    "all_event_count": all_count,
                    "evaluable_event_count": evaluable_count,
                    "non_evaluable_event_count": all_count - evaluable_count,
                    "hungarian_population": mapping.get(cluster_int),
                    "dominant_population": str(true_labels[dominant_row]),
                    "purity_on_evaluable_events": float(col[dominant_row] / evaluable_count),
                    "overlapping_true_populations": int(np.sum(col > 0)),
                    "merge_populations_ge_1pct_cluster": int(np.sum(proportions >= 0.01)),
                    "effective_true_populations": effective_categories(col),
                }
            )
        else:
            cluster_rows.append(
                {
                    "predicted_cluster": cluster_int,
                    "all_event_count": all_count,
                    "evaluable_event_count": 0,
                    "non_evaluable_event_count": all_count,
                    "hungarian_population": None,
                    "dominant_population": None,
                    "purity_on_evaluable_events": np.nan,
                    "overlapping_true_populations": 0,
                    "merge_populations_ge_1pct_cluster": 0,
                    "effective_true_populations": 0.0,
                }
            )

    population_frame = pd.DataFrame(population_rows)
    cluster_frame = pd.DataFrame(cluster_rows)
    evaluable_cluster_frame = cluster_frame[cluster_frame["evaluable_event_count"] > 0]
    supports = np.asarray(population_supports, dtype=np.float64)
    weights = supports / supports.sum()
    metrics = {
        "evaluation_mode": "multiclass_rectangular_hungarian",
        "n_total_events": int(len(prediction_all)),
        "n_evaluable_events": int(len(y_true)),
        "n_true_populations": int(len(true_labels)),
        "n_predicted_clusters_all_events": int(len(all_clusters)),
        "n_predicted_clusters_evaluable": int(len(pred_labels)),
        "clusters_absent_from_evaluation": int(len(all_clusters) - len(pred_labels)),
        "overclustering_ratio_evaluable": float(len(pred_labels) / len(true_labels)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "macro_precision": float(population_frame["precision"].mean()),
        "macro_recall": float(population_frame["recall"].mean()),
        "macro_f1": float(population_frame["f1"].mean()),
        "weighted_precision": float(np.sum(population_frame["precision"] * weights)),
        "weighted_recall": float(np.sum(population_frame["recall"] * weights)),
        "weighted_f1": float(np.sum(population_frame["f1"] * weights)),
        "hungarian_accuracy": float(np.mean(aligned == y_true.astype(str))),
        "matched_clusters": int(len(mapping)),
        "unmatched_evaluable_clusters": int(len(pred_labels) - len(mapping)),
        "unmatched_evaluable_events": int(np.sum(aligned == "__extra_cluster__")),
        "mean_population_effective_predicted_clusters": float(
            population_frame["effective_predicted_clusters"].mean()
        ),
        "median_population_effective_predicted_clusters": float(
            population_frame["effective_predicted_clusters"].median()
        ),
        "mean_population_split_clusters_ge_1pct": float(
            population_frame["split_clusters_ge_1pct_population"].mean()
        ),
        "mean_evaluable_cluster_purity": float(
            evaluable_cluster_frame["purity_on_evaluable_events"].mean()
        ),
        "weighted_evaluable_cluster_purity": float(
            sum(matrix[:, col].max() for col in range(matrix.shape[1])) / matrix.sum()
        ),
        "mean_cluster_effective_true_populations": float(
            evaluable_cluster_frame["effective_true_populations"].mean()
        ),
        "mean_cluster_merge_populations_ge_1pct": float(
            evaluable_cluster_frame["merge_populations_ge_1pct_cluster"].mean()
        ),
    }
    mapping_frame = pd.DataFrame(
        [
            {"predicted_cluster": cluster, "matched_population": population}
            for cluster, population in sorted(mapping.items())
        ]
    )
    contingency_frame = pd.DataFrame(matrix, index=true_labels, columns=pred_labels)
    return metrics, population_frame, cluster_frame, mapping_frame, contingency_frame


def evaluate_rare_target(
    y_true: np.ndarray, y_pred: np.ndarray, target: str
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    actual = y_true.astype(str) == target
    target_events = int(actual.sum())
    background_events = int((~actual).sum())
    clusters = np.unique(y_pred)
    rows: list[dict] = []
    target_counts: list[int] = []
    background_counts: list[int] = []
    for cluster in clusters:
        predicted = y_pred == cluster
        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        fn = target_events - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / target_events if target_events else 0.0
        target_counts.append(tp)
        background_counts.append(fp)
        rows.append(
            {
                "predicted_cluster": int(cluster),
                "cluster_size": int(predicted.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f_score(precision, recall),
                "f2": f_score(precision, recall, beta=2.0),
                "target_fraction_in_cluster": precision,
            }
        )
    cluster_frame = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "predicted_cluster"], ascending=[False, False, True]
    )
    best = cluster_frame.iloc[0]
    target_array = np.asarray(target_counts, dtype=np.int64)
    background_array = np.asarray(background_counts, dtype=np.int64)
    metrics = {
        "evaluation_mode": "rare_target_best_single_cluster",
        "n_total_events": int(len(y_true)),
        "n_evaluable_events": int(len(y_true)),
        "n_predicted_clusters_all_events": int(len(clusters)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "target": target,
        "target_events": target_events,
        "target_prevalence": float(target_events / len(y_true)),
        "selected_target_cluster": int(best["predicted_cluster"]),
        "selected_cluster_size": int(best["cluster_size"]),
        "target_tp": int(best["tp"]),
        "target_fp": int(best["fp"]),
        "target_fn": int(best["fn"]),
        "target_precision": float(best["precision"]),
        "target_recall": float(best["recall"]),
        "target_f1": float(best["f1"]),
        "target_f2": float(best["f2"]),
        "target_overlapping_clusters": int(np.sum(target_array > 0)),
        "target_clusters_ge_1pct_target": int(
            np.sum(target_array / target_events >= 0.01)
        ),
        "target_effective_predicted_clusters": effective_categories(target_array),
        "target_dominant_cluster_capture": float(target_array.max() / target_events),
        "background_overlapping_clusters": int(np.sum(background_array > 0)),
        "background_clusters_ge_1pct_background": int(
            np.sum(background_array / background_events >= 0.01)
        ),
        "background_effective_predicted_clusters": effective_categories(background_array),
        "background_dominant_cluster_capture": float(
            background_array.max() / background_events
        ),
    }
    contingency = pd.crosstab(
        pd.Series(y_true.astype(str), name="reference_label"),
        pd.Series(y_pred.astype(np.int64), name="predicted_cluster"),
    )
    return metrics, cluster_frame, contingency


def metrics_are_valid(metrics: dict) -> tuple[bool, str]:
    proportion_keys = [
        key
        for key in metrics
        if key.endswith(("precision", "recall", "f1", "f2", "accuracy", "purity"))
        or key in {"ari", "target_prevalence", "target_dominant_cluster_capture", "background_dominant_cluster_capture"}
    ]
    invalid: list[str] = []
    for key in proportion_keys:
        value = float(metrics[key])
        if not np.isfinite(value):
            invalid.append(f"{key}=nonfinite")
        elif key != "ari" and not 0.0 <= value <= 1.0:
            invalid.append(f"{key}={value}")
        elif key == "ari" and not -1.0 <= value <= 1.0:
            invalid.append(f"ari={value}")
    effective_keys = [key for key in metrics if "effective" in key]
    for key in effective_keys:
        value = float(metrics[key])
        if not np.isfinite(value) or value < 1.0:
            invalid.append(f"{key}={value}")
    return len(invalid) == 0, "; ".join(invalid) if invalid else "all metric ranges valid"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-mode", choices=["fixed_k", "automatic_elbow"], default="fixed_k"
    )
    parser.add_argument("--expected-k", type=int, default=20)
    args = parser.parse_args()

    config_path = args.config.resolve()
    parent_run = args.parent_run.resolve()
    parent_manifest_path = parent_run / "run_manifest.json"
    prediction_path = parent_run / "cluster_ids_all_events.npy"
    protocol = args.protocol.resolve()
    required = (config_path, parent_manifest_path, prediction_path, protocol)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + "; ".join(missing))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["datasets"][args.dataset]
    source = (Path(config["data_root"]) / args.dataset / spec["filename"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    separator = "\t" if spec["separator"] == "tab" else ","
    labels = pd.read_csv(source, sep=separator, usecols=["label"])["label"].astype(str).to_numpy()
    prediction_all = np.load(prediction_path, allow_pickle=False)
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    expected = EXPECTED[args.dataset]

    if expected["mode"] == "multiclass":
        evaluable = labels != "unassigned"
        evaluation_indices = np.flatnonzero(evaluable)
        y_true = labels[evaluable]
        y_pred = prediction_all[evaluable]
        metrics, population_frame, cluster_frame, mapping_frame, contingency_frame = (
            evaluate_multiclass(y_true, y_pred, prediction_all)
        )
    else:
        evaluable = np.ones(len(labels), dtype=bool)
        evaluation_indices = np.arange(len(labels), dtype=np.int64)
        y_true = labels
        y_pred = prediction_all
        metrics, cluster_frame, contingency_frame = evaluate_rare_target(
            y_true, y_pred, expected["target"]
        )
        population_frame = None
        mapping_frame = None

    metric_valid, metric_detail = metrics_are_valid(metrics)
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("parent_manifest_passed", bool(parent_manifest.get("all_checks_passed")), str(parent_manifest.get("all_checks_passed")))
    check("parent_dataset_matches", parent_manifest.get("dataset") == args.dataset, str(parent_manifest.get("dataset")))
    parent_mode = parent_manifest.get("xshift_mode", "fixed_k")
    if args.expected_mode == "fixed_k":
        parameter_contract_passed = bool(
            parent_mode == "fixed_k"
            and parent_manifest.get("xshift_knn_k") == args.expected_k
        )
        parameter_contract_detail = json.dumps(
            {
                "expected_mode": args.expected_mode,
                "expected_k": args.expected_k,
                "parent_mode": parent_mode,
                "parent_k": parent_manifest.get("xshift_knn_k"),
            }
        )
    else:
        parameter_contract_passed = bool(
            parent_mode == "automatic_elbow"
            and parent_manifest.get("xshift_requested_argument") in {"auto", "1"}
            and len(parent_manifest.get("automatic_scan") or []) == 30
        )
        parameter_contract_detail = json.dumps(
            {
                "expected_mode": args.expected_mode,
                "parent_mode": parent_mode,
                "requested_argument": parent_manifest.get("xshift_requested_argument"),
                "auto_trigger": parent_manifest.get("xshift_auto_trigger"),
                "scan_points": len(parent_manifest.get("automatic_scan") or []),
            }
        )
    check(
        "parent_parameter_contract",
        parameter_contract_passed,
        parameter_contract_detail,
    )
    check("parent_fit_used_no_label", parent_manifest.get("label_used_for_fit_or_selection") is False, str(parent_manifest.get("label_used_for_fit_or_selection")))
    check("source_hash_matches_parent", sha256(source) == parent_manifest.get("source_sha256"), sha256(source))
    check("total_event_count", len(labels) == len(prediction_all) == expected["total_events"], f"source={len(labels)}, predictions={len(prediction_all)}, expected={expected['total_events']}")
    check("prediction_is_1d_integer", prediction_all.ndim == 1 and np.issubdtype(prediction_all.dtype, np.integer), f"shape={prediction_all.shape}, dtype={prediction_all.dtype}")
    check("prediction_has_no_negative_ids", bool(np.all(prediction_all >= 0)), f"min={int(prediction_all.min())}")
    check("cluster_count_matches_parent", int(np.unique(prediction_all).size) == parent_manifest.get("predicted_clusters"), f"observed={np.unique(prediction_all).size}, parent={parent_manifest.get('predicted_clusters')}")
    check("evaluable_event_count", int(evaluable.sum()) == expected["evaluable_events"], f"observed={int(evaluable.sum())}, expected={expected['evaluable_events']}")
    if expected["mode"] == "multiclass":
        check("true_population_count", int(np.unique(y_true).size) == expected["true_populations"], f"observed={np.unique(y_true).size}, expected={expected['true_populations']}")
    else:
        check("target_event_count", int(np.sum(y_true == expected["target"])) == expected["target_events"], f"observed={int(np.sum(y_true == expected['target']))}, expected={expected['target_events']}")
        check("binary_reference_labels", set(np.unique(y_true)) == {expected["target"], "other"}, str(sorted(np.unique(y_true))))
    check("metric_ranges", metric_valid, metric_detail)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "evaluation_indices.npy", evaluation_indices)
    pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "algorithm": "official_X-shift",
                "parameter_mode": parent_mode,
                "knn_k": parent_manifest.get("xshift_knn_k"),
                "automatic_selected_k_inferred": parent_manifest.get(
                    "automatic_selected_k_inferred"
                ),
                **metrics,
            }
        ]
    ).to_csv(output / "run_level_metrics.csv", index=False)
    cluster_frame.to_csv(output / "cluster_level_diagnostics.csv", index=False)
    contingency_frame.to_csv(output / "contingency_true_by_predicted.csv", index_label="true_population")
    if population_frame is not None:
        population_frame.to_csv(output / "population_level_metrics.csv", index=False)
        mapping_frame.to_csv(output / "hungarian_mapping.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "evaluation_checks.csv", index=False)

    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "evaluation_mode": metrics["evaluation_mode"],
        "parent_parameter_mode": parent_mode,
        "parent_knn_k": parent_manifest.get("xshift_knn_k"),
        "parent_automatic_selected_k_inferred": parent_manifest.get(
            "automatic_selected_k_inferred"
        ),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "parent_run": str(parent_run),
        "parent_manifest": str(parent_manifest_path),
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "source": str(source),
        "source_sha256": sha256(source),
        "cluster_ids": str(prediction_path),
        "cluster_ids_sha256": sha256(prediction_path),
        "labels_used_only_for_posthoc_evaluation": True,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifact_files = sorted(path for path in output.iterdir() if path.is_file())
    artifact_hashes = {str(path): sha256(path) for path in artifact_files}
    (output / "artifact_hashes.json").write_text(json.dumps(artifact_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if expected["mode"] == "multiclass":
        result_lines = [
            f"ARI={metrics['ari']:.6f}；Macro precision={metrics['macro_precision']:.6f}；Macro recall={metrics['macro_recall']:.6f}；Macro F1={metrics['macro_f1']:.6f}。",
            f"真实群体={metrics['n_true_populations']}；评价子集预测簇={metrics['n_predicted_clusters_evaluable']}；未匹配评价簇={metrics['unmatched_evaluable_clusters']}；未匹配评价事件={metrics['unmatched_evaluable_events']:,}。",
            f"平均群体有效拆分簇数={metrics['mean_population_effective_predicted_clusters']:.3f}；加权簇纯度={metrics['weighted_evaluable_cluster_purity']:.6f}。",
        ]
    else:
        result_lines = [
            f"ARI={metrics['ari']:.6f}；目标={metrics['target']}（{metrics['target_events']:,}/{metrics['n_total_events']:,}）。",
            f"最佳单簇={metrics['selected_target_cluster']}；precision={metrics['target_precision']:.6f}；recall={metrics['target_recall']:.6f}；F1={metrics['target_f1']:.6f}；F2={metrics['target_f2']:.6f}。",
            f"目标有效拆分簇数={metrics['target_effective_predicted_clusters']:.3f}；背景有效拆分簇数={metrics['background_effective_predicted_clusters']:.3f}。",
        ]
    audit = [
        f"# {args.experiment_id} 官方X-shift {args.dataset}事后评价",
        "",
        f"评价模式：{metrics['evaluation_mode']}；可评价事件={metrics['n_evaluable_events']:,}/{metrics['n_total_events']:,}。",
        *result_lines,
        "",
        f"完整性检查：{'全部通过' if manifest['all_checks_passed'] else '存在失败'}。",
        "标签只用于父运行完成后的评价；未用于拟合、K选择或参数搜索。",
        "本结果不证明K敏感性、正式稳定性或跨算法优劣。",
    ]
    (output / "audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, **metrics, "all_checks_passed": manifest["all_checks_passed"]}, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
