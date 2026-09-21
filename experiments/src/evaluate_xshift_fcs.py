from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import flowio
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


def read_cluster_ids(path: Path) -> tuple[np.ndarray, list[str]]:
    data = flowio.FlowData(str(path))
    matrix = np.asarray(data.events, dtype=np.float64).reshape(data.event_count, data.channel_count)
    names = [
        data.channels[index].get("pnn") or data.channels[index].get("pns") or f"channel_{index}"
        for index in range(1, data.channel_count + 1)
    ]
    indices = [
        index for index, name in enumerate(names) if name.lower().replace("_", "") == "clusterid"
    ]
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one cluster_id channel, got {indices} from {names}")
    values = matrix[:, indices[0]]
    if not np.allclose(values, np.rint(values)):
        raise ValueError("cluster_id contains non-integer values")
    return np.rint(values).astype(np.int32), names


def align_and_score(y_true: np.ndarray, y_pred: np.ndarray):
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for i, label in enumerate(true_labels):
        for j, cluster in enumerate(pred_labels):
            matrix[i, j] = int(np.sum((y_true == label) & (y_pred == cluster)))
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {
        int(pred_labels[col]): str(true_labels[row]) for row, col in zip(rows, cols, strict=False)
    }
    aligned = np.array([mapping.get(int(value), "__extra_cluster__") for value in y_pred])

    population_rows = []
    for label in true_labels:
        actual = y_true == label
        predicted = aligned == label
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        population_rows.append(
            {
                "population": str(label),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metrics = {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "macro_precision": float(np.mean([row["precision"] for row in population_rows])),
        "macro_recall": float(np.mean([row["recall"] for row in population_rows])),
        "macro_f1": float(np.mean([row["f1"] for row in population_rows])),
        "accuracy": float(np.mean(y_true == aligned)),
        "n_true_populations": int(len(true_labels)),
        "n_predicted_clusters": int(len(pred_labels)),
        "matched_clusters": int(len(mapping)),
        "unmatched_clusters": int(len(pred_labels) - len(mapping)),
        "unmatched_evaluable_events": int(np.sum(aligned == "__extra_cluster__")),
    }
    return metrics, population_rows, mapping, true_labels, pred_labels, matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fcs", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    fcs = args.fcs.resolve()
    parent_manifest_path = args.parent_manifest.resolve()
    protocol = args.protocol.resolve()
    required = (source, fcs, parent_manifest_path, protocol)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + "; ".join(missing))

    frame = pd.read_csv(source, sep="\t")
    prediction_all, channel_names = read_cluster_ids(fcs)
    labels_numeric = pd.to_numeric(frame["label"], errors="coerce")
    evaluable = (labels_numeric.between(1, 24) & (labels_numeric % 1 == 0)).to_numpy()
    evaluation_indices = np.flatnonzero(evaluable)
    y_true = np.array([str(int(value)) for value in labels_numeric[evaluable]])
    y_pred = prediction_all[evaluable]

    metrics, populations, mapping, true_labels, pred_labels, contingency = align_and_score(
        y_true, y_pred
    )
    metrics["n_predicted_clusters_all_events"] = int(np.unique(prediction_all).size)
    metrics["n_predicted_clusters_evaluable"] = metrics.pop("n_predicted_clusters")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "evaluation_indices.npy", evaluation_indices)
    pd.DataFrame(populations).to_csv(output / "population_level_metrics.csv", index=False)
    pd.DataFrame(
        [{"predicted_cluster": key, "matched_population": value} for key, value in mapping.items()]
    ).to_csv(output / "hungarian_mapping.csv", index=False)
    pd.DataFrame(contingency, index=true_labels, columns=pred_labels).to_csv(
        output / "contingency_true_by_predicted.csv", index_label="true_population"
    )
    unique, counts = np.unique(prediction_all, return_counts=True)
    pd.DataFrame({"predicted_cluster": unique, "all_event_count": counts}).to_csv(
        output / "cluster_sizes_all_events.csv", index=False
    )
    pd.DataFrame(
        [{"dataset": "Levine_13dim", "algorithm": "official_X-shift", "knn_k": 20, **metrics}]
    ).to_csv(output / "run_level_metrics.csv", index=False)

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    checks = [
        {
            "check": "source_hash_matches_parent",
            "passed": file_sha256(source) == parent_manifest["source_data_sha256"],
            "detail": file_sha256(source),
        },
        {
            "check": "all_event_count_matches",
            "passed": len(frame) == len(prediction_all) == 167044,
            "detail": f"source={len(frame)}, prediction={len(prediction_all)}",
        },
        {
            "check": "evaluable_count_matches_audit",
            "passed": int(evaluable.sum()) == 81747,
            "detail": f"evaluable={int(evaluable.sum())}",
        },
        {
            "check": "output_fcs_hash_matches_parent_inventory",
            "passed": file_sha256(fcs)
            == "e3969be4283e1cf56ecdb870dbbaf2509aa4307c135c28719d54b898b421015e",
            "detail": file_sha256(fcs),
        },
        {
            "check": "all_event_predicted_cluster_count",
            "passed": metrics["n_predicted_clusters_all_events"] == 80,
            "detail": str(metrics["n_predicted_clusters_all_events"]),
        },
        {
            "check": "evaluable_predicted_cluster_count",
            "passed": metrics["n_predicted_clusters_evaluable"] == 78,
            "detail": str(metrics["n_predicted_clusters_evaluable"]),
        },
        {
            "check": "cluster_channel_present",
            "passed": any(name.lower().replace("_", "") == "clusterid" for name in channel_names),
            "detail": str(channel_names),
        },
    ]
    pd.DataFrame(checks).to_csv(output / "evaluation_checks.csv", index=False)

    manifest = {
        "experiment_id": "EXP-009A",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": file_sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "parent_manifest": str(parent_manifest_path),
        "parent_manifest_sha256": file_sha256(parent_manifest_path),
        "source": str(source),
        "source_sha256": file_sha256(source),
        "fcs": str(fcs),
        "fcs_sha256": file_sha256(fcs),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "flowio": flowio.__version__ if hasattr(flowio, "__version__") else "1.4.0",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = [
        "# EXP-009A Official X-shift evaluation on Levine_13dim",
        "",
        f"Evaluated events: {int(evaluable.sum()):,}/{len(frame):,}; reference populations: 24; predicted clusters across all events: {metrics['n_predicted_clusters_all_events']}; predicted clusters in the evaluation subset: {metrics['n_predicted_clusters_evaluable']}.",
        f"ARI={metrics['ari']:.6f}; macro F1={metrics['macro_f1']:.6f}; accuracy={metrics['accuracy']:.6f}.",
        f"Unmatched clusters: {metrics['unmatched_clusters']}; unmatched evaluated events: {metrics['unmatched_evaluable_events']:,}.",
        "",
        "## Interpretation limits",
        "",
        "- K=20 is the authors' suggested nearest-neighbor default for an initial fast run; it is not the output cluster count. This run produced 80 clusters.",
        "- Manual labels were used only for this derived evaluation and did not enter EXP-009-R1 fitting or parameter selection.",
        "- A single run on one dataset cannot support conclusions about stability, K sensitivity, or overall method superiority.",
    ]
    (output / "audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps({**metrics, "all_checks_passed": manifest["all_checks_passed"]}))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
