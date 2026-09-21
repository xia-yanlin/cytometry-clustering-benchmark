from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CYTOMETRY_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "raw")))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "EXP-044_deterministic_spade_official_endpoint_20260915"
PROTOCOL = ROOT / "protocols" / "EXP-044-R4_deterministic_spade_nilsson_external_evaluation_preregistered.md"
LABELS = DATA_ROOT / "Nilsson_rare" / "Nilsson_rare_notransform.csv"
EXPECTED_LABEL_SHA256 = "e74c804a6cb040fd5cf1bea3c9d1ce382b266d860209d443f7f3960f129c918c"
EXPECTED_EVENTS = 44_140
EXPECTED_TARGET = 358


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(truth: np.ndarray, predicted: np.ndarray, attempt: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    target = truth == "HSCs"
    rows: list[dict[str, object]] = []
    for cluster in sorted(np.unique(predicted)):
        selected = predicted == cluster
        tp = int(np.count_nonzero(target & selected))
        fp = int(np.count_nonzero(~target & selected))
        fn = int(np.count_nonzero(target & ~selected))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
        rows.append({
            "attempt": attempt,
            "cluster_id": int(cluster),
            "cluster_events": int(selected.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
        })
    best = max(rows, key=lambda row: (float(row["f1"]), float(row["recall"]), -int(row["cluster_id"])))
    summary = {
        "attempt": attempt,
        "events": int(predicted.size),
        "target_events": int(target.sum()),
        "occupied_clusters": int(np.unique(predicted).size),
        "ari_two_class_reference_vs_40_clusters": float(adjusted_rand_score(truth, predicted)),
        "selected_cluster": int(best["cluster_id"]),
        "selected_cluster_size": int(best["cluster_events"]),
        "target_tp": int(best["tp"]),
        "target_fp": int(best["fp"]),
        "target_fn": int(best["fn"]),
        "target_precision": float(best["precision"]),
        "target_recall": float(best["recall"]),
        "target_f1": float(best["f1"]),
        "target_f2": float(best["f2"]),
    }
    return summary, rows


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("label_source_hash", sha256(LABELS) == EXPECTED_LABEL_SHA256, sha256(LABELS))
    truth = pd.read_csv(LABELS, usecols=["label"])["label"].astype(str).to_numpy()
    check("label_contract", len(truth) == EXPECTED_EVENTS and int(np.count_nonzero(truth == "HSCs")) == EXPECTED_TARGET, f"events={len(truth)} target={np.count_nonzero(truth == 'HSCs')}")

    summaries: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    arrays: list[np.ndarray] = []
    for attempt in (3, 4):
        directory = RUN / f"attempt{attempt:03d}"
        validation = json.loads((directory / "validation_report.json").read_text(encoding="utf-8"))
        predicted = np.load(directory / "cluster_ids_all_events.npy", allow_pickle=False)
        check(f"attempt{attempt}_validated", bool(validation["all_checks_passed"]), str(validation["checks"]))
        check(f"attempt{attempt}_event_contract", predicted.shape == (EXPECTED_EVENTS,) and np.unique(predicted).size == 40, f"shape={predicted.shape} clusters={np.unique(predicted).size}")
        summary, rows = evaluate(truth, predicted, attempt)
        summaries.append(summary)
        cluster_rows.extend(rows)
        arrays.append(predicted)

    exact = bool(np.array_equal(arrays[0], arrays[1]))
    metric_fields = [key for key in summaries[0] if key != "attempt"]
    metrics_exact = all(summaries[0][key] == summaries[1][key] for key in metric_fields)
    check("repeat_labels_exact", exact, sha256(RUN / "attempt003" / "cluster_ids_all_events.npy"))
    check("repeat_external_metrics_exact", metrics_exact, str({key: summaries[0][key] for key in metric_fields}))

    pd.DataFrame(cluster_rows).to_csv(RUN / "external_evaluation_per_cluster.csv", index=False)
    pd.DataFrame(checks).to_csv(RUN / "external_evaluation_checks.csv", index=False)
    report = {
        "experiment_id": "EXP-044-R4",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "label_use": "external evaluation only; no fitting, parameter selection, or K selection",
        "selection_rule": "single cluster maximizing F1; tie by higher recall then lower cluster ID",
        "label_source": str(LABELS),
        "label_source_sha256": sha256(LABELS),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "attempts": summaries,
        "checks_passed": int(sum(bool(row["passed"]) for row in checks)),
        "checks_total": len(checks),
        "all_checks_passed": bool(all(bool(row["passed"]) for row in checks)),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(RUN / "external_evaluation.json", report)
    write_json(RUN / "external_evaluation_artifact_hashes.json", {
        name: sha256(RUN / name)
        for name in ["external_evaluation.json", "external_evaluation_per_cluster.csv", "external_evaluation_checks.csv"]
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
