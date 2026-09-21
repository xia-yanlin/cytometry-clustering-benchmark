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
from evaluate_xshift_cross_dataset import evaluate_multiclass, evaluate_rare_target
from run_official_xshift import read_fcs
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/datasets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_hash_manifest(source: Path) -> tuple[bool, str]:
    recorded = load_json(source / "artifact_hashes.json")
    bad = []
    for relative, expected in recorded.items():
        path = source / relative
        if not path.is_file() or sha256(path) != expected:
            bad.append(relative)
    return not bad, f"recorded={len(recorded)} bad={len(bad)}"


def load_truth(dataset: str) -> tuple[np.ndarray, np.ndarray, dict]:
    config = load_json(CONFIG)
    spec = config["datasets"][dataset]
    source = (Path(config["data_root"]) / dataset / spec["filename"]).resolve()
    series = pd.read_csv(
        source,
        sep="\t" if spec["separator"] == "tab" else ",",
        usecols=["label"],
    )["label"]
    if dataset == "Levine_13dim":
        evaluable = series.notna().to_numpy() & series.fillna(-1).between(1, 24).to_numpy()
        labels = np.where(
            evaluable, series.fillna(-1).astype(int).astype(str).to_numpy(), "unassigned"
        )
    else:
        labels = series.fillna("unassigned").astype(str).to_numpy()
        evaluable = (
            labels != "unassigned"
            if dataset not in {"Nilsson_rare", "Mosmann_rare"}
            else np.ones(len(labels), dtype=bool)
        )
    return labels, evaluable, spec


def evaluate(dataset: str, prediction: np.ndarray) -> dict:
    truth, evaluable, _ = load_truth(dataset)
    if dataset in {"Nilsson_rare", "Mosmann_rare"}:
        target = "HSCs" if dataset == "Nilsson_rare" else "activated"
        metrics, _, _ = evaluate_rare_target(truth, prediction, target)
        return metrics
    metrics, _, _, _, _ = evaluate_multiclass(truth[evaluable], prediction[evaluable], prediction)
    return metrics


def arrays_close(frame_a: pd.DataFrame, frame_b: pd.DataFrame, keys: list[str]) -> tuple[bool, str]:
    left = frame_a.sort_values(keys).reset_index(drop=True)
    right = frame_b.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return False, f"row mismatch {len(left)} != {len(right)}"
    common_numeric = sorted(
        set(left.select_dtypes(include=[np.number]).columns)
        & set(right.select_dtypes(include=[np.number]).columns) - set(keys)
    )
    keys_equal = all(left[key].astype(str).equals(right[key].astype(str)) for key in keys)
    numeric_equal = all(
        np.allclose(
            left[column].to_numpy(float),
            right[column].to_numpy(float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
        for column in common_numeric
    )
    return keys_equal and numeric_equal, f"rows={len(left)} numeric_columns={len(common_numeric)}"


def verify_exp042(source: Path) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    manifest = load_json(source / "run_manifest.json")
    add(
        "source_manifest",
        manifest.get("all_checks_passed") and manifest.get("runs_successful") == 30,
        json.dumps(
            {"runs": manifest.get("runs_successful"), "checks": manifest.get("checks_passed")}
        ),
    )
    ok, detail = check_hash_manifest(source)
    add("source_artifact_hashes", ok, detail)
    inventory = pd.read_csv(source / "evidence_inventory.csv")
    hash_bad = 0
    for row in inventory.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256(path) != row.sha256:
            hash_bad += 1
    add(
        "external_evidence_hashes",
        len(inventory) == 150 and hash_bad == 0,
        f"rows={len(inventory)} bad={hash_bad}",
    )
    interruptions = pd.read_csv(source / "interruption_evidence.csv")
    interruption_bad = sum(
        not Path(row.path).is_file() or sha256(Path(row.path)) != row.sha256
        for row in interruptions.itertuples(index=False)
    )
    add(
        "interruption_evidence_hashes",
        len(interruptions) == 2 and interruption_bad == 0,
        f"rows={len(interruptions)} bad={interruption_bad}",
    )
    label_rows = inventory[inventory.role.eq("labels")].sort_values("repeat")
    fcs_rows = inventory[inventory.role.eq("fcs")].sort_values("repeat")
    size_rows = inventory[inventory.role.eq("cluster_sizes")].sort_values("repeat")
    predictions: list[np.ndarray] = []
    fcs_bad = 0
    size_bad = 0
    recomputed = []
    for repeat in range(30):
        prediction = np.load(Path(label_rows.iloc[repeat].path), allow_pickle=False)
        predictions.append(prediction)
        fcs_matrix, fcs_names = read_fcs(Path(fcs_rows.iloc[repeat].path))
        fcs_labels = np.rint(fcs_matrix[:, -1]).astype(np.int32)
        if fcs_names[-1].lower().replace("_", "") != "clusterid" or not np.array_equal(
            fcs_labels, prediction
        ):
            fcs_bad += 1
        unique, counts = np.unique(prediction, return_counts=True)
        observed_sizes = pd.read_csv(Path(size_rows.iloc[repeat].path))
        if not (
            np.array_equal(observed_sizes["cluster_id"].to_numpy(np.int64), unique.astype(np.int64))
            and np.array_equal(observed_sizes["events"].to_numpy(np.int64), counts.astype(np.int64))
        ):
            size_bad += 1
        recomputed.append({"repeat": repeat, **evaluate("Mosmann_rare", prediction)})
    add(
        "thirty_label_arrays",
        len(predictions) == 30 and all(len(item) == 396460 for item in predictions),
        f"count={len(predictions)}",
    )
    add("fcs_label_identity", fcs_bad == 0, f"bad={fcs_bad}")
    add("cluster_size_identity", size_bad == 0, f"bad={size_bad}")
    reported = pd.read_csv(source / "run_level_metrics.csv")
    metric_ok, metric_detail = arrays_close(reported, pd.DataFrame(recomputed), ["repeat"])
    add("run_metrics_recomputed", metric_ok, metric_detail)
    pair_rows = []
    for first in range(30):
        for second in range(first + 1, 30):
            pair_rows.append(
                {
                    "repeat_a": first,
                    "repeat_b": second,
                    "partition_ari": adjusted_rand_score(predictions[first], predictions[second]),
                }
            )
    pair_ok, pair_detail = arrays_close(
        pd.read_csv(source / "pairwise_partition_ari.csv"),
        pd.DataFrame(pair_rows),
        ["repeat_a", "repeat_b"],
    )
    add("pairwise_ari_recomputed", pair_ok and len(pair_rows) == 435, pair_detail)
    add(
        "interruption_evidence_retained",
        manifest.get("original_successful_runs") == 26
        and manifest.get("recovered_successful_runs") == 4,
        "original=26 recovery=4",
    )
    return checks


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    matrix, names = read_fcs(path)
    if names[-1].lower().replace("_", "") != "clusterid":
        raise ValueError(f"missing cluster_id in {path}")
    return np.rint(matrix[:, -1]).astype(np.int32)


def verify_exp043(source: Path) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    manifest = load_json(source / "run_manifest.json")
    add(
        "source_manifest",
        manifest.get("all_checks_passed") and manifest.get("new_runs") == 10,
        json.dumps({"new_runs": manifest.get("new_runs"), "checks": manifest.get("checks_passed")}),
    )
    ok, detail = check_hash_manifest(source)
    add("source_artifact_hashes", ok, detail)
    inventory = pd.read_csv(source / "condition_inventory.csv")
    hash_bad = 0
    predictions: dict[tuple[str, int], np.ndarray] = {}
    recomputed = []
    for row in inventory.itertuples(index=False):
        manifest_path = Path(row.manifest_path)
        labels_path = Path(row.labels_path)
        if not manifest_path.is_file() or sha256(manifest_path) != row.manifest_sha256:
            hash_bad += 1
        if not labels_path.is_file() or sha256(labels_path) != row.labels_sha256:
            hash_bad += 1
            continue
        prediction = load_prediction(labels_path)
        predictions[(row.dataset, int(row.knn_k))] = prediction
        if hashlib.sha256(prediction.tobytes()).hexdigest() != row.partition_sha256:
            hash_bad += 1
        recomputed.append(
            {"dataset": row.dataset, "knn_k": int(row.knn_k), **evaluate(row.dataset, prediction)}
        )
    add(
        "condition_evidence_hashes",
        len(inventory) == 15 and hash_bad == 0,
        f"rows={len(inventory)} bad={hash_bad}",
    )
    reported = pd.read_csv(source / "sensitivity_metrics.csv")
    metric_ok, metric_detail = arrays_close(
        reported, pd.DataFrame(recomputed), ["dataset", "knn_k"]
    )
    add("fifteen_metrics_recomputed", metric_ok and len(recomputed) == 15, metric_detail)
    pair_rows = []
    for dataset in sorted(inventory.dataset.unique()):
        for k in (10, 40):
            pair_rows.append(
                {
                    "dataset": dataset,
                    "knn_k": k,
                    "reference_k": 20,
                    "partition_ari_to_k20": adjusted_rand_score(
                        predictions[(dataset, 20)], predictions[(dataset, k)]
                    ),
                    "exact_label_fraction_to_k20": float(
                        np.mean(predictions[(dataset, 20)] == predictions[(dataset, k)])
                    ),
                }
            )
    pair_ok, pair_detail = arrays_close(
        pd.read_csv(source / "partition_comparison_to_K20.csv"),
        pd.DataFrame(pair_rows),
        ["dataset", "knn_k", "reference_k"],
    )
    add("ten_partition_comparisons", pair_ok and len(pair_rows) == 10, pair_detail)
    new_manifests = inventory[inventory.source_kind.eq("new_condition")].manifest_path
    new_bad = sum(not load_json(Path(path)).get("all_checks_passed") for path in new_manifests)
    add(
        "ten_new_run_manifests",
        len(new_manifests) == 10 and new_bad == 0,
        f"count={len(new_manifests)} bad={new_bad}",
    )
    add(
        "frozen_grid_and_scope",
        set(inventory.knn_k) == {10, 20, 40}
        and len(set(inventory.dataset)) == 5
        and manifest.get("posthoc_k_selection_prohibited") is True,
        "five datasets; K=10,20,40; no posthoc selection",
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["exp042", "exp043"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    protocol = args.protocol.resolve()
    if not (source / "run_manifest.json").is_file() or not protocol.is_file():
        raise FileNotFoundError("source manifest or protocol missing")
    output.mkdir(parents=True, exist_ok=False)
    checks = verify_exp042(source) if args.stage == "exp042" else verify_exp043(source)
    frame = pd.DataFrame(checks)
    frame.to_csv(output / "verification_checks.csv", index=False)
    all_passed = bool(frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-042V-R2" if args.stage == "exp042" else "EXP-043V",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "source": str(source),
        "source_manifest_sha256": sha256(source / "run_manifest.json"),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "verification_script": str(Path(__file__).resolve()),
        "verification_script_sha256": sha256(Path(__file__).resolve()),
        "checks_passed": int(frame["passed"].sum()),
        "checks_total": len(frame),
        "all_checks_passed": all_passed,
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(output / "run_manifest.json", manifest)
    own = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "artifact_hashes.json"
    )
    write_json(output / "artifact_hashes.json", {path.name: sha256(path) for path in own})
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
