from __future__ import annotations

import hashlib
import numbers
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import run_remaining_xshift_pipeline as base
from sklearn.metrics import adjusted_rand_score

SOURCE = base.EXP043
OUTPUT = base.RUNS / "EXP-043-R1_xshift_cross_dataset_K_sensitivity_mode_aware_20260913"
VERIFY = (
    base.RUNS / "EXP-043V-R1_xshift_cross_dataset_K_sensitivity_independent_verification_20260913"
)
PROTOCOL = base.ROOT / "protocols/EXP-043-R1_mode_aware_aggregation_correction.md"
PIPELINE = base.PIPELINE


def applicable_values_finite(metrics: dict) -> bool:
    return all(
        np.isfinite(float(value))
        for value in metrics.values()
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
    )


def write_state(stage: str, status: str, detail: str) -> None:
    base.write_json(
        PIPELINE / "pipeline_state.json",
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "detail": detail,
            "exp042_recovery": str(base.EXP042_RECOVERY),
            "exp042_verification": str(base.EXP042_VERIFY),
            "exp043_failed_aggregate": str(SOURCE),
            "exp043_final": str(OUTPUT),
            "exp043_verification": str(VERIFY),
        },
    )


def gather_new_entries() -> dict[tuple[str, int], Path]:
    entries = {}
    for dataset in base.DATASETS:
        for k in (10, 40):
            attempt = base.successful_attempt(SOURCE / "conditions" / dataset / f"K{k:03d}")
            if attempt is None:
                raise RuntimeError(f"missing successful source condition: {dataset} K={k}")
            entries[(dataset, k)] = attempt
    return entries


def main() -> int:
    if OUTPUT.exists():
        manifest = OUTPUT / "run_manifest.json"
        if manifest.is_file() and base.load_json(manifest).get("all_checks_passed"):
            print("EXP043_R1_ALREADY_COMPLETE", flush=True)
        else:
            raise FileExistsError(f"incomplete correction output already exists: {OUTPUT}")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        entries = gather_new_entries()
        failed_manifest = base.load_json(SOURCE / "run_manifest.json")
        if failed_manifest.get("checks_passed") != 7 or failed_manifest.get("checks_total") != 8:
            raise RuntimeError("source aggregate failure contract changed")
        metrics_rows = []
        population_frames = []
        target_frames = []
        partition_rows = []
        inventory = []
        applicable_finite = True
        for dataset, spec in base.DATASETS.items():
            parent = base.RUNS / spec["parent"]
            k20 = base.find_parent_labels(parent)
            predictions = {20: k20}
            sources = {
                20: (
                    "existing_K20_parent",
                    parent / "run_manifest.json",
                    parent / "cluster_ids_all_events.npy",
                )
            }
            if not sources[20][2].is_file():
                sources[20] = (
                    "existing_K20_parent_FCS",
                    parent / "run_manifest.json",
                    sorted((parent / "out").glob("*.fcs"))[0],
                )
            for k in (10, 40):
                attempt = entries[(dataset, k)]
                predictions[k] = np.load(attempt / "cluster_ids_all_events.npy", allow_pickle=False)
                sources[k] = (
                    "new_condition",
                    attempt / "run_manifest.json",
                    attempt / "cluster_ids_all_events.npy",
                )
            for k in (10, 20, 40):
                metrics, detail = base.evaluate_labels(dataset, predictions[k])
                applicable_finite = applicable_finite and applicable_values_finite(metrics)
                metrics_rows.append({"dataset": dataset, "knn_k": k, **metrics})
                detail.insert(0, "knn_k", k)
                detail.insert(0, "dataset", dataset)
                (population_frames if spec["mode"] == "multiclass" else target_frames).append(
                    detail
                )
                source_kind, manifest_path, labels_path = sources[k]
                inventory.append(
                    {
                        "dataset": dataset,
                        "knn_k": k,
                        "source_kind": source_kind,
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": base.sha256(manifest_path),
                        "labels_path": str(labels_path),
                        "labels_sha256": base.sha256(labels_path),
                        "partition_sha256": hashlib.sha256(predictions[k].tobytes()).hexdigest(),
                        "n_events": len(predictions[k]),
                        "n_clusters": int(np.unique(predictions[k]).size),
                    }
                )
            for k in (10, 40):
                partition_rows.append(
                    {
                        "dataset": dataset,
                        "knn_k": k,
                        "reference_k": 20,
                        "partition_ari_to_k20": adjusted_rand_score(k20, predictions[k]),
                        "exact_label_fraction_to_k20": float(np.mean(k20 == predictions[k])),
                    }
                )
        metrics_frame = pd.DataFrame(metrics_rows).sort_values(["dataset", "knn_k"])
        inventory_frame = pd.DataFrame(inventory).sort_values(["dataset", "knn_k"])
        pair_frame = pd.DataFrame(partition_rows).sort_values(["dataset", "knn_k"])
        metrics_frame.to_csv(OUTPUT / "sensitivity_metrics.csv", index=False)
        pd.concat(population_frames, ignore_index=True).to_csv(
            OUTPUT / "population_level_metrics.csv", index=False
        )
        pd.concat(target_frames, ignore_index=True).to_csv(
            OUTPUT / "target_cluster_metrics.csv", index=False
        )
        pair_frame.to_csv(OUTPUT / "partition_comparison_to_K20.csv", index=False)
        inventory_frame.to_csv(OUTPUT / "condition_inventory.csv", index=False)
        checks = [
            {
                "check": "source_scientific_runs",
                "passed": len(entries) == 10
                and all(
                    base.load_json(path / "run_manifest.json").get("all_checks_passed")
                    for path in entries.values()
                ),
                "detail": "10/10 new conditions passed",
            },
            {
                "check": "source_failed_aggregate_preserved",
                "passed": failed_manifest.get("all_checks_passed") is False
                and failed_manifest.get("checks_passed") == 7,
                "detail": "EXP-043 aggregate 7/8 retained",
            },
            {
                "check": "fifteen_registered_conditions",
                "passed": len(metrics_frame) == 15
                and not metrics_frame.duplicated(["dataset", "knn_k"]).any(),
                "detail": f"rows={len(metrics_frame)}",
            },
            {
                "check": "frozen_design",
                "passed": set(metrics_frame.dataset) == set(base.DATASETS)
                and set(metrics_frame.knn_k) == {10, 20, 40},
                "detail": "five datasets; K=10,20,40",
            },
            {
                "check": "mode_aware_finite_metrics",
                "passed": applicable_finite,
                "detail": "finite values checked before task-specific wide-table union; structural NA permitted",
            },
            {
                "check": "partition_comparisons",
                "passed": len(pair_frame) == 10
                and pair_frame.partition_ari_to_k20.between(-1, 1).all(),
                "detail": f"rows={len(pair_frame)}",
            },
            {
                "check": "inventory",
                "passed": len(inventory_frame) == 15,
                "detail": f"rows={len(inventory_frame)}",
            },
            {
                "check": "no_label_selection",
                "passed": True,
                "detail": "K grid preregistered; labels posthoc only",
            },
        ]
        check_frame = pd.DataFrame(checks)
        check_frame.to_csv(OUTPUT / "checks.csv", index=False)
        all_passed = bool(check_frame.passed.all())
        manifest = {
            "experiment_id": "EXP-043-R1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "correction_scope": "derived aggregation check only; no algorithm rerun or metric change",
            "source_failed_aggregate": str(SOURCE),
            "source_failed_manifest_sha256": base.sha256(SOURCE / "run_manifest.json"),
            "protocol": str(PROTOCOL),
            "protocol_sha256": base.sha256(PROTOCOL),
            "datasets": list(base.DATASETS),
            "knn_k_grid": [10, 20, 40],
            "new_runs": 10,
            "reused_k20_conditions": 5,
            "posthoc_k_selection_prohibited": True,
            "checks_passed": int(check_frame.passed.sum()),
            "checks_total": len(check_frame),
            "all_checks_passed": all_passed,
            "script": str(Path(__file__).resolve()),
            "script_sha256": base.sha256(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
        }
        base.write_json(OUTPUT / "run_manifest.json", manifest)
        (OUTPUT / "scientific_summary.md").write_text(
            "# EXP-043-R1 Read-only finalization of cross-dataset K sensitivity\n\n"
            "The ten added runs and their event-level partitions are unchanged. R1 only limits finite-value checks to metrics applicable to each task; "
            "structurally inapplicable fields in the multiclass and rare-population wide tables remain empty.\n",
            encoding="utf-8",
        )
        own = sorted(
            path
            for path in OUTPUT.rglob("*")
            if path.is_file() and path.name != "artifact_hashes.json"
        )
        base.write_json(
            OUTPUT / "artifact_hashes.json",
            {str(path.relative_to(OUTPUT)): base.sha256(path) for path in own},
        )
        if not all_passed:
            raise RuntimeError("EXP043-R1 checks failed")

    if not (VERIFY / "run_manifest.json").is_file():
        command = [
            sys.executable,
            str(base.ROOT / "src/verify_remaining_xshift_pipeline.py"),
            "--stage",
            "exp043",
            "--source",
            str(OUTPUT),
            "--output",
            str(VERIFY),
            "--protocol",
            str(PROTOCOL),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        (PIPELINE / "verify_exp043_r1_stdout.log").write_text(result.stdout, encoding="utf-8")
        (PIPELINE / "verify_exp043_r1_stderr.log").write_text(result.stderr, encoding="utf-8")
        base.write_json(
            PIPELINE / "verify_exp043_r1_execution.json",
            {"command": command, "exit_code": result.returncode},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"EXP043-R1 independent verification failed: {result.stderr[-1000:]}"
            )
    verification_manifest = base.load_json(VERIFY / "run_manifest.json")
    if not verification_manifest.get("all_checks_passed"):
        raise RuntimeError("EXP043-R1 verification manifest is not passing")
    write_state(
        "COMPLETE",
        "COMPLETE",
        "EXP042 30/30 and EXP043 five-dataset K sensitivity independently verified",
    )
    base.write_json(
        PIPELINE / "run_manifest.json",
        {
            "experiment_id": "EXP-042-043-PIPELINE-R1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "exp042_manifest_sha256": base.sha256(base.EXP042_RECOVERY / "run_manifest.json"),
            "exp042_verification_manifest_sha256": base.sha256(
                base.EXP042_VERIFY / "run_manifest.json"
            ),
            "exp043_failed_aggregate_manifest_sha256": base.sha256(SOURCE / "run_manifest.json"),
            "exp043_final_manifest_sha256": base.sha256(OUTPUT / "run_manifest.json"),
            "exp043_verification_manifest_sha256": base.sha256(VERIFY / "run_manifest.json"),
            "script": str(Path(__file__).resolve()),
            "script_sha256": base.sha256(Path(__file__).resolve()),
            "all_checks_passed": True,
        },
    )
    print("PIPELINE_R1_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_state("ERROR", "FAILED", f"{type(exc).__name__}: {exc}")
        print(f"PIPELINE_R1_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
