from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from evaluate_xshift_cross_dataset import evaluate_multiclass, evaluate_rare_target
from run_official_xshift import read_fcs, run_xshift_until_cluster_output
from run_xshift_nilsson_fixedK20_30repeat import bootstrap_ci


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WORKSPACE = ROOT.parent
JAVA = ROOT / "environments/temurin8-jre-win-x64/jdk8u504-b01-jre/bin/java.exe"
JAR = WORKSPACE / "reference_implementations/vortex_29jun2017_rev2/VorteX.jar"
CONFIG = ROOT / "config/datasets.json"
EXP042_PROTOCOL = ROOT / "protocols/EXP-042_xshift_mosmann_fixedK20_30repeat_mst_bypass_preregistered.md"
EXP042_RECOVERY_PROTOCOL = ROOT / "protocols/EXP-042-R1_xshift_mosmann_recovery_completion_preregistered.md"
EXP043_PROTOCOL = ROOT / "protocols/EXP-043_xshift_cross_dataset_K_sensitivity_preregistered.md"
EXP042_SOURCE = RUNS / "EXP-042_xshift_mosmann_fixedK20_30repeat_mst_bypass_20260913"
EXP042_RECOVERY = RUNS / "EXP-042-R1_xshift_mosmann_recovery_completion_20260913"
EXP042_VERIFY = RUNS / "EXP-042V-R2_xshift_mosmann_recovery_independent_verification_20260913"
EXP043 = RUNS / "EXP-043_xshift_cross_dataset_K_sensitivity_20260913"
EXP043_VERIFY = RUNS / "EXP-043V_xshift_cross_dataset_K_sensitivity_independent_verification_20260913"
PIPELINE = RUNS / "EXP-042-043_remaining_xshift_pipeline_20260913"


DATASETS = {
    "Nilsson_rare": {
        "parent": "EXP-010B_xshift_nilsson_fixedK20_20260910",
        "events": 44140,
        "markers": 13,
        "mode": "rare",
        "target": "HSCs",
    },
    "Samusik_01": {
        "parent": "EXP-010A_xshift_samusik_fixedK20_20260910",
        "events": 86864,
        "markers": 39,
        "mode": "multiclass",
    },
    "Levine_13dim": {
        "parent": "EXP-009-R2_xshift_levine13_full_K20_20260910",
        "events": 167044,
        "markers": 13,
        "mode": "multiclass",
    },
    "Levine_32dim": {
        "parent": "EXP-010C-R1_xshift_levine32_fixedK20_mst_bypass_20260910",
        "events": 265627,
        "markers": 32,
        "mode": "multiclass",
    },
    "Mosmann_rare": {
        "parent": "EXP-010D_xshift_mosmann_fixedK20_20260910",
        "events": 396460,
        "markers": 14,
        "mode": "rare",
        "target": "activated",
    },
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(stage: str, status: str, detail: str) -> None:
    PIPELINE.mkdir(parents=True, exist_ok=True)
    write_json(
        PIPELINE / "pipeline_state.json",
        {
            "updated_utc": utcnow(),
            "pid": os.getpid(),
            "stage": stage,
            "status": status,
            "detail": detail,
            "exp042_recovery": str(EXP042_RECOVERY),
            "exp042_verification": str(EXP042_VERIFY),
            "exp043": str(EXP043),
            "exp043_verification": str(EXP043_VERIFY),
        },
    )


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock() -> Path:
    PIPELINE.mkdir(parents=True, exist_ok=True)
    lock = PIPELINE / "pipeline.lock"
    if lock.exists():
        try:
            previous = load_json(lock)
            old_pid = int(previous.get("pid", -1))
        except Exception:
            previous, old_pid = {}, -1
        if old_pid != os.getpid() and pid_alive(old_pid):
            raise RuntimeError(f"pipeline already active with pid {old_pid}")
        stale = PIPELINE / f"pipeline_lock_stale_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        lock.replace(stale)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_utc": utcnow(), "command": sys.argv}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return lock


def find_input_fcs(parent: Path) -> Path:
    candidates = sorted(parent.glob("*_all_events_markers.fcs"))
    if len(candidates) != 1:
        raise ValueError(f"expected one parent input FCS in {parent}, found {len(candidates)}")
    return candidates[0]


def find_parent_labels(parent: Path) -> np.ndarray:
    npy = parent / "cluster_ids_all_events.npy"
    if npy.is_file():
        return np.load(npy, allow_pickle=False).astype(np.int32, copy=False)
    candidates = sorted((parent / "out").glob("*.fcs"))
    if len(candidates) != 1:
        raise ValueError(f"cannot locate parent clustering output in {parent}")
    matrix, names = read_fcs(candidates[0])
    if names[-1].lower().replace("_", "") != "clusterid":
        raise ValueError(f"parent output lacks cluster_id: {candidates[0]}")
    values = matrix[:, -1]
    if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
        raise ValueError(f"invalid parent cluster IDs: {candidates[0]}")
    return np.rint(values).astype(np.int32)


def next_attempt_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    attempts = sorted(path for path in base.glob("attempt*") if path.is_dir())
    index = 0 if not attempts else max(int(path.name.removeprefix("attempt")) for path in attempts) + 1
    attempt = base / f"attempt{index:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def successful_attempt(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    for attempt in sorted(base.glob("attempt*")):
        manifest = attempt / "run_manifest.json"
        if manifest.is_file():
            try:
                data = load_json(manifest)
            except Exception:
                continue
            if data.get("all_checks_passed") and (attempt / "cluster_ids_all_events.npy").is_file():
                return attempt
    return None


def run_fixed_k_attempt(
    attempt: Path,
    parent: Path,
    dataset: str,
    k: int,
    expected_events: int,
    expected_markers: int,
    timeout_seconds: int,
    extra: dict,
) -> dict:
    input_fcs = find_input_fcs(parent)
    input_matrix, marker_names = read_fcs(input_fcs)
    if input_matrix.shape != (expected_events, expected_markers):
        raise ValueError(f"parent input shape mismatch for {dataset}: {input_matrix.shape}")
    config = parent / "importConfig.txt"
    shutil.copy2(config, attempt / "importConfig.txt")
    (attempt / "fcsFileList.txt").write_text(str(input_fcs) + "\n", encoding="utf-8")
    command = [str(JAVA), "-Xmx4G", "-cp", str(JAR), "standalone.Xshift", str(k)]
    info = run_xshift_until_cluster_output(
        command, attempt, expected_events, marker_names, timeout_seconds
    )
    candidates = sorted((attempt / "out").glob("*.fcs"))
    passed = False
    detail = "no validated output"
    n_clusters = None
    labels_hash = None
    if len(candidates) == 1:
        matrix, names = read_fcs(candidates[0])
        cluster_values = matrix[:, -1]
        structure_passed = bool(
            matrix.shape == (expected_events, expected_markers + 1)
            and names[:-1] == marker_names
            and names[-1].lower().replace("_", "") == "clusterid"
            and np.all(np.isfinite(cluster_values))
            and np.allclose(cluster_values, np.rint(cluster_values))
            and np.array_equal(matrix[:, :-1], input_matrix)
        )
        execution_passed = bool(
            (
                info.get("cluster_output_complete_before_stop")
                and info.get("mst_stage_observed")
                and info.get("completion_mode")
                == "stopped_after_cluster_output_before_mst_layout"
            )
            or (
                info.get("completion_mode") == "natural_exit"
                and info.get("exit_code") == 0
            )
        )
        passed = structure_passed and execution_passed
        detail = f"shape={matrix.shape}; marker_exact={np.array_equal(matrix[:, :-1], input_matrix)}"
        if passed:
            labels = np.rint(cluster_values).astype(np.int32)
            np.save(attempt / "cluster_ids_all_events.npy", labels)
            unique, counts = np.unique(labels, return_counts=True)
            pd.DataFrame({"cluster_id": unique, "events": counts}).to_csv(
                attempt / "cluster_sizes.csv", index=False
            )
            n_clusters = int(len(unique))
            labels_hash = sha256(attempt / "cluster_ids_all_events.npy")
    record = {
        **extra,
        "completed_utc": utcnow(),
        "dataset": dataset,
        "knn_k": k,
        **info,
        "detail": detail,
        "n_clusters": n_clusters,
        "labels_sha256": labels_hash,
        "parent_input_fcs": str(input_fcs),
        "parent_input_fcs_sha256": sha256(input_fcs),
        "parent_config_sha256": sha256(config),
        "jar_sha256": sha256(JAR),
        "java_sha256": sha256(JAVA),
        "all_checks_passed": passed,
    }
    write_json(attempt / "run_manifest.json", record)
    return record


def existing_exp042_entries() -> dict[int, Path]:
    entries: dict[int, Path] = {}
    for repeat in range(30):
        directory = EXP042_SOURCE / "runs" / f"repeat{repeat:03d}"
        manifest = directory / "run_manifest.json"
        if manifest.is_file():
            record = load_json(manifest)
            if record.get("all_checks_passed") and (directory / "cluster_ids_all_events.npy").is_file():
                entries[repeat] = directory
    return entries


def recover_exp042_repeat(repeat: int) -> Path:
    base = EXP042_RECOVERY / "runs" / f"repeat{repeat:03d}"
    prior = successful_attempt(base)
    if prior:
        print(f"EXP042_RECOVERY_SKIP repeat={repeat:03d} path={prior}", flush=True)
        return prior
    attempt = next_attempt_dir(base)
    print(f"EXP042_RECOVERY_START repeat={repeat:03d} attempt={attempt.name}", flush=True)
    record = run_fixed_k_attempt(
        attempt=attempt,
        parent=RUNS / DATASETS["Mosmann_rare"]["parent"],
        dataset="Mosmann_rare",
        k=20,
        expected_events=396460,
        expected_markers=14,
        timeout_seconds=3600,
        extra={"experiment_id": "EXP-042-R1", "repeat": repeat, "repeat_not_seed": True},
    )
    print(
        f"EXP042_RECOVERY_DONE repeat={repeat:03d} pass={record['all_checks_passed']} runtime={record['runtime_seconds']:.2f}s",
        flush=True,
    )
    if not record["all_checks_passed"]:
        raise RuntimeError(f"EXP042 recovery repeat {repeat} failed; retained at {attempt}")
    return attempt


def load_truth(dataset: str) -> tuple[np.ndarray, np.ndarray, dict]:
    config = load_json(CONFIG)
    spec = config["datasets"][dataset]
    source = (Path(config["data_root"]) / dataset / spec["filename"]).resolve()
    separator = "\t" if spec["separator"] == "tab" else ","
    series = pd.read_csv(source, sep=separator, usecols=["label"])["label"]
    if dataset == "Levine_13dim":
        evaluable = series.notna().to_numpy() & series.fillna(-1).between(1, 24).to_numpy()
        labels = np.where(evaluable, series.fillna(-1).astype(int).astype(str).to_numpy(), "unassigned")
    else:
        labels = series.fillna("unassigned").astype(str).to_numpy()
        if DATASETS[dataset]["mode"] == "multiclass":
            evaluable = labels != "unassigned"
        else:
            evaluable = np.ones(len(labels), dtype=bool)
    return labels, evaluable, {"source": source, "spec": spec}


def evaluate_labels(dataset: str, labels: np.ndarray) -> tuple[dict, pd.DataFrame]:
    truth, evaluable, _ = load_truth(dataset)
    if len(truth) != len(labels):
        raise ValueError(f"truth/prediction length mismatch for {dataset}")
    if DATASETS[dataset]["mode"] == "multiclass":
        metrics, population, _, _, _ = evaluate_multiclass(
            truth[evaluable], labels[evaluable], labels
        )
        return metrics, population
    metrics, target_clusters, _ = evaluate_rare_target(
        truth, labels, DATASETS[dataset]["target"]
    )
    return metrics, target_clusters


def finalize_exp042(entries: dict[int, Path]) -> None:
    EXP042_RECOVERY.mkdir(parents=True, exist_ok=True)
    parent = RUNS / DATASETS["Mosmann_rare"]["parent"]
    parent_labels = find_parent_labels(parent)
    rows: list[dict] = []
    details: list[pd.DataFrame] = []
    comparisons: list[dict] = []
    partitions: list[np.ndarray] = []
    inventory: list[dict] = []
    for repeat in range(30):
        directory = entries[repeat]
        manifest_path = directory / "run_manifest.json"
        manifest = load_json(manifest_path)
        labels_path = directory / "cluster_ids_all_events.npy"
        labels = np.load(labels_path, allow_pickle=False)
        metrics, detail = evaluate_labels("Mosmann_rare", labels)
        rows.append(
            {
                "repeat": repeat,
                "runtime_seconds": float(manifest["runtime_seconds"]),
                "partition_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                **metrics,
            }
        )
        detail.insert(0, "repeat", repeat)
        details.append(detail)
        comparisons.append(
            {
                "repeat": repeat,
                "parent_partition_ari": adjusted_rand_score(parent_labels, labels),
                "parent_exact_label_fraction": float(np.mean(parent_labels == labels)),
            }
        )
        partitions.append(labels)
        for role, path in {
            "manifest": manifest_path,
            "labels": labels_path,
            "cluster_sizes": directory / "cluster_sizes.csv",
            "stderr": directory / "xshift_stderr.log",
            "fcs": sorted((directory / "out").glob("*.fcs"))[0],
        }.items():
            inventory.append(
                {
                    "repeat": repeat,
                    "source": "original" if directory.is_relative_to(EXP042_SOURCE) else "recovery",
                    "role": role,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(EXP042_RECOVERY / "run_level_metrics.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        EXP042_RECOVERY / "target_cluster_metrics.csv", index=False
    )
    pd.DataFrame(comparisons).to_csv(
        EXP042_RECOVERY / "parent_partition_comparison.csv", index=False
    )
    pair_rows = []
    for first in range(30):
        for second in range(first + 1, 30):
            pair_rows.append(
                {
                    "repeat_a": first,
                    "repeat_b": second,
                    "partition_ari": adjusted_rand_score(partitions[first], partitions[second]),
                }
            )
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(EXP042_RECOVERY / "pairwise_partition_ari.csv", index=False)
    stats = []
    metric_names = [
        "runtime_seconds",
        "n_predicted_clusters_all_events",
        "ari",
        "target_precision",
        "target_recall",
        "target_f1",
        "target_f2",
    ]
    for index, metric in enumerate(metric_names):
        values = metrics_frame[metric].to_numpy(float)
        low, high = bootstrap_ci(values, 20260913 + index)
        stats.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": np.median(values),
                "min": values.min(),
                "max": values.max(),
                "bootstrap_mean_ci_low": low,
                "bootstrap_mean_ci_high": high,
            }
        )
    pd.DataFrame(stats).to_csv(
        EXP042_RECOVERY / "endpoint_descriptive_statistics.csv", index=False
    )
    pd.DataFrame(inventory).to_csv(EXP042_RECOVERY / "evidence_inventory.csv", index=False)
    interruption_rows = []
    for repeat in (26, 27):
        path = EXP042_SOURCE / "runs" / f"repeat{repeat:03d}" / "xshift_stderr.log"
        interruption_rows.append(
            {
                "repeat": repeat,
                "status": "interrupted_before_child_manifest",
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(interruption_rows).to_csv(
        EXP042_RECOVERY / "interruption_evidence.csv", index=False
    )
    checks = [
        {"check": "thirty_successful_repeats", "passed": len(entries) == 30, "detail": f"count={len(entries)}"},
        {"check": "original_successes_preserved", "passed": sum(path.is_relative_to(EXP042_SOURCE) for path in entries.values()) == 26, "detail": "expected=26"},
        {"check": "recovered_successes", "passed": sum(path.is_relative_to(EXP042_RECOVERY) for path in entries.values()) == 4, "detail": "expected=4"},
        {"check": "run_metrics", "passed": len(metrics_frame) == 30 and metrics_frame["repeat"].nunique() == 30, "detail": f"rows={len(metrics_frame)}"},
        {"check": "pairwise_count", "passed": len(pair_frame) == 435, "detail": f"rows={len(pair_frame)}"},
        {"check": "finite_metrics", "passed": bool(np.isfinite(metrics_frame.select_dtypes(include=[np.number]).to_numpy()).all()), "detail": "numeric metrics"},
        {"check": "all_cluster_counts_87", "passed": set(metrics_frame["n_predicted_clusters_all_events"]) == {87}, "detail": str(sorted(metrics_frame["n_predicted_clusters_all_events"].unique()))},
        {"check": "evidence_inventory", "passed": len(inventory) == 150, "detail": f"files={len(inventory)}"},
        {"check": "source_interruption_preserved", "passed": (EXP042_SOURCE / "runs/repeat026/xshift_stderr.log").is_file() and (EXP042_SOURCE / "runs/repeat027/xshift_stderr.log").is_file(), "detail": "original incomplete logs retained"},
    ]
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(EXP042_RECOVERY / "checks.csv", index=False)
    all_passed = bool(check_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-042-R1",
        "completed_utc": utcnow(),
        "protocol": str(EXP042_RECOVERY_PROTOCOL),
        "protocol_sha256": sha256(EXP042_RECOVERY_PROTOCOL),
        "source_incomplete_run": str(EXP042_SOURCE),
        "source_protocol_sha256": sha256(EXP042_PROTOCOL),
        "parent": str(parent),
        "runs_attempted_in_final_set": 30,
        "runs_successful": 30,
        "original_successful_runs": 26,
        "recovered_successful_runs": 4,
        "unique_partition_hashes": int(metrics_frame["partition_sha256"].nunique()),
        "pairwise_partition_ari_min": float(pair_frame["partition_ari"].min()),
        "pairwise_partition_ari_max": float(pair_frame["partition_ari"].max()),
        "checks_passed": int(check_frame["passed"].sum()),
        "checks_total": len(check_frame),
        "all_checks_passed": all_passed,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(EXP042_RECOVERY / "run_manifest.json", manifest)
    (EXP042_RECOVERY / "scientific_summary.md").write_text(
        "# EXP-042-R1 X-shift Mosmann 30次原生重复收尾\n\n"
        f"原运行保留26个成功repeat，恢复4个；30/30进入正式集合。唯一分区={manifest['unique_partition_hashes']}；"
        f"两两ARI范围=[{manifest['pairwise_partition_ari_min']:.6f}, {manifest['pairwise_partition_ari_max']:.6f}]。"
        "repeat不是受控seed，原中断目录和日志继续保留。\n",
        encoding="utf-8",
    )
    own_files = sorted(path for path in EXP042_RECOVERY.rglob("*") if path.is_file() and path.name != "artifact_hashes.json")
    write_json(
        EXP042_RECOVERY / "artifact_hashes.json",
        {str(path.relative_to(EXP042_RECOVERY)): sha256(path) for path in own_files},
    )
    if not all_passed:
        raise RuntimeError("EXP042 recovery aggregation checks failed")


def ensure_exp042() -> None:
    write_state("EXP042_RECOVERY", "RUNNING", "checking existing 26 and recovering missing repeats")
    entries = existing_exp042_entries()
    missing = [repeat for repeat in range(30) if repeat not in entries]
    if missing:
        recovered: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(recover_exp042_repeat, repeat): repeat for repeat in missing}
            for future in as_completed(futures):
                repeat = futures[future]
                recovered[repeat] = future.result()
                write_state("EXP042_RECOVERY", "RUNNING", f"recovered repeat {repeat:03d}")
        entries.update(recovered)
    for repeat in range(30):
        if repeat not in entries:
            attempt = successful_attempt(EXP042_RECOVERY / "runs" / f"repeat{repeat:03d}")
            if attempt:
                entries[repeat] = attempt
    if len(entries) != 30:
        raise RuntimeError(f"EXP042 has only {len(entries)}/30 accepted repeats")
    existing_manifest = EXP042_RECOVERY / "run_manifest.json"
    if not existing_manifest.is_file() or not load_json(existing_manifest).get("all_checks_passed"):
        finalize_exp042(entries)
    write_state("EXP042_RECOVERY", "COMPLETE", "30/30 accepted; aggregate checks passed")


def run_verifier(stage: str, source: Path, output: Path, protocol: Path) -> None:
    if (output / "run_manifest.json").is_file() and load_json(output / "run_manifest.json").get("all_checks_passed"):
        print(f"VERIFY_SKIP stage={stage} path={output}", flush=True)
        return
    command = [
        sys.executable,
        str(ROOT / "src/verify_remaining_xshift_pipeline.py"),
        "--stage",
        stage,
        "--source",
        str(source),
        "--output",
        str(output),
        "--protocol",
        str(protocol),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    PIPELINE.mkdir(parents=True, exist_ok=True)
    (PIPELINE / f"verify_{stage}_stdout.log").write_text(result.stdout, encoding="utf-8")
    (PIPELINE / f"verify_{stage}_stderr.log").write_text(result.stderr, encoding="utf-8")
    write_json(
        PIPELINE / f"verify_{stage}_execution.json",
        {"command": command, "exit_code": result.returncode, "completed_utc": utcnow()},
    )
    if result.returncode != 0:
        raise RuntimeError(f"independent verifier failed for {stage}: {result.stderr[-1000:]}")


def run_exp043_condition(dataset: str, k: int) -> Path:
    base = EXP043 / "conditions" / dataset / f"K{k:03d}"
    prior = successful_attempt(base)
    if prior:
        print(f"EXP043_SKIP dataset={dataset} K={k} path={prior}", flush=True)
        return prior
    attempt = next_attempt_dir(base)
    spec = DATASETS[dataset]
    parent = RUNS / spec["parent"]
    print(f"EXP043_START dataset={dataset} K={k} attempt={attempt.name}", flush=True)
    record = run_fixed_k_attempt(
        attempt=attempt,
        parent=parent,
        dataset=dataset,
        k=k,
        expected_events=spec["events"],
        expected_markers=spec["markers"],
        timeout_seconds=3600,
        extra={"experiment_id": "EXP-043", "label_used_for_fit_or_selection": False},
    )
    print(
        f"EXP043_DONE dataset={dataset} K={k} pass={record['all_checks_passed']} runtime={record['runtime_seconds']:.2f}s",
        flush=True,
    )
    if not record["all_checks_passed"]:
        raise RuntimeError(f"EXP043 {dataset} K={k} failed; retained at {attempt}")
    return attempt


def finalize_exp043(new_entries: dict[tuple[str, int], Path]) -> None:
    metrics_rows: list[dict] = []
    population_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    partition_rows: list[dict] = []
    inventory: list[dict] = []
    for dataset, spec in DATASETS.items():
        parent = RUNS / spec["parent"]
        k20_labels = find_parent_labels(parent)
        labels_by_k: dict[int, np.ndarray] = {20: k20_labels}
        sources: dict[int, tuple[str, Path, Path]] = {
            20: ("existing_K20_parent", parent / "run_manifest.json", parent / "cluster_ids_all_events.npy")
        }
        if not sources[20][2].is_file():
            sources[20] = ("existing_K20_parent_FCS", parent / "run_manifest.json", sorted((parent / "out").glob("*.fcs"))[0])
        for k in (10, 40):
            attempt = new_entries[(dataset, k)]
            labels_by_k[k] = np.load(attempt / "cluster_ids_all_events.npy", allow_pickle=False)
            sources[k] = ("new_condition", attempt / "run_manifest.json", attempt / "cluster_ids_all_events.npy")
        for k in (10, 20, 40):
            labels = labels_by_k[k]
            metrics, detail = evaluate_labels(dataset, labels)
            metrics_rows.append({"dataset": dataset, "knn_k": k, **metrics})
            detail.insert(0, "knn_k", k)
            detail.insert(0, "dataset", dataset)
            if spec["mode"] == "multiclass":
                population_frames.append(detail)
            else:
                target_frames.append(detail)
            source_kind, manifest_path, label_path = sources[k]
            inventory.append(
                {
                    "dataset": dataset,
                    "knn_k": k,
                    "source_kind": source_kind,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256(manifest_path),
                    "labels_path": str(label_path),
                    "labels_sha256": sha256(label_path),
                    "partition_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                    "n_events": len(labels),
                    "n_clusters": int(np.unique(labels).size),
                }
            )
        for k in (10, 40):
            partition_rows.append(
                {
                    "dataset": dataset,
                    "knn_k": k,
                    "reference_k": 20,
                    "partition_ari_to_k20": adjusted_rand_score(k20_labels, labels_by_k[k]),
                    "exact_label_fraction_to_k20": float(np.mean(k20_labels == labels_by_k[k])),
                }
            )
    metrics = pd.DataFrame(metrics_rows).sort_values(["dataset", "knn_k"])
    inventory_frame = pd.DataFrame(inventory).sort_values(["dataset", "knn_k"])
    pair_frame = pd.DataFrame(partition_rows).sort_values(["dataset", "knn_k"])
    metrics.to_csv(EXP043 / "sensitivity_metrics.csv", index=False)
    pd.concat(population_frames, ignore_index=True).to_csv(
        EXP043 / "population_level_metrics.csv", index=False
    )
    pd.concat(target_frames, ignore_index=True).to_csv(
        EXP043 / "target_cluster_metrics.csv", index=False
    )
    pair_frame.to_csv(EXP043 / "partition_comparison_to_K20.csv", index=False)
    inventory_frame.to_csv(EXP043 / "condition_inventory.csv", index=False)
    numeric = metrics.select_dtypes(include=[np.number]).to_numpy()
    checks = [
        {"check": "fifteen_registered_conditions", "passed": len(metrics) == 15 and not metrics.duplicated(["dataset", "knn_k"]).any(), "detail": f"rows={len(metrics)}"},
        {"check": "ten_new_successful_conditions", "passed": len(new_entries) == 10, "detail": f"count={len(new_entries)}"},
        {"check": "frozen_k_grid", "passed": set(metrics["knn_k"]) == {10, 20, 40}, "detail": str(sorted(metrics["knn_k"].unique()))},
        {"check": "five_datasets", "passed": set(metrics["dataset"]) == set(DATASETS), "detail": str(sorted(metrics["dataset"].unique()))},
        {"check": "finite_metrics", "passed": bool(np.isfinite(numeric).all()), "detail": "all numeric endpoints finite"},
        {"check": "partition_comparisons", "passed": len(pair_frame) == 10 and pair_frame["partition_ari_to_k20"].between(-1, 1).all(), "detail": f"rows={len(pair_frame)}"},
        {"check": "labels_posthoc_only", "passed": all(load_json(path / "run_manifest.json").get("label_used_for_fit_or_selection") is False for path in new_entries.values()), "detail": "10/10 new conditions"},
        {"check": "inventory", "passed": len(inventory_frame) == 15, "detail": f"rows={len(inventory_frame)}"},
    ]
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(EXP043 / "checks.csv", index=False)
    all_passed = bool(check_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-043",
        "completed_utc": utcnow(),
        "protocol": str(EXP043_PROTOCOL),
        "protocol_sha256": sha256(EXP043_PROTOCOL),
        "datasets": list(DATASETS),
        "knn_k_grid": [10, 20, 40],
        "new_runs": 10,
        "reused_k20_conditions": 5,
        "labels_used_only_after_partition_freeze": True,
        "posthoc_k_selection_prohibited": True,
        "checks_passed": int(check_frame["passed"].sum()),
        "checks_total": len(check_frame),
        "all_checks_passed": all_passed,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(EXP043 / "run_manifest.json", manifest)
    (EXP043 / "scientific_summary.md").write_text(
        "# EXP-043 X-shift跨数据集K敏感性\n\n"
        "五个数据集均按预注册K=10/20/40并列评价；K=20复用已验收父分区，新增10个官方全事件运行。"
        "标签仅用于事后指标，结果不得用于反选部署K或跨算法排名。\n",
        encoding="utf-8",
    )
    own_files = sorted(path for path in EXP043.rglob("*") if path.is_file() and path.name != "artifact_hashes.json")
    write_json(
        EXP043 / "artifact_hashes.json",
        {str(path.relative_to(EXP043)): sha256(path) for path in own_files},
    )
    if not all_passed:
        raise RuntimeError("EXP043 aggregation checks failed")


def ensure_exp043() -> None:
    EXP043.mkdir(parents=True, exist_ok=True)
    write_state("EXP043", "RUNNING", "cross-dataset K=10/20/40 sensitivity")
    entries: dict[tuple[str, int], Path] = {}
    total = 10
    completed = 0
    for dataset in DATASETS:
        for k in (10, 40):
            entries[(dataset, k)] = run_exp043_condition(dataset, k)
            completed += 1
            write_state("EXP043", "RUNNING", f"new conditions accepted {completed}/{total}: {dataset} K={k}")
    manifest = EXP043 / "run_manifest.json"
    if not manifest.is_file() or not load_json(manifest).get("all_checks_passed"):
        finalize_exp043(entries)
    write_state("EXP043", "COMPLETE", "15 K conditions registered; aggregate checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume EXP-042, verify it, then run and verify EXP-043.")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    PIPELINE.mkdir(parents=True, exist_ok=True)
    if args.status_only:
        state = PIPELINE / "pipeline_state.json"
        print(state.read_text(encoding="utf-8") if state.is_file() else "no pipeline state")
        return 0
    required = [JAVA, JAR, CONFIG, EXP042_PROTOCOL, EXP042_RECOVERY_PROTOCOL, EXP043_PROTOCOL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pipeline inputs: " + "; ".join(missing))
    lock = acquire_lock()
    try:
        write_state("PREFLIGHT", "RUNNING", "all required paths present")
        ensure_exp042()
        write_state("EXP042_VERIFY", "RUNNING", "independent verification")
        run_verifier("exp042", EXP042_RECOVERY, EXP042_VERIFY, EXP042_RECOVERY_PROTOCOL)
        write_state("EXP042_VERIFY", "COMPLETE", "independent verification passed")
        ensure_exp043()
        write_state("EXP043_VERIFY", "RUNNING", "independent verification")
        run_verifier("exp043", EXP043, EXP043_VERIFY, EXP043_PROTOCOL)
        write_state("COMPLETE", "COMPLETE", "EXP042 recovery and EXP043 final main experiment independently verified")
        write_json(
            PIPELINE / "run_manifest.json",
            {
                "experiment_id": "EXP-042-043-PIPELINE",
                "completed_utc": utcnow(),
                "exp042_manifest_sha256": sha256(EXP042_RECOVERY / "run_manifest.json"),
                "exp042_verification_manifest_sha256": sha256(EXP042_VERIFY / "run_manifest.json"),
                "exp043_manifest_sha256": sha256(EXP043 / "run_manifest.json"),
                "exp043_verification_manifest_sha256": sha256(EXP043_VERIFY / "run_manifest.json"),
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256(Path(__file__).resolve()),
                "all_checks_passed": True,
            },
        )
        print("PIPELINE_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        write_state("ERROR", "FAILED", f"{type(exc).__name__}: {exc}")
        print(f"PIPELINE_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
