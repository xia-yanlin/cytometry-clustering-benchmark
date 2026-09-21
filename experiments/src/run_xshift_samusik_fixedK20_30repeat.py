from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from evaluate_xshift_cross_dataset import evaluate_multiclass
from run_xshift_nilsson_fixedK20_30repeat import bootstrap_ci, class_methodrefs, read_fcs, sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--release-jar", type=Path, required=True)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    parent = args.parent.resolve(); source_repo = args.source_repo.resolve(); release_jar = args.release_jar.resolve()
    java = args.java.resolve(); data_path = args.data.resolve(); protocol = args.protocol.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs_dir = output / "runs"; runs_dir.mkdir()

    helper = Path(__file__).with_name("run_xshift_nilsson_fixedK20_30repeat.py")
    evaluator = Path(__file__).with_name("evaluate_xshift_cross_dataset.py")
    parent_manifest = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    input_fcs = parent / "Samusik_01_all_events_markers.fcs"
    config = parent / "importConfig.txt"
    parent_labels = np.load(parent / "cluster_ids_all_events.npy", allow_pickle=False)
    parent_matrix, parent_names = read_fcs(input_fcs)
    if parent_manifest["experiment_id"] != "EXP-010A" or not parent_manifest["all_checks_passed"]:
        raise ValueError("formal Samusik parent contract failed")
    if parent_matrix.shape != (86864, 39) or len(parent_names) != 39 or parent_labels.shape != (86864,):
        raise ValueError("parent input/output shape contract failed")
    parent_jar = Path(json.loads((parent / "xshift_execution.json").read_text(encoding="utf-8"))["command"][3])
    if sha256(release_jar) != sha256(parent_jar):
        raise ValueError("release JAR differs from parent")

    tag_commit = subprocess.run(["git", "-C", str(source_repo), "rev-parse", "29-Jun-2017^{}"], capture_output=True, text=True, check=True).stdout.strip()
    source_text = subprocess.run(["git", "-C", str(source_repo), "show", "29-Jun-2017^{}:src/vortex/clustering/XShiftClustering.java"], capture_output=True, text=True, check=True).stdout
    with zipfile.ZipFile(release_jar) as archive:
        shuffle_class = archive.read("util/Shuffle.class")
    refs = class_methodrefs(shuffle_class)
    random_refs = sorted({ref for ref in refs if ref[0] == "java/util/Random"})
    random_contract = {
        "tag_commit": tag_commit,
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source_has_shuffle_initialization": "new Shuffle<Datapoint>()).shuffleCopyArray" in source_text,
        "source_has_math_random_fallback": "Math.random() * (numCells)" in source_text,
        "shuffle_class_sha256": hashlib.sha256(shuffle_class).hexdigest(),
        "java_util_random_methodrefs": random_refs,
        "default_random_constructor_present": ("java/util/Random", "<init>", "()V") in refs,
        "seeded_random_constructor_present": ("java/util/Random", "<init>", "(J)V") in refs,
        "cli_seed_interface": False,
    }
    (output / "randomness_source_contract.json").write_text(json.dumps(random_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if tag_commit != "fda75cf79980222da663185e2a6a72b442b9aff3" or not random_contract["source_has_shuffle_initialization"] or not random_contract["source_has_math_random_fallback"] or not random_contract["default_random_constructor_present"] or random_contract["seeded_random_constructor_present"]:
        raise ValueError("randomness source contract failed")

    truth = pd.read_csv(data_path, usecols=["label"])["label"].astype(str).to_numpy()
    evaluable = truth != "unassigned"
    if len(truth) != 86864 or int(evaluable.sum()) != 53173 or np.unique(truth[evaluable]).size != 24:
        raise ValueError("Samusik truth contract failed")

    command = [str(java), "-Xmx4G", "-cp", str(release_jar), "standalone.Xshift", "20"]
    input_hashes = {
        "input_fcs": sha256(input_fcs), "config": sha256(config), "release_jar": sha256(release_jar),
        "java": sha256(java), "data": sha256(data_path), "helper": sha256(helper), "evaluator": sha256(evaluator),
    }
    (output / "input_hashes.json").write_text(json.dumps(input_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run_one(repeat: int) -> dict[str, object]:
        run_dir = runs_dir / f"repeat{repeat:03d}"; run_dir.mkdir()
        (run_dir / "importConfig.txt").write_bytes(config.read_bytes())
        (run_dir / "fcsFileList.txt").write_text(str(input_fcs) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=run_dir, capture_output=True, text=True, timeout=args.timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            result = None; timed_out = True
            (run_dir / "xshift_stdout.log").write_text(exc.stdout or "", encoding="utf-8")
            (run_dir / "xshift_stderr.log").write_text(exc.stderr or "", encoding="utf-8")
        runtime = time.perf_counter() - started
        if result is not None:
            (run_dir / "xshift_stdout.log").write_text(result.stdout, encoding="utf-8")
            (run_dir / "xshift_stderr.log").write_text(result.stderr, encoding="utf-8")
            exit_code = result.returncode
        else:
            exit_code = None
        candidates = sorted((run_dir / "out").glob("*.fcs")) if (run_dir / "out").is_dir() else []
        success = False; detail = ""; n_clusters = None; labels_hash = None
        if not timed_out and exit_code == 0 and len(candidates) == 1:
            matrix, names = read_fcs(candidates[0]); values = matrix[:, -1]
            success = matrix.shape == (86864, 40) and names[:-1] == parent_names and names[-1].lower().replace("_", "") == "clusterid" and np.all(np.isfinite(values)) and np.allclose(values, np.rint(values)) and np.array_equal(matrix[:, :-1], parent_matrix)
            if success:
                labels = np.rint(values).astype(np.int32)
                np.save(run_dir / "cluster_ids_all_events.npy", labels)
                labels_hash = sha256(run_dir / "cluster_ids_all_events.npy")
                unique, counts = np.unique(labels, return_counts=True); n_clusters = int(len(unique))
                pd.DataFrame({"cluster_id": unique, "events": counts}).to_csv(run_dir / "cluster_sizes.csv", index=False)
            detail = f"shape={matrix.shape}; marker_exact={np.array_equal(matrix[:, :-1], parent_matrix)}"
        execution = {"repeat": repeat, "command": command, "cwd": str(run_dir), "exit_code": exit_code, "timed_out": timed_out, "runtime_seconds": runtime, "output_fcs_count": len(candidates), "all_checks_passed": success, "detail": detail, "n_clusters": n_clusters, "labels_sha256": labels_hash}
        (run_dir / "run_manifest.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return execution

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, repeat): repeat for repeat in range(30)}
        for future in as_completed(futures):
            row = future.result(); results.append(row)
            print(f"repeat={row['repeat']:03d} pass={row['all_checks_passed']} runtime={row['runtime_seconds']:.2f}s", flush=True)
    results.sort(key=lambda row: int(row["repeat"]))
    pd.DataFrame(results).to_csv(output / "execution_index.csv", index=False)

    run_rows: list[dict[str, object]] = []; population_frames = []; partitions = []; parent_rows = []
    for row in results:
        if not row["all_checks_passed"]:
            continue
        repeat = int(row["repeat"])
        labels = np.load(runs_dir / f"repeat{repeat:03d}" / "cluster_ids_all_events.npy", allow_pickle=False)
        partitions.append(labels)
        metrics, populations, _, _, _ = evaluate_multiclass(truth[evaluable], labels[evaluable], labels)
        run_rows.append({"repeat": repeat, "runtime_seconds": row["runtime_seconds"], "partition_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), **metrics})
        populations.insert(0, "repeat", repeat); population_frames.append(populations)
        parent_rows.append({"repeat": repeat, "parent_partition_ari": adjusted_rand_score(parent_labels, labels), "parent_exact_label_fraction": float(np.mean(parent_labels == labels))})
    run_frame = pd.DataFrame(run_rows); run_frame.to_csv(output / "run_level_metrics.csv", index=False)
    pd.concat(population_frames, ignore_index=True).to_csv(output / "population_level_metrics.csv", index=False)
    pd.DataFrame(parent_rows).to_csv(output / "parent_partition_comparison.csv", index=False)

    pair_rows = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            pair_rows.append({"repeat_a": int(run_frame.iloc[i]["repeat"]), "repeat_b": int(run_frame.iloc[j]["repeat"]), "partition_ari": adjusted_rand_score(partitions[i], partitions[j])})
    pair_frame = pd.DataFrame(pair_rows); pair_frame.to_csv(output / "pairwise_partition_ari.csv", index=False)

    metrics_to_summarize = ("runtime_seconds", "n_predicted_clusters_all_events", "ari", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "hungarian_accuracy", "weighted_evaluable_cluster_purity")
    summary_rows = []
    for idx, metric in enumerate(metrics_to_summarize):
        values = run_frame[metric].to_numpy(dtype=float); low, high = bootstrap_ci(values, 20260911 + idx)
        summary_rows.append({"metric": metric, "n": len(values), "mean": values.mean(), "sd": values.std(ddof=1), "median": np.median(values), "min": values.min(), "max": values.max(), "bootstrap_mean_ci_low": low, "bootstrap_mean_ci_high": high})
    pd.DataFrame(summary_rows).to_csv(output / "endpoint_descriptive_statistics.csv", index=False)

    successful = int(sum(bool(row["all_checks_passed"]) for row in results)); unique_hashes = int(run_frame.partition_sha256.nunique()) if len(run_frame) else 0
    checks = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    check("parent_manifest", parent_manifest["experiment_id"] == "EXP-010A" and parent_manifest["all_checks_passed"], parent_manifest["experiment_id"])
    check("randomness_source", random_contract["default_random_constructor_present"] and not random_contract["seeded_random_constructor_present"] and not random_contract["cli_seed_interface"], json.dumps(random_contract))
    check("execution_accounting", len(results) == 30 and {int(row["repeat"]) for row in results} == set(range(30)), f"rows={len(results)}")
    check("all_runs_successful", successful == 30, f"success={successful}; failed={30-successful}")
    check("run_metrics", len(run_frame) == successful and run_frame["repeat"].nunique() == successful, f"rows={len(run_frame)}")
    check("population_metrics", sum(len(frame) for frame in population_frames) == successful * 24, f"rows={sum(len(frame) for frame in population_frames)}")
    check("pairwise_count", len(pair_frame) == successful * (successful - 1) // 2, f"rows={len(pair_frame)}")
    check("input_hash_uniformity", all(sha256(runs_dir / f"repeat{repeat:03d}" / "importConfig.txt") == input_hashes["config"] for repeat in range(30)), "30/30 configs")
    check("labels_valid", len(run_frame) == 30 and run_frame.n_predicted_clusters_all_events.ge(1).all(), f"cluster_range={run_frame.n_predicted_clusters_all_events.min() if len(run_frame) else None}-{run_frame.n_predicted_clusters_all_events.max() if len(run_frame) else None}")
    check("finite_primary_metrics", np.isfinite(run_frame[["ari", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "hungarian_accuracy"]].to_numpy()).all(), "all finite")
    checks_frame = pd.DataFrame(checks); checks_frame.to_csv(output / "checks.csv", index=False)
    pair_values = pair_frame.partition_ari.to_numpy(dtype=float)
    manifest = {
        "experiment_id": "EXP-039", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol), "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "helper_script": str(helper), "helper_script_sha256": sha256(helper), "evaluator_script": str(evaluator), "evaluator_script_sha256": sha256(evaluator),
        "parent": str(parent), "parent_manifest_sha256": sha256(parent / "run_manifest.json"), "source_repo": str(source_repo), "tag_commit": tag_commit,
        "release_jar": str(release_jar), "release_jar_sha256": sha256(release_jar), "java": str(java), "java_sha256": sha256(java), "data": str(data_path), "data_sha256": sha256(data_path),
        "workers": args.workers, "repeat_not_seed": True, "runs_attempted": 30, "runs_successful": successful, "runs_failed": 30-successful, "unique_partition_hashes": unique_hashes,
        "pairwise_partition_ari_mean": float(pair_values.mean()), "pairwise_partition_ari_min": float(pair_values.min()), "pairwise_partition_ari_max": float(pair_values.max()),
        "checks_passed": int(checks_frame.passed.sum()), "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame.passed.all()),
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "scientific_summary.md").write_text(
        "# EXP-039 X-shift Samusik固定K=20三十次原生重复\n\n"
        f"成功{successful}/30；唯一全事件分区哈希{unique_hashes}个；435个两两分区ARI均值/范围={pair_values.mean():.6f}/[{pair_values.min():.6f}, {pair_values.max():.6f}]。"
        f"Macro F1均值/范围={run_frame.macro_f1.mean():.6f}/[{run_frame.macro_f1.min():.6f}, {run_frame.macro_f1.max():.6f}]。"
        "CLI无seed且源码/JAR使用无参Random，运行编号只代表native repeat。\n",
        encoding="utf-8",
    )
    artifacts = [p for p in output.rglob("*") if p.is_file()]
    (output / "artifact_hashes.json").write_text(json.dumps({str(p.relative_to(output)): sha256(p) for p in artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
