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
from sklearn.metrics import adjusted_rand_score

from run_xshift_nilsson_fixedK20_30repeat import bootstrap_ci, read_fcs, sha256, target_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failed_parent = args.failed_parent.resolve(); data_path = args.data.resolve(); protocol = args.protocol.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs_dir = failed_parent / "runs"
    random_contract = json.loads((failed_parent / "randomness_source_contract.json").read_text(encoding="utf-8"))
    input_hashes = json.loads((failed_parent / "input_hashes.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(data_path)["label"].astype(str).to_numpy()
    if len(truth) != 44140 or np.count_nonzero(truth == "HSCs") != 358:
        raise ValueError("truth contract failed")

    parent_rows = []
    partitions = []
    run_rows = []
    for repeat in range(30):
        run_dir = runs_dir / f"repeat{repeat:03d}"
        manifest_path = run_dir / "run_manifest.json"
        labels_path = run_dir / "cluster_ids_all_events.npy"
        output_fcs = sorted((run_dir / "out").glob("*.fcs"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["repeat"] != repeat or not manifest["all_checks_passed"] or manifest["exit_code"] != 0 or manifest["timed_out"] or len(output_fcs) != 1:
            raise ValueError(f"parent repeat status failed: {repeat}")
        if sha256(labels_path) != manifest["labels_sha256"]:
            raise ValueError(f"parent label hash failed: {repeat}")
        labels = np.load(labels_path, allow_pickle=False)
        if labels.shape != (44140,) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"parent label shape failed: {repeat}")
        matrix, names = read_fcs(output_fcs[0])
        if matrix.shape != (44140, 14) or names[-1].lower().replace("_", "") != "clusterid" or not np.array_equal(np.rint(matrix[:, -1]).astype(np.int32), labels):
            raise ValueError(f"parent FCS/labels mismatch: {repeat}")
        partitions.append(labels)
        metrics = target_metrics(truth, labels)
        run_rows.append({"repeat": repeat, "runtime_seconds": manifest["runtime_seconds"], "n_clusters": int(np.unique(labels).size), "partition_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), "ari": adjusted_rand_score(truth, labels), **metrics})
        for role, path in (("manifest", manifest_path), ("labels", labels_path), ("output_fcs", output_fcs[0]), ("stderr", run_dir / "xshift_stderr.log")):
            parent_rows.append({"repeat": repeat, "role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})

    run_frame = pd.DataFrame(run_rows)
    run_frame.to_csv(output / "run_level_metrics.csv", index=False)
    pd.DataFrame(parent_rows).to_csv(output / "parent_artifact_hashes.csv", index=False)
    pair_rows = []
    for i in range(30):
        for j in range(i + 1, 30):
            pair_rows.append({"repeat_a": int(run_frame.iloc[i]["repeat"]), "repeat_b": int(run_frame.iloc[j]["repeat"]), "partition_ari": adjusted_rand_score(partitions[i], partitions[j])})
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output / "pairwise_partition_ari.csv", index=False)

    summary_rows = []
    for idx, metric in enumerate(("runtime_seconds", "n_clusters", "ari", "target_precision", "target_recall", "target_f1", "target_f2")):
        values = run_frame[metric].to_numpy(dtype=float)
        low, high = bootstrap_ci(values, 20260911 + idx)
        summary_rows.append({"metric": metric, "n": len(values), "mean": values.mean(), "sd": values.std(ddof=1), "median": np.median(values), "min": values.min(), "max": values.max(), "bootstrap_mean_ci_low": low, "bootstrap_mean_ci_high": high})
    pd.DataFrame(summary_rows).to_csv(output / "endpoint_descriptive_statistics.csv", index=False)

    checks = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    unique_hashes = int(run_frame.partition_sha256.nunique())
    pair_values = pair_frame.partition_ari.to_numpy(dtype=float)
    check("failed_parent_has_no_top_manifest", not (failed_parent / "run_manifest.json").exists(), "aggregation failed after 30 subruns")
    check("parent_repeat_count", len(run_frame) == 30 and set(run_frame.repeat) == set(range(30)), f"rows={len(run_frame)}")
    check("parent_artifact_count", len(parent_rows) == 120, f"rows={len(parent_rows)}")
    check("randomness_contract", random_contract["tag_commit"] == "fda75cf79980222da663185e2a6a72b442b9aff3" and random_contract["source_has_shuffle_initialization"] and random_contract["default_random_constructor_present"] and not random_contract["seeded_random_constructor_present"] and not random_contract["cli_seed_interface"], json.dumps(random_contract))
    check("input_data_hash", sha256(data_path) == input_hashes["data"], input_hashes["data"])
    check("all_partition_shapes", all(values.shape == (44140,) for values in partitions), "30/30")
    check("pairwise_count", len(pair_frame) == 435, f"rows={len(pair_frame)}")
    check("finite_metrics", np.isfinite(run_frame[["ari", "target_precision", "target_recall", "target_f1", "target_f2"]].to_numpy()).all(), "all finite")
    checks_frame = pd.DataFrame(checks); checks_frame.to_csv(output / "checks.csv", index=False)

    source_runner = Path(__file__).resolve().with_name("run_xshift_nilsson_fixedK20_30repeat.py")
    manifest = {
        "experiment_id": "EXP-038-R1", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol), "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "corrected_runner": str(source_runner), "corrected_runner_sha256": sha256(source_runner),
        "failed_parent": str(failed_parent), "randomness_source_contract_sha256": sha256(failed_parent / "randomness_source_contract.json"), "input_hashes_sha256": sha256(failed_parent / "input_hashes.json"),
        "data": str(data_path), "data_sha256": sha256(data_path), "repeat_not_seed": True,
        "runs_attempted": 30, "runs_successful": 30, "runs_failed": 0, "unique_partition_hashes": unique_hashes,
        "pairwise_partition_ari_mean": float(pair_values.mean()), "pairwise_partition_ari_min": float(pair_values.min()), "pairwise_partition_ari_max": float(pair_values.max()),
        "checks_passed": int(checks_frame.passed.sum()), "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame.passed.all()),
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    f1 = run_frame.target_f1.to_numpy(dtype=float)
    (output / "scientific_summary.md").write_text(
        "# EXP-038-R1 X-shift Nilsson固定K=20三十次原生重复\n\n"
        f"30/30个独立Java进程成功；唯一全事件分区哈希{unique_hashes}个。435个两两分区ARI均值/范围为{pair_values.mean():.6f}/[{pair_values.min():.6f}, {pair_values.max():.6f}]。"
        f"目标F1均值/范围为{f1.mean():.6f}/[{f1.min():.6f}, {f1.max():.6f}]。"
        "官方CLI无seed且源码使用默认Random初始化，本结果只能称原生repeat。\n",
        encoding="utf-8",
    )
    artifacts = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(json.dumps({path.name: sha256(path) for path in artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
