from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import flowio
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_fcs(path: Path) -> tuple[np.ndarray, list[str]]:
    data = flowio.FlowData(str(path))
    matrix = np.asarray(data.events, dtype=np.float64).reshape(data.event_count, data.channel_count)
    names = [
        data.channels[i].get("pnn") or data.channels[i].get("pns") or f"channel_{i}"
        for i in range(1, data.channel_count + 1)
    ]
    return matrix, names


def class_methodrefs(data: bytes) -> list[tuple[str, str, str]]:
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("not a Java class")
    pos = 8
    count = struct.unpack_from(">H", data, pos)[0]
    pos += 2
    cp: list[object | None] = [None] * count
    i = 1
    while i < count:
        tag = data[pos]
        pos += 1
        if tag == 1:
            size = struct.unpack_from(">H", data, pos)[0]
            pos += 2
            cp[i] = (tag, data[pos : pos + size].decode("utf-8", errors="replace"))
            pos += size
        elif tag in (3, 4):
            cp[i] = (tag, data[pos : pos + 4])
            pos += 4
        elif tag in (5, 6):
            cp[i] = (tag, data[pos : pos + 8])
            pos += 8
            i += 1
        elif tag in (7, 8, 16, 19, 20):
            cp[i] = (tag, struct.unpack_from(">H", data, pos)[0])
            pos += 2
        elif tag in (9, 10, 11, 12, 17, 18):
            cp[i] = (tag, *struct.unpack_from(">HH", data, pos))
            pos += 4
        elif tag == 15:
            cp[i] = (tag, data[pos], struct.unpack_from(">H", data, pos + 1)[0])
            pos += 3
        else:
            raise ValueError(f"unknown constant pool tag {tag}")
        i += 1

    def utf(index: int) -> str:
        item = cp[index]
        if not isinstance(item, tuple) or item[0] != 1:
            raise ValueError("bad UTF8 constant")
        return str(item[1])

    refs = []
    for item in cp:
        if isinstance(item, tuple) and item[0] in (10, 11):
            class_item = cp[int(item[1])]
            nt_item = cp[int(item[2])]
            if (
                not isinstance(class_item, tuple)
                or class_item[0] != 7
                or not isinstance(nt_item, tuple)
                or nt_item[0] != 12
            ):
                raise ValueError("bad method reference")
            refs.append((utf(int(class_item[1])), utf(int(nt_item[1])), utf(int(nt_item[2]))))
    return refs


def target_metrics(
    truth: np.ndarray, clusters: np.ndarray, target: str = "HSCs"
) -> dict[str, float | int]:
    positive = truth == target
    rows = []
    for cluster in np.unique(clusters):
        predicted = clusters == cluster
        tp = int(np.count_nonzero(positive & predicted))
        fp = int(np.count_nonzero(~positive & predicted))
        fn = int(np.count_nonzero(positive & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
        rows.append(
            (
                f1,
                recall,
                -int(cluster),
                int(cluster),
                tp,
                fp,
                fn,
                precision,
                recall,
                f1,
                f2,
                int(predicted.sum()),
            )
        )
    best = max(rows)
    return {
        "selected_cluster": best[3],
        "target_tp": best[4],
        "target_fp": best[5],
        "target_fn": best[6],
        "target_precision": best[7],
        "target_recall": best[8],
        "target_f1": best[9],
        "target_f2": best[10],
        "selected_cluster_size": best[11],
    }


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        stop = min(start + 1000, n_boot)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


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
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    parent = args.parent.resolve()
    source_repo = args.source_repo.resolve()
    release_jar = args.release_jar.resolve()
    java = args.java.resolve()
    data_path = args.data.resolve()
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs_dir = output / "runs"
    runs_dir.mkdir()

    parent_manifest = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    input_fcs = parent / "Nilsson_rare_all_events_markers.fcs"
    config = parent / "importConfig.txt"
    parent_matrix, parent_names = read_fcs(input_fcs)
    if parent_matrix.shape != (44140, 13) or len(parent_names) != 13:
        raise ValueError("parent input FCS contract failed")
    if sha256(release_jar) != sha256(
        Path(
            json.loads((parent / "xshift_execution.json").read_text(encoding="utf-8"))["command"][3]
        )
    ):
        raise ValueError("release JAR differs from parent")

    tag_commit = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "29-Jun-2017^{}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    source_text = subprocess.run(
        [
            "git",
            "-C",
            str(source_repo),
            "show",
            "29-Jun-2017^{}:src/vortex/clustering/XShiftClustering.java",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with zipfile.ZipFile(release_jar) as archive:
        shuffle_class = archive.read("util/Shuffle.class")
    methodrefs = class_methodrefs(shuffle_class)
    random_refs = sorted({ref for ref in methodrefs if ref[0] == "java/util/Random"})
    random_contract = {
        "tag_commit": tag_commit,
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source_has_shuffle_initialization": "new Shuffle<Datapoint>()).shuffleCopyArray"
        in source_text,
        "source_has_math_random_fallback": "Math.random() * (numCells)" in source_text,
        "shuffle_class_sha256": hashlib.sha256(shuffle_class).hexdigest(),
        "java_util_random_methodrefs": random_refs,
        "default_random_constructor_present": ("java/util/Random", "<init>", "()V") in methodrefs,
        "seeded_random_constructor_present": ("java/util/Random", "<init>", "(J)V") in methodrefs,
        "cli_seed_interface": False,
    }
    (output / "randomness_source_contract.json").write_text(
        json.dumps(random_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if (
        tag_commit != "fda75cf79980222da663185e2a6a72b442b9aff3"
        or not random_contract["source_has_shuffle_initialization"]
        or not random_contract["default_random_constructor_present"]
        or random_contract["seeded_random_constructor_present"]
    ):
        raise ValueError("randomness source contract failed")

    truth_frame = pd.read_csv(data_path)
    truth = truth_frame["label"].astype(str).to_numpy()
    if len(truth) != 44140 or np.count_nonzero(truth == "HSCs") != 358:
        raise ValueError("Nilsson truth contract failed")

    command = [str(java), "-Xmx4G", "-cp", str(release_jar), "standalone.Xshift", "20"]
    input_hashes = {
        "input_fcs": sha256(input_fcs),
        "config": sha256(config),
        "release_jar": sha256(release_jar),
        "java": sha256(java),
        "data": sha256(data_path),
    }
    (output / "input_hashes.json").write_text(
        json.dumps(input_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def run_one(repeat: int) -> dict[str, object]:
        run_dir = runs_dir / f"repeat{repeat:03d}"
        run_dir.mkdir()
        (run_dir / "importConfig.txt").write_bytes(config.read_bytes())
        (run_dir / "fcsFileList.txt").write_text(str(input_fcs) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command, cwd=run_dir, capture_output=True, text=True, timeout=args.timeout_seconds
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            result = None
            timed_out = True
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
        success = False
        detail = ""
        n_clusters = None
        labels_hash = None
        if not timed_out and exit_code == 0 and len(candidates) == 1:
            matrix, names = read_fcs(candidates[0])
            values = matrix[:, -1]
            success = (
                matrix.shape == (44140, 14)
                and names[:-1] == parent_names
                and names[-1].lower().replace("_", "") == "clusterid"
                and np.all(np.isfinite(values))
                and np.allclose(values, np.rint(values))
                and np.array_equal(matrix[:, :-1], parent_matrix)
            )
            if success:
                labels = np.rint(values).astype(np.int32)
                np.save(run_dir / "cluster_ids_all_events.npy", labels)
                labels_hash = sha256(run_dir / "cluster_ids_all_events.npy")
                n_clusters = int(np.unique(labels).size)
                unique, counts = np.unique(labels, return_counts=True)
                pd.DataFrame({"cluster_id": unique, "events": counts}).to_csv(
                    run_dir / "cluster_sizes.csv", index=False
                )
            detail = f"shape={matrix.shape}; names={names}; marker_exact={np.array_equal(matrix[:, :-1], parent_matrix)}"
        execution = {
            "repeat": repeat,
            "command": command,
            "cwd": str(run_dir),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "runtime_seconds": runtime,
            "output_fcs_count": len(candidates),
            "all_checks_passed": success,
            "detail": detail,
            "n_clusters": n_clusters,
            "labels_sha256": labels_hash,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return execution

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, repeat): repeat for repeat in range(30)}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                f"repeat={row['repeat']:03d} pass={row['all_checks_passed']} runtime={row['runtime_seconds']:.2f}s",
                flush=True,
            )
    results.sort(key=lambda row: int(row["repeat"]))
    pd.DataFrame(results).to_csv(output / "execution_index.csv", index=False)

    run_rows = []
    partitions = []
    for row in results:
        if not row["all_checks_passed"]:
            continue
        repeat = int(row["repeat"])
        labels = np.load(
            runs_dir / f"repeat{repeat:03d}" / "cluster_ids_all_events.npy", allow_pickle=False
        )
        partitions.append(labels)
        metrics = target_metrics(truth, labels)
        run_rows.append(
            {
                "repeat": repeat,
                "runtime_seconds": row["runtime_seconds"],
                "n_clusters": int(np.unique(labels).size),
                "partition_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                "ari": adjusted_rand_score(truth, labels),
                **metrics,
            }
        )
    run_frame = pd.DataFrame(run_rows)
    run_frame.to_csv(output / "run_level_metrics.csv", index=False)
    pair_rows = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            pair_rows.append(
                {
                    "repeat_a": int(run_frame.iloc[i]["repeat"]),
                    "repeat_b": int(run_frame.iloc[j]["repeat"]),
                    "partition_ari": adjusted_rand_score(partitions[i], partitions[j]),
                }
            )
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output / "pairwise_partition_ari.csv", index=False)
    summary_rows = []
    for idx, metric in enumerate(
        (
            "runtime_seconds",
            "n_clusters",
            "ari",
            "target_precision",
            "target_recall",
            "target_f1",
            "target_f2",
        )
    ):
        values = run_frame[metric].to_numpy(dtype=float)
        low, high = bootstrap_ci(values, 20260911 + idx)
        summary_rows.append(
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
    pd.DataFrame(summary_rows).to_csv(output / "endpoint_descriptive_statistics.csv", index=False)

    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    successful = int(sum(bool(row["all_checks_passed"]) for row in results))
    unique_hashes = run_frame.partition_sha256.nunique() if len(run_frame) else 0
    check(
        "parent_manifest",
        parent_manifest["all_checks_passed"] and parent_manifest["experiment_id"] == "EXP-010B",
        parent_manifest["experiment_id"],
    )
    check(
        "randomness_source",
        all(
            [
                random_contract["source_has_shuffle_initialization"],
                random_contract["source_has_math_random_fallback"],
                random_contract["default_random_constructor_present"],
                not random_contract["seeded_random_constructor_present"],
                not random_contract["cli_seed_interface"],
            ]
        ),
        json.dumps(random_contract),
    )
    check(
        "execution_accounting",
        len(results) == 30 and {int(row["repeat"]) for row in results} == set(range(30)),
        f"rows={len(results)}",
    )
    check(
        "all_runs_successful", successful == 30, f"success={successful}; failed={30 - successful}"
    )
    check(
        "run_metrics",
        len(run_frame) == successful and run_frame.repeat.nunique() == successful,
        f"rows={len(run_frame)}",
    )
    check(
        "pairwise_count",
        len(pair_frame) == successful * (successful - 1) // 2,
        f"rows={len(pair_frame)}",
    )
    check(
        "input_hash_uniformity",
        all(
            sha256(runs_dir / f"repeat{repeat:03d}" / "importConfig.txt") == input_hashes["config"]
            for repeat in range(30)
        ),
        "30/30 configs",
    )
    check(
        "labels_valid",
        len(run_frame) == 30 and run_frame.n_clusters.ge(1).all(),
        f"cluster_range={run_frame.n_clusters.min() if len(run_frame) else None}-{run_frame.n_clusters.max() if len(run_frame) else None}",
    )
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "checks.csv", index=False)
    pair_values = pair_frame.partition_ari.to_numpy(dtype=float)
    manifest = {
        "experiment_id": "EXP-038",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "parent": str(parent),
        "parent_manifest_sha256": sha256(parent / "run_manifest.json"),
        "source_repo": str(source_repo),
        "tag_commit": tag_commit,
        "release_jar": str(release_jar),
        "release_jar_sha256": sha256(release_jar),
        "java": str(java),
        "java_sha256": sha256(java),
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "workers": args.workers,
        "repeat_not_seed": True,
        "runs_attempted": 30,
        "runs_successful": successful,
        "runs_failed": 30 - successful,
        "unique_partition_hashes": int(unique_hashes),
        "pairwise_partition_ari_mean": float(pair_values.mean()),
        "pairwise_partition_ari_min": float(pair_values.min()),
        "pairwise_partition_ari_max": float(pair_values.max()),
        "checks_passed": int(checks_frame.passed.sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame.passed.all()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    f1 = run_frame.target_f1.to_numpy(dtype=float)
    (output / "scientific_summary.md").write_text(
        "# EXP-038 Thirty native X-shift repeats on Nilsson at fixed K=20\n\n"
        f"Successful runs: {successful}/30; unique all-event partition hashes: {unique_hashes}. The mean/range of pairwise partition ARI was {pair_values.mean():.6f}/[{pair_values.min():.6f}, {pair_values.max():.6f}]. "
        f"The mean/range of target F1 was {f1.mean():.6f}/[{f1.min():.6f}, {f1.max():.6f}]. "
        "The official CLI exposes no seed and the source contains unseeded random initialization, so these are interpreted only as independent-process native repeats.\n",
        encoding="utf-8",
    )
    artifacts = [path for path in output.rglob("*") if path.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {str(path.relative_to(output)): sha256(path) for path in artifacts},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
