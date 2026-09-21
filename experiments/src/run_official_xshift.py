from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import flowio
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def observed_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    return "comma" if first_line.count(",") > first_line.count("\t") else "tab"


def run(command: list[str], cwd: Path) -> dict:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def save_execution(output: Path, name: str, execution: dict) -> None:
    (output / f"{name}_stdout.log").write_text(execution["stdout"], encoding="utf-8")
    (output / f"{name}_stderr.log").write_text(execution["stderr"], encoding="utf-8")
    payload = {key: value for key, value in execution.items() if key not in {"stdout", "stderr"}}
    (output / f"{name}_execution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_fcs(path: Path) -> tuple[np.ndarray, list[str]]:
    data = flowio.FlowData(str(path))
    matrix = np.asarray(data.events, dtype=np.float64).reshape(data.event_count, data.channel_count)
    names = [
        data.channels[index].get("pnn")
        or data.channels[index].get("pns")
        or f"channel_{index}"
        for index in range(1, data.channel_count + 1)
    ]
    return matrix, names


def run_xshift_until_cluster_output(
    command: list[str],
    cwd: Path,
    expected_events: int,
    expected_marker_names: list[str],
    timeout_seconds: int,
) -> dict:
    """Run official X-shift, stopping only after its clustering export is complete.

    The official standalone entry point always continues from FCS export into an
    MST plus ForceAtlas2 layout.  That graph is not used for clustering metrics
    and becomes extremely expensive when X-shift produces hundreds of clusters.
    Source order in standalone.Xshift is exportClusterSet -> "Building MST Graph".
    We therefore stop only after both conditions are independently observed:
    (1) the log has entered the MST stage and (2) the exported FCS is complete and
    structurally valid.  A normal Java exit remains preferred when it occurs first.
    """
    started = time.perf_counter()
    stdout_path = cwd / "xshift_stdout.log"
    stderr_path = cwd / "xshift_stderr.log"
    completion_mode = "natural_exit"
    cluster_output_complete = False
    mst_stage_observed = False
    output_fcs: Path | None = None
    last_progress_report = -30

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout_seconds:
                completion_mode = "timeout"
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break

            if int(elapsed) - last_progress_report >= 30:
                print(
                    f"X-shift running: elapsed={elapsed:.0f}s; "
                    f"stage={'MST' if mst_stage_observed else 'clustering/export'}",
                    flush=True,
                )
                last_progress_report = int(elapsed)

            try:
                stderr_handle.flush()
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                mst_stage_observed = "Building MST Graph" in stderr_text
                output_candidates = sorted((cwd / "out").glob("*.fcs"))
                if mst_stage_observed and len(output_candidates) == 1:
                    candidate_matrix, candidate_names = read_fcs(output_candidates[0])
                    cluster_name_ok = (
                        candidate_names[-1].lower().replace("_", "") == "clusterid"
                    )
                    cluster_values = candidate_matrix[:, -1]
                    cluster_output_complete = bool(
                        candidate_matrix.shape
                        == (expected_events, len(expected_marker_names) + 1)
                        and candidate_names[:-1] == expected_marker_names
                        and cluster_name_ok
                        and np.all(np.isfinite(cluster_values))
                        and np.allclose(cluster_values, np.rint(cluster_values))
                    )
                    if cluster_output_complete:
                        output_fcs = output_candidates[0]
                        completion_mode = "stopped_after_cluster_output_before_mst_layout"
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
            except (OSError, ValueError, IndexError):
                # The exporter may still be writing; retry until it closes a valid FCS.
                pass
            time.sleep(1)

        exit_code = process.wait()

    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": exit_code,
        "runtime_seconds": time.perf_counter() - started,
        "completion_mode": completion_mode,
        "cluster_output_complete_before_stop": cluster_output_complete,
        "mst_stage_observed": mst_stage_observed,
        "validated_output_fcs_before_stop": str(output_fcs) if output_fcs else None,
        "timeout_seconds": timeout_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--release-zip", type=Path, required=True)
    parser.add_argument("--release-jar", type=Path, required=True)
    parser.add_argument("--csv2fcs-jar", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument(
        "--auto-trigger",
        choices=["literal_auto", "numeric_one"],
        default="literal_auto",
        help="The official CLI's literal 'auto' is defective; numeric_one enters its K<3 branch.",
    )
    parser.add_argument("--heap-gb", type=int, default=4)
    parser.add_argument("--xshift-timeout-seconds", type=int, default=21600)
    args = parser.parse_args()

    config_path = args.config.resolve()
    protocol = args.protocol.resolve()
    java = args.java.resolve()
    release_zip = args.release_zip.resolve()
    release_jar = args.release_jar.resolve()
    csv2fcs_jar = args.csv2fcs_jar.resolve()
    required_files = (config_path, protocol, java, release_zip, release_jar, csv2fcs_jar)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + "; ".join(missing))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset not in config: {args.dataset}")
    spec = config["datasets"][args.dataset]
    source = (Path(config["data_root"]) / args.dataset / spec["filename"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_separator = observed_separator(source)
    if actual_separator != spec["separator"]:
        raise ValueError(
            f"Separator mismatch: registered={spec['separator']}, observed={actual_separator}"
        )
    separator = "\t" if actual_separator == "tab" else ","
    frame = pd.read_csv(source, sep=separator)
    excluded = set(spec["exclude_columns"])
    marker_columns = [column for column in frame.columns if column not in excluded]
    if len(marker_columns) != int(spec["expected_markers"]):
        raise ValueError(
            f"Marker mismatch: expected={spec['expected_markers']}, observed={len(marker_columns)}"
        )
    marker_frame = frame[marker_columns]
    if not all(pd.api.types.is_numeric_dtype(marker_frame[column]) for column in marker_columns):
        raise TypeError("All clustering markers must be numeric")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_csv = output / f"{args.dataset}_all_events_markers.csv"
    marker_frame.to_csv(input_csv, index=False)
    input_fcs = output / f"{args.dataset}_all_events_markers.fcs"
    config_lines = [
        "clustering_columns=" + ",".join(str(index) for index in range(1, len(marker_columns) + 1)),
        "limit_events_per_file=-1",
        "transformation=" + ("NONE" if spec["cofactor"] is None else "ASINH"),
        "scaling_factor=" + ("1" if spec["cofactor"] is None else str(spec["cofactor"])),
        "noise_threshold=0.0",
        "euclidian_length_threshold=0.0",
        "rescale=NONE",
        "quantile=0.95",
        "rescale_separately=false",
    ]
    (output / "importConfig.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    (output / "fcsFileList.txt").write_text(str(input_fcs) + "\n", encoding="utf-8")

    convert = run(
        [
            str(java),
            "-jar",
            str(csv2fcs_jar),
            f"-InputFile:{input_csv}",
            f"-OutputFile:{input_fcs}",
            "-DataType:float",
            "-Range:auto-exact",
        ],
        output,
    )
    save_execution(output, "csv2fcs", convert)
    xshift_argument = (
        ("1" if args.auto_trigger == "numeric_one" else "auto")
        if args.auto
        else str(args.k)
    )
    xshift = run_xshift_until_cluster_output(
        [
            str(java),
            f"-Xmx{args.heap_gb}G",
            "-cp",
            str(release_jar),
            "standalone.Xshift",
            xshift_argument,
        ],
        output,
        expected_events=len(frame),
        expected_marker_names=marker_columns,
        timeout_seconds=args.xshift_timeout_seconds,
    )
    (output / "xshift_execution.json").write_text(
        json.dumps(xshift, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    input_matrix = None
    input_names: list[str] = []
    input_error = None
    try:
        input_matrix, input_names = read_fcs(input_fcs)
    except Exception as error:
        input_error = repr(error)
    output_fcs_files = sorted((output / "out").glob("*.fcs")) if (output / "out").is_dir() else []
    output_matrix = None
    output_names: list[str] = []
    output_error = None
    if len(output_fcs_files) == 1:
        try:
            output_matrix, output_names = read_fcs(output_fcs_files[0])
        except Exception as error:
            output_error = repr(error)

    cluster_ids = None
    cluster_index = None
    if output_matrix is not None:
        cluster_indices = [
            index
            for index, name in enumerate(output_names)
            if name.lower().replace("_", "") == "clusterid"
        ]
        if len(cluster_indices) == 1:
            cluster_index = cluster_indices[0]
            cluster_values = output_matrix[:, cluster_index]
            if np.allclose(cluster_values, np.rint(cluster_values)):
                cluster_ids = np.rint(cluster_values).astype(np.int32)

    marker_values_exact = False
    if input_matrix is not None and output_matrix is not None and cluster_index is not None:
        output_markers = np.delete(output_matrix, cluster_index, axis=1)
        marker_values_exact = bool(np.array_equal(input_matrix, output_markers))

    xshift_stderr_text = (output / "xshift_stderr.log").read_text(
        encoding="utf-8", errors="replace"
    )
    automatic_scan = [
        {"k": int(k), "predicted_clusters": int(float(clusters))}
        for k, clusters in re.findall(
            r"Trying X-shift, K = (\d+), got (\d+(?:\.\d+)?) clusters",
            xshift_stderr_text,
        )
    ]
    automatic_selected_k_candidates: list[int] = []
    if args.auto and cluster_ids is not None:
        output_cluster_count = int(np.unique(cluster_ids[cluster_ids >= 0]).size)
        automatic_selected_k_candidates = [
            row["k"]
            for row in automatic_scan
            if row["predicted_clusters"] == output_cluster_count
        ]

    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("source_contract", actual_separator == spec["separator"], f"separator={actual_separator}")
    check("input_has_no_label", "label" not in marker_columns, str(marker_columns))
    check("marker_count", len(marker_columns) == spec["expected_markers"], str(len(marker_columns)))
    check("csv2fcs_exit_zero", convert["exit_code"] == 0, str(convert["exit_code"]))
    check(
        "input_fcs_valid",
        input_matrix is not None
        and input_matrix.shape == (len(frame), len(marker_columns))
        and input_names == marker_columns,
        json.dumps(
            {"shape": list(input_matrix.shape) if input_matrix is not None else None,
             "names": input_names, "error": input_error}, ensure_ascii=False
        ),
    )
    xshift_cluster_stage_complete = bool(
        xshift["exit_code"] == 0
        or (
            xshift["completion_mode"]
            == "stopped_after_cluster_output_before_mst_layout"
            and xshift["cluster_output_complete_before_stop"]
            and xshift["mst_stage_observed"]
        )
    )
    check(
        "xshift_cluster_stage_complete",
        xshift_cluster_stage_complete,
        json.dumps(
            {
                "exit_code": xshift["exit_code"],
                "completion_mode": xshift["completion_mode"],
                "cluster_output_complete_before_stop": xshift[
                    "cluster_output_complete_before_stop"
                ],
                "mst_stage_observed": xshift["mst_stage_observed"],
            }
        ),
    )
    check("one_output_fcs", len(output_fcs_files) == 1, str([str(path) for path in output_fcs_files]))
    check(
        "output_fcs_valid",
        output_matrix is not None
        and output_matrix.shape == (len(frame), len(marker_columns) + 1)
        and output_names[: len(marker_columns)] == marker_columns,
        json.dumps(
            {"shape": list(output_matrix.shape) if output_matrix is not None else None,
             "names": output_names, "error": output_error}, ensure_ascii=False
        ),
    )
    check(
        "cluster_ids_valid",
        cluster_ids is not None
        and int(cluster_ids.min()) >= -1
        and int(np.unique(cluster_ids[cluster_ids >= 0]).size) >= 2,
        json.dumps(
            {"min": int(cluster_ids.min()) if cluster_ids is not None else None,
             "clusters": int(np.unique(cluster_ids[cluster_ids >= 0]).size)
             if cluster_ids is not None else None}
        ),
    )
    if args.auto:
        check(
            "automatic_scan_complete",
            len(automatic_scan) == 30,
            json.dumps(
                {
                    "scan_points": len(automatic_scan),
                    "selected_k_candidates_from_output_cluster_count": automatic_selected_k_candidates,
                }
            ),
        )
    check("marker_values_exact_input_to_output", marker_values_exact, str(marker_values_exact))
    pd.DataFrame(checks).to_csv(output / "qualification_checks.csv", index=False)

    if cluster_ids is not None:
        np.save(output / "cluster_ids_all_events.npy", cluster_ids)
        unique, counts = np.unique(cluster_ids, return_counts=True)
        pd.DataFrame({"cluster_id": unique, "event_count": counts}).to_csv(
            output / "cluster_sizes_all_events.csv", index=False
        )

    java_version = run([str(java), "-version"], output)
    save_execution(output, "java_version", java_version)
    artifacts = {
        str(config_path): sha256(config_path),
        str(protocol): sha256(protocol),
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        str(source): sha256(source),
        str(input_csv): sha256(input_csv),
        str(input_fcs): sha256(input_fcs) if input_fcs.is_file() else None,
        str(release_zip): sha256(release_zip),
        str(release_jar): sha256(release_jar),
        str(csv2fcs_jar): sha256(csv2fcs_jar),
        str(output_fcs_files[0]): sha256(output_fcs_files[0]) if len(output_fcs_files) == 1 else None,
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "source": str(source),
        "source_sha256": sha256(source),
        "total_events": len(frame),
        "markers": marker_columns,
        "n_markers": len(marker_columns),
        "cofactor": spec["cofactor"],
        "xshift_mode": "automatic_elbow" if args.auto else "fixed_k",
        "xshift_auto_trigger": args.auto_trigger if args.auto else None,
        "xshift_requested_argument": xshift_argument,
        "xshift_knn_k": None if args.auto else args.k,
        "automatic_scan": automatic_scan if args.auto else None,
        "automatic_selected_k_candidates_from_output_cluster_count": automatic_selected_k_candidates
        if args.auto
        else None,
        "automatic_selected_k_inferred": automatic_selected_k_candidates[0]
        if args.auto and len(automatic_selected_k_candidates) == 1
        else None,
        "heap_gb": args.heap_gb,
        "fit_policy": "all_events",
        "label_used_for_fit_or_selection": False,
        "predicted_clusters": int(np.unique(cluster_ids[cluster_ids >= 0]).size)
        if cluster_ids is not None else None,
        "unassigned_events": int(np.sum(cluster_ids < 0)) if cluster_ids is not None else None,
        "csv2fcs_runtime_seconds": convert["runtime_seconds"],
        "xshift_runtime_seconds": xshift["runtime_seconds"],
        "xshift_completion_mode": xshift["completion_mode"],
        "mst_postprocess_required_for_metrics": False,
        "mst_stage_observed": xshift["mst_stage_observed"],
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "flowio_version": importlib.metadata.version("flowio"),
        "java_version": (java_version["stdout"] + java_version["stderr"]).strip(),
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    regime_text = (
        "官方命令行自动肘点"
        if args.auto
        else f"固定K={args.k}"
    )
    audit = [
        f"# {args.experiment_id} 官方X-shift {args.dataset}{regime_text}全量运行",
        "",
        f"事件={len(frame):,}；标记物={len(marker_columns)}；cofactor={spec['cofactor']}；预测簇={manifest['predicted_clusters']}。",
        f"CSV→FCS耗时={convert['runtime_seconds']:.3f}s；X-shift耗时={xshift['runtime_seconds']:.3f}s。",
        f"X-shift完成模式={xshift['completion_mode']}；MST后处理不进入聚类评价。",
        f"参数制度={regime_text}；自动扫描点={len(automatic_scan)}；自动K候选={automatic_selected_k_candidates if args.auto else '不适用'}。",
        f"输入输出标记物逐值一致={marker_values_exact}；全部检查={'通过' if manifest['all_checks_passed'] else '失败'}。",
        "",
        "标签未进入拟合或参数选择；本运行不包含外部评价、K扫描或稳定性推断。",
    ]
    (output / "audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
