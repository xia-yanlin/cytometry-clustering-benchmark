from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_xshift_cross_dataset import evaluate_multiclass, metrics_are_valid


CONTRACTS = {
    "Samusik_01": {"rows": 86864, "markers": 39, "evaluable": 53173, "populations": 24, "sha": "86dd329dedec30b86ab0cc845923620393a36f16bd07d8f5d87ba4dbb3511565"},
    "Levine_32dim": {"rows": 265627, "markers": 32, "evaluable": 104184, "populations": 14, "sha": "bb1f9cb63377f701795f116e557f8ede05141b6aca2a0270177c7b1eb4c807e4"},
}
SEEDS = tuple(range(30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_source_contract(config_path: Path, dataset: str):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["datasets"][dataset]
    source = (Path(config["data_root"]) / dataset / spec["filename"]).resolve()
    header = pd.read_csv(source, sep={"tab": "\t", "comma": ","}[spec["separator"]], nrows=0)
    markers = [column for column in header.columns if column not in set(spec["exclude_columns"])]
    return source, markers, float(spec["cofactor"]), {"tab": "\t", "comma": ","}[spec["separator"]]


def load_reference_labels(source: Path, separator: str):
    frame = pd.read_csv(source, sep=separator, usecols=["label"], low_memory=False)
    labels = frame["label"].astype(str).to_numpy()
    evaluable = labels != "unassigned"
    return labels, evaluable


def rebuild_index(output: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        run = output / "runs" / f"seed{seed:03d}"
        if (run / "run_manifest.json").is_file():
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            metrics = pd.read_csv(run / "run_level_metrics.csv").iloc[0].to_dict()
            rows.append({"run_name": run.name, "seed": seed, "runtime_seconds": manifest["r_runtime_seconds"], "all_checks_passed": manifest["all_checks_passed"], **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "run_index.csv", index=False)
    return frame


def progress(output: Path, experiment_id: str, protocol: Path, source: Path, r_script: Path, r_library: Path, failed: str | None = None) -> dict:
    index = rebuild_index(output)
    passed = int(index["all_checks_passed"].astype(bool).sum()) if len(index) else 0
    value = {
        "experiment_id": experiment_id, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failed else ("complete" if len(index) == passed == 30 else "in_progress"),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "orchestrator": str(Path(__file__).resolve()), "orchestrator_sha256": sha256(Path(__file__).resolve()),
        "r_script": str(r_script), "r_script_sha256": sha256(r_script),
        "r_library": str(r_library), "source": str(source), "source_sha256": sha256(source),
        "method_identity": "pure_R_FlowSOM_2.18.0_clustering_plus_Python_readonly_evaluation",
        "fit_policy": "all_events_direct_training", "xdim": 10, "ydim": 10, "rlen": 30,
        "requested_metaclusters": 40, "seeds": list(SEEDS), "expected_runs": 30,
        "completed_runs": len(index), "passed_runs": passed, "failed_run": failed,
        "all_runs_complete_and_passed": len(index) == passed == 30 and failed is None,
    }
    write_json(output / "progress_manifest.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--r-script", type=Path, required=True)
    parser.add_argument("--rscript-exe", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-runs", type=int, default=0)
    args = parser.parse_args()
    config = args.config.resolve(); protocol = args.protocol.resolve(); r_script = args.r_script.resolve()
    rscript_exe = args.rscript_exe.resolve(); r_library = args.r_library.resolve(); output = args.output.resolve()
    contract = CONTRACTS[args.dataset]
    source, markers, cofactor, separator = load_source_contract(config, args.dataset)
    if sha256(source) != contract["sha"] or len(markers) != contract["markers"]:
        raise RuntimeError("source or marker contract failed")
    if output.exists():
        old = json.loads((output / "progress_manifest.json").read_text(encoding="utf-8")) if (output / "progress_manifest.json").is_file() else None
        if old is not None and (old.get("protocol_sha256") != sha256(protocol) or old.get("experiment_id") != args.experiment_id):
            raise RuntimeError("resume contract differs")
    else:
        (output / "runs").mkdir(parents=True)
        (output / "marker_columns.txt").write_text("\n".join(markers) + "\n", encoding="utf-8")
    source_alias = output / "input_source_ascii.txt"
    if not source_alias.exists():
        os.link(source, source_alias)
    if source_alias.stat().st_size != source.stat().st_size or sha256(source_alias) != sha256(source):
        raise RuntimeError("ASCII hard-link source alias is not byte-identical")
    progress(output, args.experiment_id, protocol, source, r_script, r_library)
    env = os.environ.copy(); env["R_LIBS_USER"] = str(r_library)
    command = [str(rscript_exe), str(r_script), str(source_alias), str(output / "marker_columns.txt"), str(cofactor), args.dataset, str(output), str(args.max_new_runs), sha256(protocol)]
    with (output / "r_batch_stdout.log").open("a", encoding="utf-8") as stdout, (output / "r_batch_stderr.log").open("a", encoding="utf-8") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        progress(output, args.experiment_id, protocol, source, r_script, r_library, "R_batch")
        raise RuntimeError(f"R batch failed with exit code {result.returncode}")

    labels, evaluable = load_reference_labels(source, separator)
    if len(labels) != contract["rows"] or int(evaluable.sum()) != contract["evaluable"] or len(np.unique(labels[evaluable])) != contract["populations"]:
        raise RuntimeError("postfit reference label contract failed")

    for seed in SEEDS:
        run = output / "runs" / f"seed{seed:03d}"
        if not (run / "r_run_manifest.json").is_file() or (run / "run_manifest.json").is_file():
            continue
        r_manifest = json.loads((run / "r_run_manifest.json").read_text(encoding="utf-8"))
        r_checks = pd.read_csv(run / "r_checks.csv")
        node_labels = np.fromfile(run / "event_node_labels.int32le", dtype="<i4")
        event_clusters = np.fromfile(run / "event_metacluster_labels.int32le", dtype="<i4")
        codes = pd.read_csv(run / "som_codes.csv").to_numpy(dtype=float)
        node_meta = pd.read_csv(run / "node_metaclusters.csv")
        metrics, populations, clusters, mapping, contingency = evaluate_multiclass(labels[evaluable], event_clusters[evaluable], event_clusters)
        metric_ok, metric_detail = metrics_are_valid(metrics)
        checks = []
        def check(name: str, passed: bool, detail: str) -> None:
            checks.append({"check": name, "passed": bool(passed), "detail": detail})
        check("r_manifest", r_manifest.get("all_checks_passed") is True and r_manifest.get("package_version") == "2.18.0", str(r_manifest.get("package_version")))
        check("r_checks", r_checks["passed"].astype(str).str.lower().eq("true").all() and len(r_checks) == 10, f"passed={int(r_checks['passed'].astype(str).str.lower().eq('true').sum())}/10")
        check("event_shapes", node_labels.shape == event_clusters.shape == (contract["rows"],), str(event_clusters.shape))
        check("node_range", node_labels.min() >= 1 and node_labels.max() <= 100, str((node_labels.min(), node_labels.max())))
        check("metacluster_range", event_clusters.min() >= 1 and event_clusters.max() <= 40, str((event_clusters.min(), event_clusters.max())))
        check("code_shape", codes.shape == (100, contract["markers"]) and np.isfinite(codes).all(), str(codes.shape))
        check("node_mapping", len(node_meta) == 100 and np.array_equal(event_clusters, node_meta["metacluster_label_1based"].to_numpy(dtype=np.int32)[node_labels - 1]), "event labels reconstructed from R node outputs")
        check("evaluation_contract", metrics["n_total_events"] == contract["rows"] and metrics["n_evaluable_events"] == contract["evaluable"] and metrics["n_true_populations"] == contract["populations"], f"events={metrics['n_total_events']}; evaluable={metrics['n_evaluable_events']}")
        check("metric_ranges", metric_ok, metric_detail)
        check("labels_permission", r_manifest.get("labels_used_for_fit") is False, "labels loaded only after R batch")
        check_frame = pd.DataFrame(checks); check_frame.to_csv(run / "checks.csv", index=False)
        r_summary = pd.read_csv(run / "r_run_summary.csv").iloc[0].to_dict()
        pd.DataFrame([{**r_summary, "algorithm": "R_FlowSOM_2.18.0_full_event_fixedK40", "evaluation_scope": "all_events_fit_evaluable_reference_labels_only", **metrics}]).to_csv(run / "run_level_metrics.csv", index=False)
        populations.to_csv(run / "population_level_metrics.csv", index=False); clusters.to_csv(run / "metacluster_level_diagnostics.csv", index=False)
        mapping.to_csv(run / "hungarian_mapping.csv", index=False); contingency.to_csv(run / "contingency_true_by_predicted.csv", index_label="true_population")
        manifest = {
            "experiment_id": args.experiment_id, "dataset": args.dataset, "seed": seed,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "source_sha256": sha256(source),
            "r_input_ascii_hardlink": str(source_alias), "r_input_ascii_hardlink_sha256": sha256(source_alias),
            "protocol": str(protocol), "protocol_sha256": sha256(protocol),
            "orchestrator": str(Path(__file__).resolve()), "orchestrator_sha256": sha256(Path(__file__).resolve()),
            "r_script": str(r_script), "r_script_sha256": sha256(r_script),
            "r_run_manifest_sha256": sha256(run / "r_run_manifest.json"),
            "method_identity": "pure_R_FlowSOM_2.18.0_clustering_plus_Python_readonly_evaluation",
            "fit_policy": "all_events_direct_training", "labels_used_for_fit": False,
            "r_runtime_seconds": float(r_manifest["runtime_seconds"]),
            "packages_for_evaluation": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
            "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "checks_passed": int(check_frame["passed"].sum()), "checks_total": len(check_frame), "all_checks_passed": bool(check_frame["passed"].all()),
        }
        write_json(run / "run_manifest.json", manifest)
        write_json(run / "artifact_hashes.json", {path.name: sha256(path) for path in run.iterdir() if path.is_file() and path.name != "artifact_hashes.json"})
        if not manifest["all_checks_passed"]:
            progress(output, args.experiment_id, protocol, source, r_script, r_library, run.name)
            raise RuntimeError(f"checks failed for {run.name}")
        print(json.dumps({"dataset": args.dataset, "seed": seed, "r_runtime_seconds": manifest["r_runtime_seconds"], "ari": metrics["ari"], "macro_f1": metrics["macro_f1"], "checks": "10/10"}, ensure_ascii=False), flush=True)
        progress(output, args.experiment_id, protocol, source, r_script, r_library)
    final = progress(output, args.experiment_id, protocol, source, r_script, r_library)
    if final["all_runs_complete_and_passed"]:
        write_json(output / "artifact_hashes.json", {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"})
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
