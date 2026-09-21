from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import flowsom as fs
import numpy as np
import pandas as pd
from flowsom.models import map_data_to_codes
from loguru import logger

from evaluate_xshift_cross_dataset import evaluate_multiclass, metrics_are_valid
from run_flowsom_grid_rlen_factorial import load_data, sha256, write_json
from run_flowsom_nested_marker_dimension import (
    CHAINS,
    DIMS,
    EVALUABLE_EVENTS,
    SEEDS,
    SOURCE_SHA,
    TOTAL_EVENTS,
    TRUE_POPULATIONS,
    build_subsets,
    expected_names,
    run_name,
)


EXPERIMENT_ID = "EXP-018-R1"


def rebuild_index(output: Path) -> pd.DataFrame:
    rows = []
    for name in expected_names():
        run = output / "runs" / name
        if not (run / "run_manifest.json").is_file():
            continue
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(run / "run_level_metrics.csv").iloc[0].to_dict()
        rows.append({"run_name": name, "chain": manifest["marker_chain"], "n_markers": manifest["n_markers"], "seed": manifest["seed"], "runtime_seconds": manifest["runtime_seconds"], "all_checks_passed": manifest["all_checks_passed"], **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "run_index.csv", index=False)
    return frame


def write_progress(output: Path, protocol: Path, script: Path, qualification: Path, source: Path, training_indices: Path, subsets: Path, failed_parent: Path, failed_run: str | None = None) -> dict:
    index = rebuild_index(output)
    completed = len(index)
    passed = int(index["all_checks_passed"].astype(bool).sum()) if completed else 0
    value = {
        "experiment_id": EXPERIMENT_ID,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failed_run else ("complete" if completed == 125 and passed == 125 else "in_progress"),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "script": str(script), "script_sha256": sha256(script),
        "qualification_run": str(qualification), "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
        "source": str(source), "source_sha256": sha256(source),
        "training_indices": str(training_indices), "training_indices_sha256": sha256(training_indices),
        "marker_subsets": str(subsets), "marker_subsets_sha256": sha256(subsets),
        "failed_parent": str(failed_parent), "failed_parent_manifest_sha256": sha256(failed_parent / "failure_manifest.json"),
        "expected_runs": 125, "completed_runs": completed, "passed_runs": passed, "failed_run": failed_run,
        "all_runs_complete_and_passed": bool(completed == 125 and passed == 125 and failed_run is None),
    }
    write_json(output / "progress_manifest.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification-run", type=Path, required=True)
    parser.add_argument("--failed-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-runs", type=int, default=None)
    args = parser.parse_args()

    config, protocol, qualification = args.config.resolve(), args.protocol.resolve(), args.qualification_run.resolve()
    failed_parent, output, script = args.failed_parent.resolve(), args.output.resolve(), Path(__file__).resolve()
    qualification_manifest = json.loads((qualification / "run_manifest.json").read_text(encoding="utf-8"))
    failed_manifest = json.loads((failed_parent / "failure_manifest.json").read_text(encoding="utf-8"))
    if not qualification_manifest.get("all_checks_passed") or qualification_manifest.get("flowsom_version") != "0.2.2":
        raise RuntimeError("EXP-013-R2 qualification is not eligible")
    if failed_manifest.get("scientific_output_eligible") is not False or importlib.metadata.version("flowsom") != "0.2.2":
        raise RuntimeError("Failed-parent or software contract mismatch")

    source, _, markers, matrix, labels = load_data(config, "Levine_32dim")
    evaluable = labels != "unassigned"
    if sha256(source) != SOURCE_SHA or matrix.shape != (TOTAL_EVENTS, 32) or int(evaluable.sum()) != EVALUABLE_EVENTS or len(np.unique(labels[evaluable])) != TRUE_POPULATIONS:
        raise RuntimeError("Data contract failed")
    if output.exists():
        prior = json.loads((output / "progress_manifest.json").read_text(encoding="utf-8"))
        if prior["protocol_sha256"] != sha256(protocol) or prior["script_sha256"] != sha256(script):
            raise RuntimeError("Cannot resume changed protocol or script")
    else:
        output.mkdir(parents=True)
        (output / "runs").mkdir()

    training_indices_path = output / "training_indices.npy"
    if training_indices_path.is_file():
        training_indices = np.load(training_indices_path, allow_pickle=False)
    else:
        training_indices = np.sort(np.random.default_rng(20260910).choice(TOTAL_EVENTS, size=20000, replace=False)).astype(np.int64)
        np.save(training_indices_path, training_indices)
    subsets_path = output / "marker_permutations.csv"
    expected_subsets = build_subsets(markers)
    if subsets_path.is_file():
        subset_table = pd.read_csv(subsets_path)
    else:
        subset_table = expected_subsets
        subset_table.to_csv(subsets_path, index=False)
    if not subset_table.equals(expected_subsets) or training_indices.shape != (20000,) or np.unique(training_indices).size != 20000:
        raise RuntimeError("Frozen subset or training-index contract failed")

    write_progress(output, protocol, script, qualification, source, training_indices_path, subsets_path, failed_parent)
    logger.disable("flowsom")
    marker_to_column = {marker: position for position, marker in enumerate(markers)}
    new_runs = 0
    for chain in CHAINS:
        permutation = subset_table.loc[subset_table["chain"].eq(chain)].sort_values("rank")["marker"].tolist()
        for dim in DIMS:
            member_set = set(permutation[:dim])
            selected_markers = [marker for marker in markers if marker in member_set]
            columns = [marker_to_column[marker] for marker in selected_markers]
            selected_matrix = matrix[:, columns]
            training_matrix = selected_matrix[training_indices]
            for seed in SEEDS:
                name = run_name(chain, dim, seed)
                run_dir = output / "runs" / name
                if (run_dir / "run_manifest.json").is_file():
                    continue
                if run_dir.exists():
                    raise RuntimeError(f"Incomplete run requires a new revision: {run_dir}")
                if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                    print(json.dumps(write_progress(output, protocol, script, qualification, source, training_indices_path, subsets_path, failed_parent), ensure_ascii=False))
                    return 0
                run_dir.mkdir()
                try:
                    started = time.perf_counter()
                    model = fs.FlowSOM(ad.AnnData(X=pd.DataFrame(training_matrix, columns=selected_markers)), cols_to_use=selected_markers, n_clusters=40, xdim=10, ydim=10, rlen=30, seed=seed)
                    codes = np.asarray(model.model.codes, dtype=np.float64)
                    node_to_meta = np.asarray(model.model._y_codes, dtype=np.int16)
                    node_float, distances = map_data_to_codes(selected_matrix, codes)
                    nodes = node_float.astype(np.int16)
                    metas = node_to_meta[nodes].astype(np.int16)
                    distances = np.asarray(distances, dtype=np.float32)
                    runtime = time.perf_counter() - started
                    np.save(run_dir / "node_labels_all_events.npy", nodes)
                    np.save(run_dir / "metacluster_labels_all_events.npy", metas)
                    np.save(run_dir / "bmu_distances_all_events.npy", distances)
                    np.save(run_dir / "som_codes.npy", codes)
                    np.save(run_dir / "node_to_metacluster.npy", node_to_meta)
                    (run_dir / "selected_markers.txt").write_text("\n".join(selected_markers) + "\n", encoding="utf-8")

                    metrics, population, clusters, mapping, contingency = evaluate_multiclass(labels[evaluable], metas[evaluable], metas)
                    metrics_ok, metrics_detail = metrics_are_valid(metrics)
                    level = {"dataset": "Levine_32dim", "algorithm": "official_Python_FlowSOM_0.2.2", "marker_chain": chain, "n_markers": dim, "seed": seed, "grid_side": 10, "rlen": 30, "n_metaclusters_requested": 40, "mean_bmu_distance": float(np.mean(distances)), "median_bmu_distance": float(np.median(distances)), "median_bmu_distance_per_sqrt_marker": float(np.median(distances) / np.sqrt(dim)), "q90_bmu_distance": float(np.quantile(distances, 0.9)), "occupied_nodes_all_events": int(np.unique(nodes).size), "occupied_metaclusters_all_events": int(np.unique(metas).size), **metrics}
                    previous_dim = DIMS[DIMS.index(dim) - 1] if dim != DIMS[0] else None
                    previous_members = set(permutation[:previous_dim]) if previous_dim else set()
                    checks = [
                        ("qualification_link", qualification_manifest["all_checks_passed"], str(qualification / "run_manifest.json")),
                        ("source_hash", sha256(source) == SOURCE_SHA, sha256(source)),
                        ("training_indices", len(training_indices) == 20000 and np.unique(training_indices).size == 20000, sha256(training_indices_path)),
                        ("marker_membership", len(member_set) == dim and previous_members.issubset(member_set), json.dumps(sorted(member_set))),
                        ("canonical_marker_order", selected_markers == [marker for marker in markers if marker in member_set], json.dumps(selected_markers)),
                        ("event_shapes", nodes.shape == metas.shape == distances.shape == (TOTAL_EVENTS,), str(nodes.shape)),
                        ("node_contract", codes.shape == (100, dim) and node_to_meta.shape == (100,) and int(nodes.min()) >= 0 and int(nodes.max()) < 100, f"codes={codes.shape}; nodes={int(nodes.min())}-{int(nodes.max())}"),
                        ("metacluster_contract", len(np.unique(node_to_meta)) == 40 and set(np.unique(metas)).issubset(set(np.unique(node_to_meta))), f"node_meta={len(np.unique(node_to_meta))}; occupied={len(np.unique(metas))}"),
                        ("finite_outputs", np.isfinite(codes).all() and np.isfinite(distances).all() and bool(np.all(distances >= 0)), f"range={float(distances.min())}-{float(distances.max())}"),
                        ("evaluation_contract", metrics["n_evaluable_events"] == EVALUABLE_EVENTS and metrics["n_true_populations"] == TRUE_POPULATIONS, f"events={metrics['n_evaluable_events']}; populations={metrics['n_true_populations']}"),
                        ("metric_ranges", metrics_ok, metrics_detail),
                    ]
                    checks_frame = pd.DataFrame([{"check": key, "passed": bool(passed), "detail": detail} for key, passed, detail in checks])
                    checks_frame.to_csv(run_dir / "qualification_checks.csv", index=False)
                    pd.DataFrame([level]).to_csv(run_dir / "run_level_metrics.csv", index=False)
                    population.to_csv(run_dir / "population_level_metrics.csv", index=False)
                    clusters.to_csv(run_dir / "metacluster_level_diagnostics.csv", index=False)
                    mapping.to_csv(run_dir / "hungarian_mapping.csv", index=False)
                    contingency.to_csv(run_dir / "contingency_true_by_predicted.csv", index_label="true_population")
                    manifest = {
                        "experiment_id": EXPERIMENT_ID, "run_name": name, "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset": "Levine_32dim", "source": str(source), "source_sha256": sha256(source),
                        "protocol": str(protocol), "protocol_sha256": sha256(protocol), "script": str(script), "script_sha256": sha256(script),
                        "qualification_run": str(qualification), "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
                        "failed_parent": str(failed_parent), "failed_parent_manifest_sha256": sha256(failed_parent / "failure_manifest.json"),
                        "training_indices_sha256": sha256(training_indices_path), "marker_permutations_sha256": sha256(subsets_path),
                        "marker_chain": chain, "permutation_seed": 270000 + chain, "n_markers": dim, "selected_markers": selected_markers, "seed": seed,
                        "grid_side": 10, "rlen": 30, "n_metaclusters_requested": 40,
                        "fit_policy": "fixed_unlabeled_20000_training_sample_then_map_all_events", "labels_used_for_fit_sampling_or_parameter_selection": False,
                        "runtime_seconds": runtime, "flowsom_version": importlib.metadata.version("flowsom"), "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame["passed"].all()),
                    }
                    write_json(run_dir / "run_manifest.json", manifest)
                    (run_dir / "audit.md").write_text(f"# {EXPERIMENT_ID} {name}\n\nchain={chain}，dim={dim}，seed={seed}；marker集合按随机嵌套链选择、按源文件列序进入模型；标签仅用于事后评价。\nARI={metrics['ari']:.6f}，Macro F1={metrics['macro_f1']:.6f}，median BMU/sqrt(d)={level['median_bmu_distance_per_sqrt_marker']:.6f}；检查{manifest['checks_passed']}/{manifest['checks_total']}。\n", encoding="utf-8")
                    write_json(run_dir / "artifact_hashes.json", {str(path): sha256(path) for path in run_dir.iterdir() if path.is_file()})
                    if not manifest["all_checks_passed"]:
                        raise RuntimeError(f"Integrity checks failed: {name}")
                    new_runs += 1
                    print(json.dumps({"run": name, "runtime_seconds": runtime, "ari": metrics["ari"], "macro_f1": metrics["macro_f1"], "checks": f"{manifest['checks_passed']}/{manifest['checks_total']}"}), flush=True)
                except Exception as error:
                    (run_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    write_json(run_dir / "failure.json", {"run_name": name, "exception": repr(error), "scientific_output_eligible": False})
                    print(json.dumps(write_progress(output, protocol, script, qualification, source, training_indices_path, subsets_path, failed_parent, failed_run=name), ensure_ascii=False), file=sys.stderr)
                    return 1
                write_progress(output, protocol, script, qualification, source, training_indices_path, subsets_path, failed_parent)

    progress = write_progress(output, protocol, script, qualification, source, training_indices_path, subsets_path, failed_parent)
    print(json.dumps(progress, ensure_ascii=False))
    return 0 if progress["all_runs_complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
