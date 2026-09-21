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
from evaluate_xshift_cross_dataset import evaluate_multiclass, metrics_are_valid
from flowsom.models import map_data_to_codes
from loguru import logger
from run_flowsom_grid_rlen_factorial import load_data, sha256, write_json

DIMS = [8, 13, 20, 26, 32]
CHAINS = list(range(5))
SEEDS = list(range(5))
TOTAL_EVENTS = 265627
EVALUABLE_EVENTS = 104184
TRUE_POPULATIONS = 14
SOURCE_SHA = "bb1f9cb63377f701795f116e557f8ede05141b6aca2a0270177c7b1eb4c807e4"


def run_name(chain: int, dim: int, seed: int) -> str:
    return f"chain{chain:02d}_dim{dim:02d}_seed{seed:03d}"


def expected_names() -> list[str]:
    return [run_name(chain, dim, seed) for chain in CHAINS for dim in DIMS for seed in SEEDS]


def build_subsets(markers: list[str]) -> pd.DataFrame:
    rows = []
    for chain in CHAINS:
        permutation = np.random.default_rng(270000 + chain).permutation(markers).tolist()
        for rank, marker in enumerate(permutation, start=1):
            rows.append(
                {"chain": chain, "permutation_seed": 270000 + chain, "rank": rank, "marker": marker}
            )
    return pd.DataFrame(rows)


def rebuild_index(output: Path) -> pd.DataFrame:
    rows = []
    for name in expected_names():
        run = output / "runs" / name
        if not (run / "run_manifest.json").is_file():
            continue
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(run / "run_level_metrics.csv").iloc[0].to_dict()
        rows.append(
            {
                "run_name": name,
                "chain": manifest["marker_chain"],
                "n_markers": manifest["n_markers"],
                "seed": manifest["seed"],
                "runtime_seconds": manifest["runtime_seconds"],
                "all_checks_passed": manifest["all_checks_passed"],
                **metrics,
            }
        )
    index = pd.DataFrame(rows)
    index.to_csv(output / "run_index.csv", index=False)
    return index


def write_progress(
    output: Path,
    protocol: Path,
    script: Path,
    qualification: Path,
    source: Path,
    training_indices: Path,
    subsets: Path,
    failed_run: str | None = None,
) -> dict:
    index = rebuild_index(output)
    completed = len(index)
    passed = int(index["all_checks_passed"].astype(bool).sum()) if completed else 0
    progress = {
        "experiment_id": "EXP-018",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed"
        if failed_run
        else ("complete" if completed == 125 and passed == 125 else "in_progress"),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(script),
        "script_sha256": sha256(script),
        "qualification_run": str(qualification),
        "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
        "source": str(source),
        "source_sha256": sha256(source),
        "training_indices": str(training_indices),
        "training_indices_sha256": sha256(training_indices),
        "marker_subsets": str(subsets),
        "marker_subsets_sha256": sha256(subsets),
        "expected_runs": 125,
        "completed_runs": completed,
        "passed_runs": passed,
        "failed_run": failed_run,
        "all_runs_complete_and_passed": bool(
            completed == 125 and passed == 125 and failed_run is None
        ),
    }
    write_json(output / "progress_manifest.json", progress)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-runs", type=int, default=None)
    args = parser.parse_args()

    config = args.config.resolve()
    protocol = args.protocol.resolve()
    qualification = args.qualification_run.resolve()
    output = args.output.resolve()
    script = Path(__file__).resolve()
    qualification_manifest = json.loads(
        (qualification / "run_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not qualification_manifest.get("all_checks_passed")
        or qualification_manifest.get("flowsom_version") != "0.2.2"
    ):
        raise RuntimeError("EXP-013-R2 qualification is not eligible")
    if importlib.metadata.version("flowsom") != "0.2.2":
        raise RuntimeError("flowsom==0.2.2 required")

    source, _, markers, matrix, labels = load_data(config, "Levine_32dim")
    evaluable = labels != "unassigned"
    if sha256(source) != SOURCE_SHA or matrix.shape != (TOTAL_EVENTS, 32) or len(markers) != 32:
        raise RuntimeError("Levine_32dim input contract failed")
    if (
        int(evaluable.sum()) != EVALUABLE_EVENTS
        or len(np.unique(labels[evaluable])) != TRUE_POPULATIONS
    ):
        raise RuntimeError("Reference-label contract failed")

    if output.exists():
        prior = json.loads((output / "progress_manifest.json").read_text(encoding="utf-8"))
        if prior["protocol_sha256"] != sha256(protocol) or prior["source_sha256"] != sha256(source):
            raise RuntimeError("Cannot resume changed protocol or source")
    else:
        output.mkdir(parents=True)
        (output / "runs").mkdir()

    training_indices_path = output / "training_indices.npy"
    if training_indices_path.is_file():
        training_indices = np.load(training_indices_path, allow_pickle=False)
    else:
        training_indices = np.sort(
            np.random.default_rng(20260910).choice(TOTAL_EVENTS, size=20000, replace=False)
        ).astype(np.int64)
        np.save(training_indices_path, training_indices)
    subsets_path = output / "marker_permutations.csv"
    if subsets_path.is_file():
        subset_table = pd.read_csv(subsets_path)
    else:
        subset_table = build_subsets(markers)
        subset_table.to_csv(subsets_path, index=False)
    expected_subsets = build_subsets(markers)
    if not subset_table.equals(expected_subsets):
        raise RuntimeError("Marker permutation contract changed")
    if training_indices.shape != (20000,) or np.unique(training_indices).size != 20000:
        raise RuntimeError("Training index contract failed")

    write_progress(
        output, protocol, script, qualification, source, training_indices_path, subsets_path
    )
    logger.disable("flowsom")
    new_runs = 0
    marker_to_column = {marker: position for position, marker in enumerate(markers)}
    for chain in CHAINS:
        ordered_markers = (
            subset_table.loc[subset_table["chain"].eq(chain)].sort_values("rank")["marker"].tolist()
        )
        for dim in DIMS:
            selected_markers = ordered_markers[:dim]
            columns = [marker_to_column[marker] for marker in selected_markers]
            selected_matrix = matrix[:, columns]
            training_matrix = selected_matrix[training_indices]
            for seed in SEEDS:
                name = run_name(chain, dim, seed)
                run_dir = output / "runs" / name
                if (run_dir / "run_manifest.json").is_file():
                    continue
                if run_dir.exists():
                    raise RuntimeError(f"Incomplete run requires a revision: {run_dir}")
                if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                    print(
                        json.dumps(
                            write_progress(
                                output,
                                protocol,
                                script,
                                qualification,
                                source,
                                training_indices_path,
                                subsets_path,
                            ),
                            ensure_ascii=False,
                        )
                    )
                    return 0
                run_dir.mkdir()
                try:
                    started = time.perf_counter()
                    training_adata = ad.AnnData(
                        X=pd.DataFrame(training_matrix, columns=selected_markers)
                    )
                    model = fs.FlowSOM(
                        training_adata,
                        cols_to_use=selected_markers,
                        n_clusters=40,
                        xdim=10,
                        ydim=10,
                        rlen=30,
                        seed=seed,
                    )
                    codes = np.asarray(model.model.codes, dtype=np.float64)
                    node_to_meta = np.asarray(model.model._y_codes, dtype=np.int16)
                    nodes_float, distances = map_data_to_codes(selected_matrix, codes)
                    nodes = nodes_float.astype(np.int16)
                    metas = node_to_meta[nodes].astype(np.int16)
                    distances = np.asarray(distances, dtype=np.float32)
                    runtime = time.perf_counter() - started

                    np.save(run_dir / "node_labels_all_events.npy", nodes)
                    np.save(run_dir / "metacluster_labels_all_events.npy", metas)
                    np.save(run_dir / "bmu_distances_all_events.npy", distances)
                    np.save(run_dir / "som_codes.npy", codes)
                    np.save(run_dir / "node_to_metacluster.npy", node_to_meta)
                    (run_dir / "selected_markers.txt").write_text(
                        "\n".join(selected_markers) + "\n", encoding="utf-8"
                    )

                    metrics, population, clusters, mapping, contingency = evaluate_multiclass(
                        labels[evaluable], metas[evaluable], metas
                    )
                    valid_metrics, metric_detail = metrics_are_valid(metrics)
                    run_metrics = {
                        "dataset": "Levine_32dim",
                        "algorithm": "official_Python_FlowSOM_0.2.2",
                        "marker_chain": chain,
                        "n_markers": dim,
                        "seed": seed,
                        "grid_side": 10,
                        "rlen": 30,
                        "n_metaclusters_requested": 40,
                        "mean_bmu_distance": float(np.mean(distances)),
                        "median_bmu_distance": float(np.median(distances)),
                        "median_bmu_distance_per_sqrt_marker": float(
                            np.median(distances) / np.sqrt(dim)
                        ),
                        "q90_bmu_distance": float(np.quantile(distances, 0.9)),
                        "occupied_nodes_all_events": int(np.unique(nodes).size),
                        "occupied_metaclusters_all_events": int(np.unique(metas).size),
                        **metrics,
                    }
                    checks: list[dict] = []

                    def check(check_name: str, passed: bool, detail: str) -> None:
                        checks.append(
                            {"check": check_name, "passed": bool(passed), "detail": detail}
                        )

                    prior_set = (
                        set(ordered_markers[: DIMS[DIMS.index(dim) - 1]])
                        if dim != DIMS[0]
                        else set()
                    )
                    check(
                        "qualification_link",
                        qualification_manifest["all_checks_passed"],
                        str(qualification / "run_manifest.json"),
                    )
                    check("source_hash", sha256(source) == SOURCE_SHA, sha256(source))
                    check(
                        "training_indices",
                        len(training_indices) == 20000
                        and np.unique(training_indices).size == 20000,
                        sha256(training_indices_path),
                    )
                    check(
                        "marker_subset",
                        len(selected_markers) == dim
                        and len(set(selected_markers)) == dim
                        and prior_set.issubset(set(selected_markers)),
                        json.dumps(selected_markers),
                    )
                    check(
                        "event_shapes",
                        nodes.shape == metas.shape == distances.shape == (TOTAL_EVENTS,),
                        str(nodes.shape),
                    )
                    check(
                        "node_contract",
                        codes.shape == (100, dim)
                        and node_to_meta.shape == (100,)
                        and int(nodes.min()) >= 0
                        and int(nodes.max()) < 100,
                        f"codes={codes.shape}; nodes={int(nodes.min())}-{int(nodes.max())}",
                    )
                    check(
                        "metacluster_contract",
                        len(np.unique(node_to_meta)) == 40
                        and set(np.unique(metas)).issubset(set(np.unique(node_to_meta))),
                        f"node_meta={len(np.unique(node_to_meta))}; occupied={len(np.unique(metas))}",
                    )
                    check(
                        "finite_outputs",
                        np.isfinite(codes).all()
                        and np.isfinite(distances).all()
                        and bool(np.all(distances >= 0)),
                        f"range={float(distances.min())}-{float(distances.max())}",
                    )
                    check(
                        "evaluation_contract",
                        metrics["n_evaluable_events"] == EVALUABLE_EVENTS
                        and metrics["n_true_populations"] == TRUE_POPULATIONS,
                        f"events={metrics['n_evaluable_events']}; populations={metrics['n_true_populations']}",
                    )
                    check("metric_ranges", valid_metrics, metric_detail)
                    checks_frame = pd.DataFrame(checks)
                    checks_frame.to_csv(run_dir / "qualification_checks.csv", index=False)
                    pd.DataFrame([run_metrics]).to_csv(
                        run_dir / "run_level_metrics.csv", index=False
                    )
                    population.to_csv(run_dir / "population_level_metrics.csv", index=False)
                    clusters.to_csv(run_dir / "metacluster_level_diagnostics.csv", index=False)
                    mapping.to_csv(run_dir / "hungarian_mapping.csv", index=False)
                    contingency.to_csv(
                        run_dir / "contingency_true_by_predicted.csv", index_label="true_population"
                    )
                    manifest = {
                        "experiment_id": "EXP-018",
                        "run_name": name,
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset": "Levine_32dim",
                        "source": str(source),
                        "source_sha256": sha256(source),
                        "protocol": str(protocol),
                        "protocol_sha256": sha256(protocol),
                        "script": str(script),
                        "script_sha256": sha256(script),
                        "qualification_run": str(qualification),
                        "qualification_manifest_sha256": sha256(
                            qualification / "run_manifest.json"
                        ),
                        "training_indices_sha256": sha256(training_indices_path),
                        "marker_permutations_sha256": sha256(subsets_path),
                        "marker_chain": chain,
                        "permutation_seed": 270000 + chain,
                        "n_markers": dim,
                        "selected_markers": selected_markers,
                        "seed": seed,
                        "grid_side": 10,
                        "rlen": 30,
                        "n_metaclusters_requested": 40,
                        "fit_policy": "fixed_unlabeled_20000_training_sample_then_map_all_events",
                        "labels_used_for_fit_sampling_or_parameter_selection": False,
                        "runtime_seconds": runtime,
                        "flowsom_version": importlib.metadata.version("flowsom"),
                        "python": sys.version,
                        "python_executable": sys.executable,
                        "platform": platform.platform(),
                        "checks_passed": int(checks_frame["passed"].sum()),
                        "checks_total": len(checks_frame),
                        "all_checks_passed": bool(checks_frame["passed"].all()),
                    }
                    write_json(run_dir / "run_manifest.json", manifest)
                    (run_dir / "audit.md").write_text(
                        f"# EXP-018 {name}\n\nNested-marker experiment within Levine_32dim; chain={chain}, dim={dim}, seed={seed}; 10×10 grid, rlen=30, 40 metaclusters; labels were used only for post hoc evaluation.\n"
                        f"ARI={metrics['ari']:.6f}, macro F1={metrics['macro_f1']:.6f}, median BMU/sqrt(d)={run_metrics['median_bmu_distance_per_sqrt_marker']:.6f}; checks passed: {manifest['checks_passed']}/{manifest['checks_total']}.\n",
                        encoding="utf-8",
                    )
                    artifacts = [path for path in run_dir.iterdir() if path.is_file()]
                    write_json(
                        run_dir / "artifact_hashes.json",
                        {str(path): sha256(path) for path in artifacts},
                    )
                    if not manifest["all_checks_passed"]:
                        raise RuntimeError(f"Integrity checks failed: {name}")
                    new_runs += 1
                    print(
                        json.dumps(
                            {
                                "run": name,
                                "runtime_seconds": runtime,
                                "ari": metrics["ari"],
                                "macro_f1": metrics["macro_f1"],
                                "median_bmu_per_sqrt_marker": run_metrics[
                                    "median_bmu_distance_per_sqrt_marker"
                                ],
                                "checks": f"{manifest['checks_passed']}/{manifest['checks_total']}",
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    (run_dir / "failure_traceback.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    write_json(
                        run_dir / "failure.json",
                        {
                            "run_name": name,
                            "exception": repr(error),
                            "scientific_output_eligible": False,
                        },
                    )
                    print(
                        json.dumps(
                            write_progress(
                                output,
                                protocol,
                                script,
                                qualification,
                                source,
                                training_indices_path,
                                subsets_path,
                                failed_run=name,
                            ),
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )
                    return 1
                write_progress(
                    output,
                    protocol,
                    script,
                    qualification,
                    source,
                    training_indices_path,
                    subsets_path,
                )

    progress = write_progress(
        output, protocol, script, qualification, source, training_indices_path, subsets_path
    )
    print(json.dumps(progress, ensure_ascii=False))
    return 0 if progress["all_runs_complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
