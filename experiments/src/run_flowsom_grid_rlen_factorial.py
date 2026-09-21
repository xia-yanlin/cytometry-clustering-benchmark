from __future__ import annotations

import argparse
import hashlib
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

GRID_RLEN = [
    (10, 10),
    (15, 10),
    (20, 10),
    (10, 20),
    (15, 20),
    (20, 20),
    (10, 30),
    (15, 30),
    (20, 30),
]

DATASET_CONTRACTS = {
    "Levine_32dim": {
        "total_events": 265627,
        "evaluable_events": 104184,
        "true_populations": 14,
        "n_markers": 32,
        "source_sha256": "bb1f9cb63377f701795f116e557f8ede05141b6aca2a0270177c7b1eb4c807e4",
    },
    "Samusik_01": {
        "total_events": 86864,
        "evaluable_events": 53173,
        "true_populations": 24,
        "n_markers": 39,
        "source_sha256": "86dd329dedec30b86ab0cc845923620393a36f16bd07d8f5d87ba4dbb3511565",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data(
    config_path: Path, dataset: str
) -> tuple[Path, pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["datasets"][dataset]
    source = (Path(config["data_root"]) / dataset / spec["filename"]).resolve()
    separator = {"tab": "\t", "comma": ","}[spec["separator"]]
    frame = pd.read_csv(source, sep=separator)
    markers = [column for column in frame.columns if column not in spec["exclude_columns"]]
    matrix = frame[markers].to_numpy(dtype=np.float64)
    if spec["cofactor"] is not None:
        matrix = np.arcsinh(matrix / float(spec["cofactor"]))
    labels = frame["label"].astype(str).to_numpy()
    return source, frame, markers, matrix, labels


def expected_run_names(seeds: list[int]) -> list[str]:
    return [
        f"grid{side:02d}_rlen{rlen:02d}_seed{seed:03d}"
        for side, rlen in GRID_RLEN
        for seed in seeds
    ]


def rebuild_index(output: Path, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        manifest_path = output / "runs" / name / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = pd.read_csv(output / "runs" / name / "run_level_metrics.csv").iloc[0].to_dict()
        rows.append(
            {
                "run_name": name,
                "grid_side": manifest["grid_side"],
                "n_nodes": manifest["n_nodes"],
                "rlen": manifest["rlen"],
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
    *,
    experiment_id: str,
    protocol: Path,
    script: Path,
    qualification: Path,
    source: Path,
    markers: list[str],
    training_indices: Path,
    seeds: list[int],
    names: list[str],
    failed_run: str | None = None,
) -> dict:
    index = rebuild_index(output, names)
    completed = len(index)
    passed = int(index["all_checks_passed"].astype(bool).sum()) if completed else 0
    progress = {
        "experiment_id": experiment_id,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed"
        if failed_run
        else ("complete" if completed == len(names) and passed == len(names) else "in_progress"),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(script),
        "script_sha256": sha256(script),
        "qualification_run": str(qualification),
        "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
        "source": str(source),
        "source_sha256": sha256(source),
        "markers": markers,
        "n_markers": len(markers),
        "training_indices": str(training_indices),
        "training_indices_sha256": sha256(training_indices),
        "expected_runs": len(names),
        "completed_runs": completed,
        "passed_runs": passed,
        "failed_run": failed_run,
        "all_runs_complete_and_passed": completed == len(names)
        and passed == len(names)
        and failed_run is None,
    }
    write_json(output / "progress_manifest.json", progress)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", default="Levine_32dim")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification-run", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-size", type=int, default=20000)
    parser.add_argument("--training-sample-seed", type=int, default=20260910)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-new-runs", type=int, default=None)
    args = parser.parse_args()

    if args.dataset not in DATASET_CONTRACTS:
        raise ValueError(f"Unsupported factorial dataset: {args.dataset}")
    contract = DATASET_CONTRACTS[args.dataset]
    seeds = [int(value) for value in args.seeds.split(",")]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("EXP-014A seeds are locked to 0,1,2,3,4")
    if args.training_size != 20000 or args.training_sample_seed != 20260910:
        raise ValueError("EXP-014A training sample contract changed")

    config_path = args.config.resolve()
    protocol = args.protocol.resolve()
    qualification = args.qualification_run.resolve()
    output = args.output.resolve()
    script = Path(__file__).resolve()
    qualification_manifest_path = qualification / "run_manifest.json"
    if (
        not config_path.is_file()
        or not protocol.is_file()
        or not qualification_manifest_path.is_file()
    ):
        raise FileNotFoundError("Config, protocol, or qualification manifest is missing")
    expected_protocol_token = "levine32" if args.dataset == "Levine_32dim" else "samusik"
    if expected_protocol_token not in protocol.name.lower():
        raise ValueError(f"Protocol does not match dataset {args.dataset}: {protocol.name}")
    qualification_manifest = json.loads(qualification_manifest_path.read_text(encoding="utf-8"))
    if (
        not qualification_manifest.get("all_checks_passed")
        or qualification_manifest.get("flowsom_version") != "0.2.2"
    ):
        raise RuntimeError("FlowSOM qualification run is not eligible")
    if importlib.metadata.version("flowsom") != "0.2.2":
        raise RuntimeError("This runner requires flowsom==0.2.2")

    source, frame, markers, matrix, labels = load_data(config_path, args.dataset)
    if (
        matrix.shape != (contract["total_events"], contract["n_markers"])
        or len(markers) != contract["n_markers"]
    ):
        raise ValueError(f"Unexpected data contract: {matrix.shape}; markers={len(markers)}")
    evaluable = labels != "unassigned"
    if (
        int(evaluable.sum()) != contract["evaluable_events"]
        or len(np.unique(labels[evaluable])) != contract["true_populations"]
    ):
        raise ValueError("Unexpected reference-label contract")

    if output.exists():
        existing = json.loads((output / "progress_manifest.json").read_text(encoding="utf-8"))
        if (
            existing["experiment_id"] != args.experiment_id
            or existing["protocol_sha256"] != sha256(protocol)
            or existing["source_sha256"] != sha256(source)
        ):
            raise RuntimeError("Cannot resume: protocol or source hash differs")
    else:
        output.mkdir(parents=True)
        (output / "runs").mkdir()
        (output / "marker_columns.txt").write_text("\n".join(markers) + "\n", encoding="utf-8")

    training_indices_path = output / "training_indices.npy"
    if training_indices_path.exists():
        training_indices = np.load(training_indices_path, allow_pickle=False)
    else:
        rng = np.random.default_rng(args.training_sample_seed)
        training_indices = np.sort(
            rng.choice(len(matrix), size=args.training_size, replace=False)
        ).astype(np.int64)
        np.save(training_indices_path, training_indices)
    if (
        training_indices.shape != (args.training_size,)
        or np.unique(training_indices).size != args.training_size
        or int(training_indices.min()) < 0
        or int(training_indices.max()) >= len(matrix)
    ):
        raise ValueError("Training indices failed integrity checks")
    training_matrix = matrix[training_indices]

    names = expected_run_names(seeds)
    write_progress(
        output,
        experiment_id=args.experiment_id,
        protocol=protocol,
        script=script,
        qualification=qualification,
        source=source,
        markers=markers,
        training_indices=training_indices_path,
        seeds=seeds,
        names=names,
    )

    logger.disable("flowsom")
    new_runs = 0
    for side, rlen in GRID_RLEN:
        for seed in seeds:
            run_name = f"grid{side:02d}_rlen{rlen:02d}_seed{seed:03d}"
            run_dir = output / "runs" / run_name
            if (run_dir / "run_manifest.json").is_file():
                continue
            if run_dir.exists():
                raise RuntimeError(
                    f"Preserved incomplete run requires a new experiment revision: {run_dir}"
                )
            if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                progress = write_progress(
                    output,
                    experiment_id=args.experiment_id,
                    protocol=protocol,
                    script=script,
                    qualification=qualification,
                    source=source,
                    markers=markers,
                    training_indices=training_indices_path,
                    seeds=seeds,
                    names=names,
                )
                print(json.dumps(progress, ensure_ascii=False))
                return 0

            run_dir.mkdir()
            try:
                started = time.perf_counter()
                training_adata = ad.AnnData(X=pd.DataFrame(training_matrix, columns=markers))
                model = fs.FlowSOM(
                    training_adata,
                    cols_to_use=markers,
                    n_clusters=40,
                    xdim=side,
                    ydim=side,
                    rlen=rlen,
                    seed=seed,
                )
                codes = np.asarray(model.model.codes, dtype=np.float64)
                node_to_meta = np.asarray(model.model._y_codes, dtype=np.int16)
                node_labels_float, bmu_distances = map_data_to_codes(matrix, codes)
                node_labels = node_labels_float.astype(np.int16)
                metacluster_labels = node_to_meta[node_labels].astype(np.int16)
                bmu_distances = np.asarray(bmu_distances, dtype=np.float32)
                runtime = time.perf_counter() - started

                np.save(run_dir / "node_labels_all_events.npy", node_labels)
                np.save(run_dir / "metacluster_labels_all_events.npy", metacluster_labels)
                np.save(run_dir / "bmu_distances_all_events.npy", bmu_distances)
                np.save(run_dir / "som_codes.npy", codes)
                np.save(run_dir / "node_to_metacluster.npy", node_to_meta)

                metrics, population_frame, cluster_frame, mapping_frame, contingency_frame = (
                    evaluate_multiclass(
                        labels[evaluable], metacluster_labels[evaluable], metacluster_labels
                    )
                )
                metric_valid, metric_detail = metrics_are_valid(metrics)
                unsupervised = {
                    "mean_bmu_distance": float(np.mean(bmu_distances)),
                    "median_bmu_distance": float(np.median(bmu_distances)),
                    "q90_bmu_distance": float(np.quantile(bmu_distances, 0.9)),
                    "occupied_nodes_all_events": int(np.unique(node_labels).size),
                    "occupied_metaclusters_all_events": int(np.unique(metacluster_labels).size),
                }
                checks: list[dict] = []

                def check(name: str, passed: bool, detail: str) -> None:
                    checks.append({"check": name, "passed": bool(passed), "detail": detail})

                n_nodes = side * side
                check(
                    "qualification_link",
                    qualification_manifest["all_checks_passed"],
                    qualification_manifest_path.as_posix(),
                )
                check("source_hash", sha256(source) == contract["source_sha256"], sha256(source))
                check(
                    "training_indices",
                    len(training_indices) == 20000 and np.unique(training_indices).size == 20000,
                    sha256(training_indices_path),
                )
                check(
                    "event_shapes",
                    node_labels.shape
                    == metacluster_labels.shape
                    == bmu_distances.shape
                    == (contract["total_events"],),
                    str((node_labels.shape, metacluster_labels.shape, bmu_distances.shape)),
                )
                check(
                    "node_contract",
                    codes.shape == (n_nodes, contract["n_markers"])
                    and node_to_meta.shape == (n_nodes,)
                    and int(node_labels.min()) >= 0
                    and int(node_labels.max()) < n_nodes,
                    f"codes={codes.shape}; node_range={int(node_labels.min())}-{int(node_labels.max())}",
                )
                check(
                    "metacluster_contract",
                    len(np.unique(node_to_meta)) == 40
                    and set(np.unique(metacluster_labels)).issubset(set(np.unique(node_to_meta))),
                    f"node_meta={np.unique(node_to_meta).size}; occupied={np.unique(metacluster_labels).size}",
                )
                check(
                    "finite_outputs",
                    np.isfinite(codes).all()
                    and np.isfinite(bmu_distances).all()
                    and bool(np.all(bmu_distances >= 0)),
                    f"distance_range={float(bmu_distances.min())}-{float(bmu_distances.max())}",
                )
                check(
                    "evaluation_contract",
                    metrics["n_evaluable_events"] == contract["evaluable_events"]
                    and metrics["n_true_populations"] == contract["true_populations"],
                    f"evaluable={metrics['n_evaluable_events']}; populations={metrics['n_true_populations']}",
                )
                check("metric_ranges", metric_valid, metric_detail)
                checks_frame = pd.DataFrame(checks)
                checks_frame.to_csv(run_dir / "qualification_checks.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "dataset": args.dataset,
                            "algorithm": "official_Python_FlowSOM_0.2.2",
                            "grid_side": side,
                            "rlen": rlen,
                            "seed": seed,
                            "n_metaclusters_requested": 40,
                            **unsupervised,
                            **metrics,
                        }
                    ]
                ).to_csv(run_dir / "run_level_metrics.csv", index=False)
                population_frame.to_csv(run_dir / "population_level_metrics.csv", index=False)
                cluster_frame.to_csv(run_dir / "metacluster_level_diagnostics.csv", index=False)
                mapping_frame.to_csv(run_dir / "hungarian_mapping.csv", index=False)
                contingency_frame.to_csv(
                    run_dir / "contingency_true_by_predicted.csv", index_label="true_population"
                )

                manifest = {
                    "experiment_id": args.experiment_id,
                    "run_name": run_name,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": args.dataset,
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "config": str(config_path),
                    "config_sha256": sha256(config_path),
                    "protocol": str(protocol),
                    "protocol_sha256": sha256(protocol),
                    "script": str(script),
                    "script_sha256": sha256(script),
                    "metric_module": str(
                        (script.parent / "evaluate_xshift_cross_dataset.py").resolve()
                    ),
                    "metric_module_sha256": sha256(
                        script.parent / "evaluate_xshift_cross_dataset.py"
                    ),
                    "qualification_run": str(qualification),
                    "qualification_manifest_sha256": sha256(qualification_manifest_path),
                    "training_size": args.training_size,
                    "training_sample_seed": args.training_sample_seed,
                    "training_indices_sha256": sha256(training_indices_path),
                    "fit_policy": "fixed_unlabeled_random_training_sample_then_map_all_events",
                    "label_used_for_training_sampling_fit_or_parameter_selection": False,
                    "labels_used_only_for_posthoc_evaluation": True,
                    "grid_side": side,
                    "n_nodes": n_nodes,
                    "rlen": rlen,
                    "seed": seed,
                    "n_metaclusters_requested": 40,
                    "runtime_seconds": runtime,
                    "flowsom_version": importlib.metadata.version("flowsom"),
                    "python": sys.version,
                    "python_executable": sys.executable,
                    "platform": platform.platform(),
                    "checks_passed": int(checks_frame["passed"].sum()),
                    "checks_total": int(len(checks_frame)),
                    "all_checks_passed": bool(checks_frame["passed"].all()),
                }
                write_json(run_dir / "run_manifest.json", manifest)
                (run_dir / "audit.md").write_text(
                    "\n".join(
                        [
                            f"# {args.experiment_id} {run_name}",
                            "",
                            f"FlowSOM 0.2.2; {args.dataset}; a fixed unlabeled 20,000-event training sample mapped to all {contract['total_events']:,} events; {side}×{side} grid; rlen={rlen}; seed={seed}; 40 fixed metaclusters.",
                            f"median BMU distance={unsupervised['median_bmu_distance']:.6f}；ARI={metrics['ari']:.6f}；Macro P/R/F1={metrics['macro_precision']:.6f}/{metrics['macro_recall']:.6f}/{metrics['macro_f1']:.6f}。",
                            f"Integrity checks passed: {int(checks_frame['passed'].sum())}/{len(checks_frame)}. Labels were used only for evaluation after mapping.",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                artifacts = [path for path in run_dir.iterdir() if path.is_file()]
                write_json(
                    run_dir / "artifact_hashes.json",
                    {str(path): sha256(path) for path in artifacts},
                )
                if not manifest["all_checks_passed"]:
                    raise RuntimeError(f"Run integrity checks failed: {run_name}")
                new_runs += 1
                print(
                    json.dumps(
                        {
                            "run": run_name,
                            "runtime_seconds": runtime,
                            **unsupervised,
                            "ari": metrics["ari"],
                            "macro_f1": metrics["macro_f1"],
                            "checks": f"{manifest['checks_passed']}/{manifest['checks_total']}",
                        },
                        ensure_ascii=False,
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
                        "run_name": run_name,
                        "failed_utc": datetime.now(timezone.utc).isoformat(),
                        "exception": repr(error),
                        "scientific_output_eligible": False,
                    },
                )
                progress = write_progress(
                    output,
                    experiment_id=args.experiment_id,
                    protocol=protocol,
                    script=script,
                    qualification=qualification,
                    source=source,
                    markers=markers,
                    training_indices=training_indices_path,
                    seeds=seeds,
                    names=names,
                    failed_run=run_name,
                )
                print(json.dumps(progress, ensure_ascii=False), file=sys.stderr)
                return 1

            write_progress(
                output,
                experiment_id=args.experiment_id,
                protocol=protocol,
                script=script,
                qualification=qualification,
                source=source,
                markers=markers,
                training_indices=training_indices_path,
                seeds=seeds,
                names=names,
            )

    progress = write_progress(
        output,
        experiment_id=args.experiment_id,
        protocol=protocol,
        script=script,
        qualification=qualification,
        source=source,
        markers=markers,
        training_indices=training_indices_path,
        seeds=seeds,
        names=names,
    )
    print(json.dumps(progress, ensure_ascii=False))
    return 0 if progress["all_runs_complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
