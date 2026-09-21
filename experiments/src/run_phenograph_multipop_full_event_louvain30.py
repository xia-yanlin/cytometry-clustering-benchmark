from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_xshift_cross_dataset import evaluate_multiclass, metrics_are_valid
from phenograph.cluster import run_louvain, sort_by_size
from phenograph.core import find_neighbors, jaccard_kernel, neighbor_graph
from scipy import sparse as sp
from sklearn.metrics import adjusted_rand_score

REPEATS = tuple(range(30))
K = 30
MIN_CLUSTER_SIZE = 10
CONTRACTS = {
    "Levine_13dim": {
        "rows": 167044,
        "markers": 13,
        "evaluable": 81747,
        "populations": 24,
        "sha": "941c6328909f0abf3bd18801fd9ca95779356b225ba422c1f2daf0d2a737f354",
        "label_policy": "finite_numeric_1_to_24",
    },
    "Levine_32dim": {
        "rows": 265627,
        "markers": 32,
        "evaluable": 104184,
        "populations": 14,
        "sha": "bb1f9cb63377f701795f116e557f8ede05141b6aca2a0270177c7b1eb4c807e4",
        "label_policy": "not_unassigned",
    },
    "Samusik_01": {
        "rows": 86864,
        "markers": 39,
        "evaluable": 53173,
        "populations": 24,
        "sha": "86dd329dedec30b86ab0cc845923620393a36f16bd07d8f5d87ba4dbb3511565",
        "label_policy": "not_unassigned",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data(config_path: Path, dataset: str):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["datasets"][dataset]
    source = (Path(config["data_root"]) / dataset / spec["filename"]).resolve()
    frame = pd.read_csv(
        source, sep={"tab": "\t", "comma": ","}[spec["separator"]], low_memory=False
    )
    markers = [column for column in frame.columns if column not in set(spec["exclude_columns"])]
    matrix = frame[markers].to_numpy(dtype=np.float64)
    if spec["cofactor"] is not None:
        matrix = np.arcsinh(matrix / float(spec["cofactor"]))
    labels = frame["label"]
    if CONTRACTS[dataset]["label_policy"] == "finite_numeric_1_to_24":
        numeric = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=np.float64)
        evaluable = np.isfinite(numeric) & (numeric >= 1) & (numeric <= 24)
        labels_text = np.asarray(
            [
                str(int(value)) if keep else "__not_evaluable__"
                for value, keep in zip(numeric, evaluable, strict=False)
            ],
            dtype=object,
        )
    else:
        labels_text = labels.astype(str).to_numpy()
        evaluable = labels_text != "unassigned"
    return source, markers, matrix, labels_text, evaluable


def rebuild_index(output: Path) -> pd.DataFrame:
    rows = []
    for repeat in REPEATS:
        run = output / "runs" / f"repeat{repeat:03d}"
        if (run / "run_manifest.json").is_file():
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            metrics = pd.read_csv(run / "run_level_metrics.csv").iloc[0].to_dict()
            rows.append(
                {
                    "run_name": run.name,
                    "repeat": repeat,
                    "runtime_seconds": manifest["runtime_seconds"],
                    "all_checks_passed": manifest["all_checks_passed"],
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "run_index.csv", index=False)
    return frame


def progress(
    output: Path,
    experiment_id: str,
    protocol: Path,
    qualification: Path,
    source: Path,
    graph: Path,
    neighbors: Path,
    failed: str | None = None,
) -> dict:
    index = rebuild_index(output)
    passed = int(index["all_checks_passed"].astype(bool).sum()) if len(index) else 0
    value = {
        "experiment_id": experiment_id,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed"
        if failed
        else ("complete" if len(index) == passed == 30 else "in_progress"),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "qualification": str(qualification),
        "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
        "source": str(source),
        "source_sha256": sha256(source),
        "fixed_graph": str(graph),
        "fixed_graph_sha256": sha256(graph),
        "neighbors": str(neighbors),
        "neighbors_sha256": sha256(neighbors),
        "k": K,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "q_tol": 1e-3,
        "louvain_time_limit": 2000,
        "evaluation_scope": "all_events_fit_evaluable_reference_labels_only",
        "repeat_indices": list(REPEATS),
        "seed_control": "unavailable_in_official_default_louvain_interface",
        "expected_runs": 30,
        "completed_runs": len(index),
        "passed_runs": passed,
        "failed_run": failed,
        "all_runs_complete_and_passed": len(index) == passed == 30 and failed is None,
    }
    write_json(output / "progress_manifest.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-runs", type=int, default=None)
    args = parser.parse_args()
    config, protocol, qualification, output = (
        path.resolve() for path in [args.config, args.protocol, args.qualification, args.output]
    )
    contract = CONTRACTS[args.dataset]
    qualification_manifest = json.loads(
        (qualification / "run_manifest.json").read_text(encoding="utf-8")
    )
    if (
        qualification_manifest.get("experiment_id") != "EXP-016-R2"
        or qualification_manifest.get("all_checks_passed") is not True
    ):
        raise RuntimeError("EXP-016-R2 official default Louvain qualification is required")
    source, markers, matrix, labels, evaluable = load_data(config, args.dataset)
    if sha256(source) != contract["sha"] or matrix.shape != (contract["rows"], contract["markers"]):
        raise ValueError("dataset source or matrix contract failed")
    if (
        int(evaluable.sum()) != contract["evaluable"]
        or int(np.unique(labels[evaluable]).size) != contract["populations"]
    ):
        raise ValueError("evaluation-label contract failed")
    if output.exists():
        old = (
            json.loads((output / "progress_manifest.json").read_text(encoding="utf-8"))
            if (output / "progress_manifest.json").is_file()
            else None
        )
        if old is not None and (
            old.get("protocol_sha256") != sha256(protocol)
            or old.get("experiment_id") != args.experiment_id
        ):
            raise RuntimeError("resume contract differs")
    else:
        (output / "runs").mkdir(parents=True)
        (output / "marker_columns.txt").write_text("\n".join(markers) + "\n", encoding="utf-8")
    graph_path = output / "fixed_full_event_graph.npz"
    neighbors_path = output / "neighbor_indices_k30.npy"
    if not graph_path.is_file():
        started = time.perf_counter()
        _, neighbor_indices = find_neighbors(
            matrix, k=K, metric="euclidean", method="kdtree", n_jobs=1
        )
        np.save(neighbors_path, neighbor_indices.astype(np.int32))
        raw_graph = neighbor_graph(jaccard_kernel, {"idx": neighbor_indices})
        graph = sp.tril((raw_graph + raw_graph.transpose()).multiply(0.5), -1).tocoo()
        sp.save_npz(graph_path, graph)
        write_json(
            output / "graph_manifest.json",
            {
                "build_seconds": time.perf_counter() - started,
                "shape": list(graph.shape),
                "nnz": int(graph.nnz),
                "weight_min": float(graph.data.min()),
                "weight_max": float(graph.data.max()),
                "strict_lower_triangle": bool(np.all(graph.row > graph.col)),
                "finite_positive_weights": bool(
                    np.isfinite(graph.data).all() and np.all(graph.data > 0)
                ),
                "labels_used": False,
            },
        )
    graph = sp.load_npz(graph_path).tocoo()
    neighbor_indices = np.load(neighbors_path, allow_pickle=False)
    if graph.shape != (contract["rows"], contract["rows"]) or neighbor_indices.shape != (
        contract["rows"],
        K,
    ):
        raise ValueError("graph or neighbor shape contract failed")
    if (
        not np.all(graph.row > graph.col)
        or not np.isfinite(graph.data).all()
        or not np.all(graph.data > 0)
    ):
        raise ValueError("graph structural contract failed")
    progress(
        output, args.experiment_id, protocol, qualification, source, graph_path, neighbors_path
    )
    new_runs = 0
    for repeat in REPEATS:
        run = output / "runs" / f"repeat{repeat:03d}"
        if (run / "run_manifest.json").is_file():
            continue
        if args.max_new_runs is not None and new_runs >= args.max_new_runs:
            print(
                json.dumps(
                    progress(
                        output,
                        args.experiment_id,
                        protocol,
                        qualification,
                        source,
                        graph_path,
                        neighbors_path,
                    ),
                    ensure_ascii=False,
                )
            )
            return 0
        if run.exists():
            raise RuntimeError(f"incomplete run preserved: {run}")
        run.mkdir()
        try:
            started = time.perf_counter()
            raw_membership, quality = run_louvain(graph, 1e-3, 2000)
            communities = np.asarray(
                sort_by_size(np.asarray(raw_membership), MIN_CLUSTER_SIZE), dtype=np.int32
            )
            runtime = time.perf_counter() - started
            np.save(
                run / "raw_membership_all_events.npy", np.asarray(raw_membership, dtype=np.int32)
            )
            np.save(run / "community_labels_all_events.npy", communities)
            metrics, populations, clusters, mapping, contingency = evaluate_multiclass(
                labels[evaluable], communities[evaluable], communities
            )
            metric_ok, metric_detail = metrics_are_valid(metrics)
            valid_communities = np.unique(communities[communities >= 0])
            checks = []

            def check(name: str, passed: bool, detail: str) -> None:
                checks.append({"check": name, "passed": bool(passed), "detail": detail})

            check(
                "qualification", qualification_manifest["all_checks_passed"] is True, "EXP-016-R2"
            )
            check("source", sha256(source) == contract["sha"], sha256(source))
            check(
                "graph",
                graph.shape == (contract["rows"], contract["rows"])
                and np.all(graph.row > graph.col),
                f"shape={graph.shape}; nnz={graph.nnz}",
            )
            check(
                "neighbors",
                neighbor_indices.shape == (contract["rows"], K),
                str(neighbor_indices.shape),
            )
            check(
                "membership",
                communities.shape == (contract["rows"],)
                and communities.min() >= -1
                and len(valid_communities) > 1,
                f"shape={communities.shape}; communities={len(valid_communities)}",
            )
            check(
                "relabel_partition",
                adjusted_rand_score(raw_membership, communities) == 1.0,
                "ARI=1",
            )
            check("quality", np.isfinite(quality), str(quality))
            check(
                "evaluation",
                metrics["n_total_events"] == contract["rows"]
                and metrics["n_evaluable_events"] == contract["evaluable"]
                and metrics["n_true_populations"] == contract["populations"],
                f"total={metrics['n_total_events']}; evaluable={metrics['n_evaluable_events']}; populations={metrics['n_true_populations']}",
            )
            check("metrics", metric_ok, metric_detail)
            check(
                "labels_permission",
                True,
                "labels used only to define posthoc evaluable rows and metrics",
            )
            check_frame = pd.DataFrame(checks)
            check_frame.to_csv(run / "checks.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": args.dataset,
                        "algorithm": "PhenoGraph_v1.5.7_default_Louvain",
                        "repeat": repeat,
                        "seed_control": "unavailable",
                        "k_neighbors": K,
                        "evaluation_scope": "all_events_fit_evaluable_reference_labels_only",
                        "n_valid_communities": int(len(valid_communities)),
                        "outlier_events": int(np.sum(communities < 0)),
                        "outlier_fraction": float(np.mean(communities < 0)),
                        "quality_q": float(quality),
                        **metrics,
                    }
                ]
            ).to_csv(run / "run_level_metrics.csv", index=False)
            populations.to_csv(run / "population_level_metrics.csv", index=False)
            clusters.to_csv(run / "community_level_diagnostics.csv", index=False)
            mapping.to_csv(run / "hungarian_mapping.csv", index=False)
            contingency.to_csv(
                run / "contingency_true_by_predicted.csv", index_label="true_population"
            )
            manifest = {
                "experiment_id": args.experiment_id,
                "dataset": args.dataset,
                "repeat": repeat,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "source_sha256": sha256(source),
                "graph_sha256": sha256(graph_path),
                "neighbors_sha256": sha256(neighbors_path),
                "protocol": str(protocol),
                "protocol_sha256": sha256(protocol),
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256(Path(__file__).resolve()),
                "qualification": str(qualification),
                "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
                "fit_policy": "all_events_full_graph",
                "labels_used_for_graph_or_clustering": False,
                "k_neighbors": K,
                "q_tol": 1e-3,
                "louvain_time_limit": 2000,
                "min_cluster_size": MIN_CLUSTER_SIZE,
                "seed_control": "unavailable_in_official_default_louvain_interface",
                "runtime_seconds": runtime,
                "quality_q": float(quality),
                "packages": {
                    name: version(name) for name in ("PhenoGraph", "numpy", "scipy", "scikit-learn")
                },
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "checks_passed": int(check_frame["passed"].sum()),
                "checks_total": len(check_frame),
                "all_checks_passed": bool(check_frame["passed"].all()),
            }
            write_json(run / "run_manifest.json", manifest)
            write_json(
                run / "artifact_hashes.json",
                {path.name: sha256(path) for path in run.iterdir() if path.is_file()},
            )
            if not manifest["all_checks_passed"]:
                raise RuntimeError("run checks failed")
            new_runs += 1
            print(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "repeat": repeat,
                        "runtime_seconds": runtime,
                        "quality_q": quality,
                        "communities": len(valid_communities),
                        "ari": metrics["ari"],
                        "macro_f1": metrics["macro_f1"],
                        "checks": "10/10",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as error:
            (run / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(
                run / "failure.json",
                {"exception": repr(error), "scientific_output_eligible": False},
            )
            progress(
                output,
                args.experiment_id,
                protocol,
                qualification,
                source,
                graph_path,
                neighbors_path,
                str(run.relative_to(output)),
            )
            raise
        progress(
            output, args.experiment_id, protocol, qualification, source, graph_path, neighbors_path
        )
    final = progress(
        output, args.experiment_id, protocol, qualification, source, graph_path, neighbors_path
    )
    artifacts = [
        path for path in output.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"
    ]
    write_json(
        output / "artifact_hashes.json",
        {str(path.relative_to(output)): sha256(path) for path in artifacts},
    )
    print(json.dumps(final, ensure_ascii=False))
    return 0 if final["all_runs_complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
