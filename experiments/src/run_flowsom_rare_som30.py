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
from flowsom.models import map_data_to_codes
from loguru import logger


DATASETS = {
    "Nilsson_rare": {
        "rows": 44_140,
        "markers": 13,
        "target": "HSCs",
        "target_events": 358,
        "sha256": "e74c804a6cb040fd5cf1bea3c9d1ce382b266d860209d443f7f3960f129c918c",
    },
    "Mosmann_rare": {
        "rows": 396_460,
        "markers": 14,
        "target": "activated",
        "target_events": 109,
        "sha256": "684541587d408477b89da4882e237000ab03353bc9cb440cd4a1bc27736d81ae",
    },
}
SEEDS = tuple(range(30))
TRAINING_SIZE = 20_000
TRAINING_SAMPLE_SEED = 20_260_911
GRID_SIDE = 10
RLEN = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data(config_path: Path, dataset: str) -> tuple[Path, list[str], np.ndarray, np.ndarray]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["datasets"][dataset]
    source = (Path(config["data_root"]) / dataset / spec["filename"]).resolve()
    sep = {"comma": ",", "tab": "\t"}[spec["separator"]]
    frame = pd.read_csv(source, sep=sep, low_memory=False)
    markers = [column for column in frame.columns if column not in set(spec["exclude_columns"])]
    matrix = np.arcsinh(frame[markers].to_numpy(dtype=np.float64) / float(spec["cofactor"]))
    labels = frame["label"].astype(str).to_numpy()
    return source, markers, matrix, labels


def rebuild_index(output: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run_dir = output / "runs" / dataset / f"seed{seed:03d}"
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "runtime_seconds": manifest["runtime_seconds"],
                    "occupied_nodes": manifest["occupied_nodes"],
                    "checks_passed": manifest["checks_passed"],
                    "checks_total": manifest["checks_total"],
                    "all_checks_passed": manifest["all_checks_passed"],
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "run_index.csv", index=False)
    return frame


def write_progress(output: Path, protocol: Path, config: Path, qualification: Path, failed_run: str | None = None) -> dict[str, object]:
    index = rebuild_index(output)
    passed = int(index.all_checks_passed.astype(bool).sum()) if len(index) else 0
    progress = {
        "experiment_id": "EXP-033A",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failed_run else ("complete" if len(index) == passed == 60 else "in_progress"),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "config": str(config), "config_sha256": sha256(config),
        "qualification": str(qualification), "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
        "datasets": list(DATASETS), "seeds": list(SEEDS), "training_size": TRAINING_SIZE,
        "training_sample_seed": TRAINING_SAMPLE_SEED, "grid_side": GRID_SIDE, "rlen": RLEN,
        "expected_runs": 60, "completed_runs": len(index), "passed_runs": passed,
        "failed_run": failed_run, "all_runs_complete_and_passed": len(index) == passed == 60 and failed_run is None,
    }
    write_json(output / "progress_manifest.json", progress)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, protocol, qualification, output = (args.config.resolve(), args.protocol.resolve(), args.qualification.resolve(), args.output.resolve())
    q_manifest = json.loads((qualification / "run_manifest.json").read_text(encoding="utf-8"))
    if q_manifest.get("all_checks_passed") is not True or importlib.metadata.version("flowsom") != "0.2.2":
        raise RuntimeError("EXP-013-R2 qualified FlowSOM 0.2.2 environment required")
    if output.exists():
        previous = json.loads((output / "progress_manifest.json").read_text(encoding="utf-8"))
        if previous.get("protocol_sha256") != sha256(protocol) or previous.get("script_sha256") != sha256(Path(__file__).resolve()):
            raise RuntimeError("Resume contract differs from current protocol or source")
    else:
        (output / "runs").mkdir(parents=True)
    logger.disable("flowsom")
    write_progress(output, protocol, config, qualification)

    for dataset, contract in DATASETS.items():
        source, markers, matrix, labels = load_data(config, dataset)
        if sha256(source) != contract["sha256"] or matrix.shape != (contract["rows"], contract["markers"]):
            raise ValueError(f"input contract failed for {dataset}")
        if int(np.sum(labels == contract["target"])) != contract["target_events"]:
            raise ValueError(f"target count failed for {dataset}")
        dataset_dir = output / "runs" / dataset
        dataset_dir.mkdir(exist_ok=True)
        marker_path = dataset_dir / "marker_columns.txt"
        index_path = dataset_dir / "training_indices.npy"
        if not marker_path.exists():
            marker_path.write_text("\n".join(markers) + "\n", encoding="utf-8")
        if not index_path.exists():
            rng = np.random.default_rng(TRAINING_SAMPLE_SEED)
            np.save(index_path, np.sort(rng.choice(len(matrix), size=TRAINING_SIZE, replace=False)).astype(np.int64))
        training_indices = np.load(index_path, allow_pickle=False)
        if training_indices.shape != (TRAINING_SIZE,) or len(np.unique(training_indices)) != TRAINING_SIZE or training_indices.min() < 0 or training_indices.max() >= len(matrix):
            raise ValueError(f"training index contract failed for {dataset}")
        training_matrix = matrix[training_indices]

        for seed in SEEDS:
            run_dir = dataset_dir / f"seed{seed:03d}"
            if (run_dir / "run_manifest.json").is_file():
                continue
            if run_dir.exists():
                raise RuntimeError(f"Incomplete run preserved and blocks resume: {run_dir}")
            run_dir.mkdir()
            try:
                started = time.perf_counter()
                model = fs.FlowSOM(
                    ad.AnnData(X=pd.DataFrame(training_matrix, columns=markers)),
                    cols_to_use=markers, n_clusters=40, xdim=GRID_SIDE, ydim=GRID_SIDE,
                    rlen=RLEN, seed=seed,
                )
                codes = np.asarray(model.model.codes, dtype=np.float64)
                node_float, distances = map_data_to_codes(matrix, codes)
                nodes = node_float.astype(np.int16)
                distances = np.asarray(distances, dtype=np.float32)
                runtime = time.perf_counter() - started
                np.save(run_dir / "som_codes.npy", codes)
                np.save(run_dir / "node_labels_all_events.npy", nodes)
                np.save(run_dir / "bmu_distances_all_events.npy", distances)
                checks: list[dict[str, object]] = []
                def check(name: str, passed: bool, detail: str) -> None:
                    checks.append({"check": name, "passed": bool(passed), "detail": detail})
                check("source_hash", sha256(source) == contract["sha256"], sha256(source))
                check("marker_order", marker_path.read_text(encoding="utf-8").splitlines() == markers, f"markers={len(markers)}")
                check("training_indices", training_indices.shape == (TRAINING_SIZE,) and len(np.unique(training_indices)) == TRAINING_SIZE, sha256(index_path))
                check("code_shape", codes.shape == (100, contract["markers"]), str(codes.shape))
                check("event_shape", nodes.shape == distances.shape == (contract["rows"],), str(nodes.shape))
                check("node_range", nodes.min() >= 0 and nodes.max() < 100, f"{nodes.min()}-{nodes.max()}")
                check("finite", np.isfinite(codes).all() and np.isfinite(distances).all() and np.all(distances >= 0), f"distance={distances.min()}-{distances.max()}")
                check("label_permission", True, "labels used only before run for target-count input validation")
                checks_frame = pd.DataFrame(checks)
                checks_frame.to_csv(run_dir / "checks.csv", index=False)
                manifest = {
                    "experiment_id": "EXP-033A", "dataset": dataset, "seed": seed,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "source": str(source), "source_sha256": sha256(source),
                    "protocol": str(protocol), "protocol_sha256": sha256(protocol),
                    "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
                    "qualification": str(qualification), "qualification_manifest_sha256": sha256(qualification / "run_manifest.json"),
                    "training_indices": str(index_path), "training_indices_sha256": sha256(index_path),
                    "training_size": TRAINING_SIZE, "training_sample_seed": TRAINING_SAMPLE_SEED,
                    "fit_policy": "fixed_unlabeled_full_event_sample_then_map_all_events",
                    "label_used_for_sampling_fit_or_parameter_selection": False,
                    "python_flowsom_constructor_n_clusters_ignored_downstream": 40,
                    "grid_side": GRID_SIDE, "rlen": RLEN, "runtime_seconds": runtime,
                    "occupied_nodes": int(np.unique(nodes).size),
                    "checks_passed": int(checks_frame.passed.sum()), "checks_total": len(checks_frame),
                    "all_checks_passed": bool(checks_frame.passed.all()),
                    "python": sys.version, "python_executable": sys.executable,
                    "flowsom": importlib.metadata.version("flowsom"), "numpy": np.__version__, "pandas": pd.__version__,
                    "platform": platform.platform(),
                }
                write_json(run_dir / "run_manifest.json", manifest)
                write_json(run_dir / "artifact_hashes.json", {p.name: sha256(p) for p in run_dir.iterdir() if p.is_file()})
                if not manifest["all_checks_passed"]:
                    raise RuntimeError("run checks failed")
                print(json.dumps({"dataset": dataset, "seed": seed, "runtime": runtime, "occupied_nodes": manifest["occupied_nodes"], "checks": "8/8"}, ensure_ascii=False), flush=True)
            except Exception as exc:
                (run_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                write_json(run_dir / "failure.json", {"exception": repr(exc), "scientific_output_eligible": False})
                write_progress(output, protocol, config, qualification, str(run_dir.relative_to(output)))
                raise
            write_progress(output, protocol, config, qualification)
    final = write_progress(output, protocol, config, qualification)
    artifacts = [p for p in output.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"]
    write_json(output / "artifact_hashes.json", {str(p.relative_to(output)): sha256(p) for p in artifacts})
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if final["all_runs_complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
