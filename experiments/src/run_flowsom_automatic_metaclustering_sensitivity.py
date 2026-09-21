from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get("CYTOMETRY_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "raw"))
)

import numpy as np
import pandas as pd
from evaluate_xshift_cross_dataset import evaluate_multiclass, metrics_are_valid
from sklearn.metrics import adjusted_rand_score

PARENTS = {
    "Levine_32dim": (
        "EXP-021A_flowsom_levine32_stability30_20260910",
        14,
        DATA_ROOT / "Levine_32dim" / "Levine_32dim_notransform.txt",
    ),
    "Samusik_01": (
        "EXP-021B_flowsom_samusik_stability30_20260910",
        24,
        DATA_ROOT / "Samusik_01" / "Samusik_01_notransform.txt",
    ),
}
MAIN_REGIMES = ("official_auto_max40", "official_fixed_ktrue", "official_fixed_k40")
SENSITIVITY_REGIME = "official_auto_max40_selector_seed_sensitivity"
METRICS = ("ari", "macro_precision", "macro_recall", "macro_f1", "hungarian_accuracy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    args = parser.parse_args()
    workspace, protocol, output = (
        args.workspace.resolve(),
        args.protocol.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    codes_dir = output / "codes_inputs"
    r_dir = output / "r_outputs"
    event_dir = output / "event_partitions"
    codes_dir.mkdir()
    r_dir.mkdir()
    event_dir.mkdir()
    exp_root = workspace / "experiments"
    runs_root = exp_root / "runs"
    r_script = exp_root / "src" / "run_flowsom_r_automatic_metaclustering_batch.R"
    r_library = exp_root / "environments" / "blflowsom_r452" / "library"

    input_rows: list[dict[str, object]] = []
    parent_hashes: list[dict[str, object]] = []
    parent_nodes: dict[tuple[str, int], np.ndarray] = {}
    labels_by_dataset: dict[str, np.ndarray] = {}
    for dataset, (parent_name, true_k, data_path) in PARENTS.items():
        parent = runs_root / parent_name
        progress = json.loads((parent / "progress_manifest.json").read_text(encoding="utf-8"))
        if progress.get("all_runs_complete_and_passed") is not True:
            raise RuntimeError(f"ineligible parent: {parent}")
        labels = (
            pd.read_csv(data_path, usecols=["label"], low_memory=False)["label"]
            .astype(str)
            .to_numpy()
        )
        labels_by_dataset[dataset] = labels
        for seed in range(30):
            run_dir = parent / "runs" / f"seed{seed:03d}"
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("all_checks_passed") is not True:
                raise RuntimeError(f"ineligible parent run: {run_dir}")
            codes_path = run_dir / "som_codes.npy"
            nodes_path = run_dir / "node_labels_all_events.npy"
            codes = np.load(codes_path, allow_pickle=False).astype(np.float64, copy=False)
            nodes = np.load(nodes_path, allow_pickle=False).astype(np.int16, copy=False)
            if codes.shape[0] != 100 or nodes.shape != labels.shape:
                raise ValueError(f"parent shape contract failed: {dataset} seed {seed}")
            csv_path = codes_dir / f"{dataset}_som_seed{seed:03d}.csv"
            pd.DataFrame(
                codes, columns=[f"feature_{i + 1:02d}" for i in range(codes.shape[1])]
            ).to_csv(csv_path, index=False)
            input_rows.append(
                {
                    "dataset": dataset,
                    "som_seed": seed,
                    "codes_path": str(csv_path.resolve()),
                    "true_k": true_k,
                }
            )
            parent_nodes[(dataset, seed)] = nodes
            for role, path in (
                ("parent_manifest", run_dir / "run_manifest.json"),
                ("som_codes", codes_path),
                ("node_labels", nodes_path),
            ):
                parent_hashes.append(
                    {
                        "dataset": dataset,
                        "som_seed": seed,
                        "role": role,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    pd.DataFrame(input_rows).to_csv(output / "r_input_manifest.csv", index=False)
    pd.DataFrame(parent_hashes).to_csv(output / "parent_input_hashes.csv", index=False)

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(r_library)
    command = [
        str(args.rscript.resolve()),
        str(r_script.resolve()),
        str((output / "r_input_manifest.csv").resolve()),
        str(r_dir.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    (output / "r_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "r_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        write_json(
            output / "failure.json",
            {
                "stage": "R automatic metaclustering",
                "exit_code": completed.returncode,
                "scientific_output_eligible": False,
            },
        )
        return 1

    r_summary = pd.read_csv(r_dir / "r_run_summary.csv")
    mappings = pd.read_csv(r_dir / "node_metacluster_mappings.csv")
    metric_rows: list[dict[str, object]] = []
    population_frames, cluster_frames, assignment_frames = [], [], []
    event_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    for row in r_summary.itertuples(index=False):
        selector_seed = int(row.selector_seed)
        subset = mappings[
            (mappings.dataset == row.dataset)
            & (mappings.som_seed == row.som_seed)
            & (mappings.regime == row.regime)
            & (mappings.selector_seed == selector_seed)
        ].sort_values("node_index_0based")
        node_map = subset["metacluster_label_1based"].to_numpy(dtype=np.int16)
        if node_map.shape != (100,):
            raise ValueError(f"mapping shape failed: {row}")
        nodes = parent_nodes[(row.dataset, int(row.som_seed))]
        event_labels = node_map[nodes]
        labels = labels_by_dataset[row.dataset]
        evaluable = labels != "unassigned"
        metrics, populations, clusters, assignment, _ = evaluate_multiclass(
            labels[evaluable], event_labels[evaluable], event_labels
        )
        ok, detail = metrics_are_valid(metrics)
        if not ok:
            raise ValueError(detail)
        metric_rows.append(
            {
                "dataset": row.dataset,
                "som_seed": int(row.som_seed),
                "regime": row.regime,
                "selector_seed": selector_seed,
                "requested_k": int(row.requested_k),
                "selected_k": int(row.selected_k),
                "occupied_node_metaclusters": int(row.occupied_node_metaclusters),
                "occupied_event_metaclusters": int(np.unique(event_labels).size),
                "r_runtime_seconds": float(row.r_runtime_seconds),
                "label_information_used_for_selection": False,
                **metrics,
            }
        )
        keys = {
            "dataset": row.dataset,
            "som_seed": int(row.som_seed),
            "regime": row.regime,
            "selector_seed": selector_seed,
        }
        population_frames.append(populations.assign(**keys))
        cluster_frames.append(clusters.assign(**keys))
        assignment_frames.append(assignment.assign(**keys))
        event_arrays.setdefault((row.dataset, row.regime), []).append(
            event_labels.astype(np.int16, copy=False)
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "run_level_metrics.csv", index=False)
    pd.concat(population_frames, ignore_index=True).to_csv(
        output / "population_level_metrics.csv", index=False
    )
    pd.concat(cluster_frames, ignore_index=True).to_csv(
        output / "metacluster_level_diagnostics.csv", index=False
    )
    pd.concat(assignment_frames, ignore_index=True).to_csv(
        output / "hungarian_mappings.csv", index=False
    )
    for (dataset, regime), arrays in event_arrays.items():
        key = "partitions"
        np.savez_compressed(event_dir / f"{dataset}__{regime}.npz", **{key: np.stack(arrays)})

    distribution_rows = []
    for (dataset, regime), group in metrics_frame.groupby(["dataset", "regime"], sort=True):
        for metric in ("selected_k", *METRICS):
            values = group[metric].to_numpy(dtype=float)
            distribution_rows.append(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(output / "distribution_summary.csv", index=False)

    paired_rows, contrast_rows = [], []
    rng = np.random.default_rng(20_260_910)
    main = metrics_frame[metrics_frame.regime.isin(MAIN_REGIMES)]
    for dataset in PARENTS:
        pivot = main[main.dataset == dataset].pivot(
            index="som_seed", columns="regime", values=list(METRICS)
        )
        for comparator in ("official_fixed_ktrue", "official_fixed_k40"):
            for metric in METRICS:
                diffs = (
                    pivot[(metric, "official_auto_max40")].to_numpy()
                    - pivot[(metric, comparator)].to_numpy()
                )
                for seed, value in zip(pivot.index, diffs, strict=True):
                    paired_rows.append(
                        {
                            "dataset": dataset,
                            "contrast": f"auto_minus_{comparator}",
                            "metric": metric,
                            "som_seed": int(seed),
                            "difference": float(value),
                        }
                    )
                draws = rng.choice(diffs, size=(10_000, len(diffs)), replace=True).mean(axis=1)
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "contrast": f"auto_minus_{comparator}",
                        "metric": metric,
                        "n": len(diffs),
                        "mean_difference": float(diffs.mean()),
                        "sd_difference": float(diffs.std(ddof=1)),
                        "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                        "bootstrap_replicates": 10_000,
                    }
                )
    pd.DataFrame(paired_rows).to_csv(output / "paired_differences.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output / "paired_contrast_summary.csv", index=False)

    pair_rows = []
    for dataset in PARENTS:
        archive = np.load(event_dir / f"{dataset}__{SENSITIVITY_REGIME}.npz", allow_pickle=False)[
            "partitions"
        ]
        for i in range(30):
            for j in range(i + 1, 30):
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "selector_seed_a": i,
                        "selector_seed_b": j,
                        "partition_ari": adjusted_rand_score(archive[i], archive[j]),
                    }
                )
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output / "selector_seed_pairwise_partition_ari.csv", index=False)

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    r_checks = pd.read_csv(r_dir / "r_checks.csv")
    check(
        "r_checks",
        len(r_checks) == 9 and bool(r_checks.passed.all()),
        f"passed={int(r_checks.passed.sum())}/{len(r_checks)}",
    )
    check(
        "row_counts",
        len(metrics_frame) == 240
        and len(main) == 180
        and int((metrics_frame.regime == SENSITIVITY_REGIME).sum()) == 60,
        f"total={len(metrics_frame)}; main={len(main)}",
    )
    check(
        "regime_counts",
        all(len(group) == 30 for _, group in metrics_frame.groupby(["dataset", "regime"])),
        str(metrics_frame.groupby(["dataset", "regime"]).size().to_dict()),
    )
    check(
        "selection_range",
        metrics_frame.selected_k.between(2, 40).all(),
        f"range={metrics_frame.selected_k.min()}-{metrics_frame.selected_k.max()}",
    )
    check(
        "label_information",
        not metrics_frame.label_information_used_for_selection.any(),
        "all false",
    )
    check(
        "metric_ranges",
        metrics_frame[list(METRICS)]
        .apply(lambda col: col.between(-1 if col.name == "ari" else 0, 1).all())
        .all(),
        "all endpoints in legal ranges",
    )
    check(
        "partition_archives",
        len(list(event_dir.glob("*.npz"))) == 8
        and all(
            np.load(path, allow_pickle=False)["partitions"].shape[0] == 30
            for path in event_dir.glob("*.npz")
        ),
        "8 archives x 30 partitions",
    )
    check(
        "paired_design",
        len(paired_rows) == 600 and len(contrast_rows) == 20,
        f"paired={len(paired_rows)}; summaries={len(contrast_rows)}",
    )
    check(
        "selector_pair_counts",
        len(pair_frame) == 870 and pair_frame.partition_ari.between(-1, 1).all(),
        f"pairs={len(pair_frame)}",
    )
    check(
        "parent_hash_rows",
        len(parent_hashes) == 180
        and all(
            Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
            for row in parent_hashes
        ),
        f"hashes={len(parent_hashes)}",
    )
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "checks.csv", index=False)
    all_passed = bool(checks_frame.passed.all())
    result = {
        "experiment_id": "EXP-032",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "r_script": str(r_script),
        "r_script_sha256": sha256(r_script),
        "rscript": str(args.rscript.resolve()),
        "parent_runs": {dataset: str(runs_root / spec[0]) for dataset, spec in PARENTS.items()},
        "r_flowsom_version": "2.18.0",
        "auto_max": 40,
        "fixed_metaclustering_seed": 12345,
        "main_rows": len(main),
        "selector_sensitivity_rows": int((metrics_frame.regime == SENSITIVITY_REGIME).sum()),
        "checks_passed": int(checks_frame.passed.sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": all_passed,
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    write_json(output / "run_manifest.json", result)
    auto_counts = (
        metrics_frame[metrics_frame.regime == "official_auto_max40"]
        .groupby(["dataset", "selected_k"])
        .size()
        .to_dict()
    )
    selector_counts = (
        metrics_frame[metrics_frame.regime == SENSITIVITY_REGIME]
        .groupby(["dataset", "selected_k"])
        .size()
        .to_dict()
    )
    (output / "scientific_summary.md").write_text(
        "# EXP-032 Sensitivity of official FlowSOM automatic metacluster selection\n\n"
        f"Automatic-K counts in the primary paired analysis: {auto_counts}.\n\nAutomatic-K counts across selector seeds: {selector_counts}.\n\n"
        f"Checks passed: {result['checks_passed']}/{result['checks_total']}. Complete metrics and paired differences for automatic K, K=true, and fixed K=40 are retained in machine-readable tables.\n",
        encoding="utf-8",
    )
    artifacts = [
        path for path in output.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"
    ]
    write_json(
        output / "artifact_hashes.json",
        {str(path.relative_to(output)): sha256(path) for path in artifacts},
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
