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
from evaluate_xshift_cross_dataset import evaluate_multiclass
from run_flowsom_grid_rlen_factorial import load_data

DATASETS = ["Levine_32dim", "Samusik_01"]
METRICS = ["ari", "macro_precision", "macro_recall", "macro_f1", "hungarian_accuracy"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inventory_failures(path: Path) -> list[str]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    return [
        name
        for name, expected in inventory.items()
        if not Path(name).is_file() or sha256(Path(name)) != expected
    ]


def load_string_array(path: Path) -> np.ndarray:
    """Load trusted, hash-checked label arrays and reject non-string objects."""
    values = np.load(path, allow_pickle=True)
    if values.dtype.kind == "O":
        if not all(isinstance(value, str) for value in values.ravel()):
            raise TypeError(f"Non-string object found in label array: {path}")
        return values.astype(str)
    if values.dtype.kind not in {"U", "S"}:
        raise TypeError(f"Expected a string label array, got {values.dtype}: {path}")
    return values.astype(str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent_run.resolve()
    config = args.config.resolve()
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    failure = json.loads((parent / "failure_manifest.json").read_text(encoding="utf-8"))
    checks: list[dict] = []
    results: list[dict] = []
    populations: list[pd.DataFrame] = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    parent_ok = (
        failure.get("status") == "event_level_scenarios_complete_top_level_summary_failed"
        and failure.get("dataset_scenario_artifacts_complete") is True
    )
    check("experiment", "parent_failure", parent_ok, json.dumps(failure))

    for dataset in DATASETS:
        dataset_dir = parent / "datasets" / dataset
        failures = inventory_failures(dataset_dir / "artifact_hashes.json")
        check(dataset, "parent_artifact_hashes", not failures, json.dumps(failures))
        source_links = json.loads((dataset_dir / "source_links.json").read_text(encoding="utf-8"))
        source, _, markers, matrix, all_labels = load_data(config, dataset)
        evaluable = all_labels != "unassigned"
        expected_labels = all_labels[evaluable]
        labels = load_string_array(dataset_dir / "reference_labels.npy")
        evaluation_indices = np.load(dataset_dir / "evaluation_indices.npy", allow_pickle=False)
        source_ok = (
            str(source.resolve()) == str(Path(source_links["data"]).resolve())
            and sha256(source) == source_links["data_sha256"]
        )
        labels_ok = np.array_equal(labels, expected_labels) and np.array_equal(
            evaluation_indices, np.flatnonzero(evaluable)
        )
        check(
            dataset,
            "source_and_reference",
            source_ok and labels_ok,
            f"source={sha256(source)}; events={len(labels)}; markers={len(markers)}",
        )

        predictions: dict[str, np.ndarray] = {}
        for algorithm, link in source_links["prediction_runs"].items():
            run = Path(link["path"])
            manifest_ok = sha256(run / "run_manifest.json") == link["manifest_sha256"]
            if algorithm == "X-shift":
                full_predictions = np.load(run / "cluster_ids_all_events.npy", allow_pickle=False)
            else:
                full_predictions = np.load(
                    run / "metacluster_labels_all_events.npy", allow_pickle=False
                )
            predictions[algorithm] = full_predictions[evaluable]
            check(
                dataset,
                f"{algorithm}_prediction_link",
                manifest_ok and len(full_predictions) == len(matrix),
                f"run={run}; events={len(full_predictions)}",
            )

        neighbors = np.load(dataset_dir / "neighbor_indices_k15.npy", allow_pickle=False)
        disagreement = np.load(dataset_dir / "neighbor_disagreement.npy", allow_pickle=False)
        alternative = load_string_array(dataset_dir / "alternative_neighbor_label.npy")
        has_alternative = np.load(dataset_dir / "has_alternative_label.npy", allow_pickle=False)
        uncertainty_ok = neighbors.shape == (len(labels), 15) and not np.any(
            neighbors == np.arange(len(labels))[:, None]
        )
        uncertainty_ok = (
            uncertainty_ok
            and disagreement.shape == (len(labels),)
            and disagreement.min() >= 0
            and disagreement.max() <= 1
        )
        uncertainty_ok = uncertainty_ok and alternative.shape == has_alternative.shape == (
            len(labels),
        )
        check(
            dataset,
            "uncertainty_arrays",
            uncertainty_ok,
            f"neighbors={neighbors.shape}; disagreement={float(disagreement.min())}-{float(disagreement.max())}; alternatives={int(has_alternative.sum())}",
        )

        scenario_table = pd.read_csv(dataset_dir / "scenario_manifest.csv")
        scenario_set_ok = (
            len(scenario_table) == 37
            and (scenario_table["kind"] == "random_control_flip").sum() == 30
        )
        check(
            dataset,
            "scenario_set",
            scenario_set_ok,
            str(scenario_table.groupby("kind").size().to_dict()),
        )
        for scenario in scenario_table.itertuples(index=False):
            labels_path = dataset_dir / "scenarios" / f"{scenario.scenario}_labels.npy"
            mask_path = dataset_dir / "scenarios" / f"{scenario.scenario}_keep_mask.npy"
            scenario_labels = load_string_array(labels_path)
            keep = np.load(mask_path, allow_pickle=False)
            changed = int(np.sum(scenario_labels != labels))
            excluded = int(np.sum(~keep))
            scenario_ok = (
                sha256(labels_path) == scenario.labels_sha256
                and sha256(mask_path) == scenario.mask_sha256
            )
            scenario_ok = (
                scenario_ok
                and scenario_labels.shape == keep.shape == labels.shape
                and changed == int(scenario.changed_labels)
                and excluded == int(scenario.excluded_events)
            )
            check(
                dataset,
                f"scenario_{scenario.scenario}",
                scenario_ok,
                f"changed={changed}; excluded={excluded}; retained={int(keep.sum())}",
            )
            for algorithm, predicted in predictions.items():
                metrics, population, _, _, _ = evaluate_multiclass(
                    scenario_labels[keep], predicted[keep], predicted[keep]
                )
                results.append(
                    {
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "scenario": scenario.scenario,
                        "kind": scenario.kind,
                        "nominal_fraction": float(scenario.nominal_fraction),
                        "seed": scenario.seed,
                        "changed_labels": changed,
                        "excluded_events": excluded,
                        **metrics,
                    }
                )
                population.insert(0, "scenario", scenario.scenario)
                population.insert(0, "algorithm", algorithm)
                population.insert(0, "dataset", dataset)
                populations.append(population)

    results_frame = pd.DataFrame(results)
    baseline_columns = {metric: f"baseline_{metric}" for metric in METRICS}
    baseline = results_frame.loc[
        results_frame["kind"] == "unchanged", ["dataset", "algorithm", *METRICS]
    ].rename(columns=baseline_columns)
    results_frame = results_frame.merge(
        baseline, on=["dataset", "algorithm"], how="left", validate="many_to_one"
    )
    for metric in METRICS:
        results_frame[f"delta_{metric}_from_baseline"] = (
            results_frame[metric] - results_frame[f"baseline_{metric}"]
        )
    results_frame.to_csv(output / "scenario_metrics.csv", index=False)
    pd.concat(populations, ignore_index=True).to_csv(
        output / "scenario_population_metrics.csv", index=False
    )

    delta_columns = [
        "delta_ari_from_baseline",
        "delta_macro_f1_from_baseline",
        "delta_hungarian_accuracy_from_baseline",
    ]
    random_summary = (
        results_frame.loc[results_frame["kind"] == "random_control_flip"]
        .groupby(["dataset", "algorithm"])[delta_columns]
        .agg(["mean", "std", "min", "max"])
    )
    random_summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in random_summary.columns
    ]
    random_summary.reset_index().to_csv(output / "random_control_summary.csv", index=False)
    results_frame.loc[
        results_frame["kind"].isin(["unchanged", "boundary_flip", "boundary_exclude"])
    ].to_csv(output / "deterministic_scenario_metrics.csv", index=False)

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "finalization_checks.csv", index=False)
    all_passed = bool(checks_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-019-R2",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run": str(parent),
        "parent_failure_manifest_sha256": sha256(parent / "failure_manifest.json"),
        "config": str(config),
        "config_sha256": sha256(config),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "datasets": DATASETS,
        "algorithms": sorted(results_frame["algorithm"].unique().tolist()),
        "scenarios_per_dataset": 37,
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": all_passed,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ten_percent = results_frame.loc[
        results_frame["scenario"] == "boundary_flip_10pct",
        ["dataset", "algorithm", "delta_ari_from_baseline", "delta_macro_f1_from_baseline"],
    ]
    lines = [
        "# EXP-019-R2 Reference-label uncertainty sensitivity",
        "",
        f"Parent scenarios were not regenerated. Event-level labels and masks were used to recompute 37 scenarios for each of two algorithms on two datasets. Finalization checks passed: {manifest['checks_passed']}/{manifest['checks_total']}.",
        "",
        "Ten-percent boundary flips relative to baseline:",
    ]
    for row in ten_percent.itertuples(index=False):
        lines.append(
            f"- {row.dataset} / {row.algorithm}: ΔARI={row.delta_ari_from_baseline:+.6f}, ΔMacro F1={row.delta_macro_f1_from_baseline:+.6f}."
        )
    lines.extend(
        [
            "",
            "These perturbation proportions are not empirical mislabeling rates, and the results do not replace real multi-gater agreement data.",
        ]
    )
    (output / "scientific_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps({str(path): sha256(path) for path in artifacts}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
