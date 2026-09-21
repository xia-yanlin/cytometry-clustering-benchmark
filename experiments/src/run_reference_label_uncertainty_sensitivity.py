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
from sklearn.neighbors import NearestNeighbors

DATASETS = {
    "Levine_32dim": {
        "events": 265627,
        "evaluable": 104184,
        "populations": 14,
        "sha": "bb1f9cb63377f701795f116e557f8ede05141b6aca2a0270177c7b1eb4c807e4",
    },
    "Samusik_01": {
        "events": 86864,
        "evaluable": 53173,
        "populations": 24,
        "sha": "86dd329dedec30b86ab0cc845923620393a36f16bd07d8f5d87ba4dbb3511565",
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


def ranked_indices(
    labels: np.ndarray, scores: np.ndarray, eligible: np.ndarray, fraction: float
) -> np.ndarray:
    chosen = []
    for label in sorted(np.unique(labels)):
        members = np.flatnonzero((labels == label) & eligible)
        count = min(len(members), max(1, int(round(fraction * int(np.sum(labels == label))))))
        order = np.lexsort((members, -scores[members]))
        chosen.extend(members[order[:count]].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def random_indices(
    labels: np.ndarray, eligible: np.ndarray, target_counts: dict[str, int], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(np.unique(labels)):
        members = np.flatnonzero((labels == label) & eligible)
        count = min(len(members), target_counts[label])
        chosen.extend(rng.choice(members, size=count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def local_uncertainty(
    matrix: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    standardized = (matrix - means) / scales
    model = NearestNeighbors(n_neighbors=16, metric="euclidean", algorithm="auto", n_jobs=1)
    neighbors = model.fit(standardized).kneighbors(return_distance=False)
    cleaned = np.empty((len(matrix), 15), dtype=np.int64)
    for row, values in enumerate(neighbors):
        without_self = values[values != row]
        if len(without_self) < 15:
            raise RuntimeError(f"Could not remove self neighbor for row {row}")
        cleaned[row] = without_self[:15]
    neighbor_labels = labels[cleaned]
    disagreement = np.mean(neighbor_labels != labels[:, None], axis=1)
    alternative = np.empty(len(labels), dtype=labels.dtype)
    has_alternative = np.zeros(len(labels), dtype=bool)
    for row in range(len(labels)):
        values, counts = np.unique(
            neighbor_labels[row][neighbor_labels[row] != labels[row]], return_counts=True
        )
        if len(values):
            best_count = counts.max()
            alternative[row] = sorted(values[counts == best_count].tolist())[0]
            has_alternative[row] = True
        else:
            alternative[row] = labels[row]
    return cleaned, disagreement.astype(np.float32), alternative, has_alternative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--levine-xshift", type=Path, required=True)
    parser.add_argument("--samusik-xshift", type=Path, required=True)
    parser.add_argument("--levine-flowsom", type=Path, required=True)
    parser.add_argument("--samusik-flowsom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config, protocol, output = args.config.resolve(), args.protocol.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "datasets").mkdir()
    prediction_runs = {
        "Levine_32dim": {
            "X-shift": args.levine_xshift.resolve(),
            "FlowSOM": args.levine_flowsom.resolve(),
        },
        "Samusik_01": {
            "X-shift": args.samusik_xshift.resolve(),
            "FlowSOM": args.samusik_flowsom.resolve(),
        },
    }
    checks: list[dict] = []
    results: list[dict] = []
    population_results: list[pd.DataFrame] = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    for dataset, contract in DATASETS.items():
        dataset_dir = output / "datasets" / dataset
        dataset_dir.mkdir()
        (dataset_dir / "scenarios").mkdir()
        source, _, markers, matrix, all_labels = load_data(config, dataset)
        evaluable = all_labels != "unassigned"
        evaluation_indices = np.flatnonzero(evaluable).astype(np.int64)
        labels = all_labels[evaluable]
        eval_matrix = matrix[evaluable]
        check(
            dataset,
            "data_contract",
            sha256(source) == contract["sha"]
            and matrix.shape[0] == contract["events"]
            and len(labels) == contract["evaluable"]
            and len(np.unique(labels)) == contract["populations"],
            f"source={sha256(source)}; matrix={matrix.shape}; evaluable={len(labels)}; populations={len(np.unique(labels))}",
        )
        np.save(dataset_dir / "evaluation_indices.npy", evaluation_indices)
        np.save(dataset_dir / "reference_labels.npy", labels)
        (dataset_dir / "marker_columns.txt").write_text("\n".join(markers) + "\n", encoding="utf-8")

        predictions = {}
        for algorithm, run in prediction_runs[dataset].items():
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            manifest_ok = manifest.get("all_checks_passed") is True
            if algorithm == "X-shift":
                full_predictions = np.load(run / "cluster_ids_all_events.npy", allow_pickle=False)
            else:
                full_predictions = np.load(
                    run / "metacluster_labels_all_events.npy", allow_pickle=False
                )
            predictions[algorithm] = full_predictions[evaluable]
            check(
                dataset,
                f"{algorithm}_prediction_contract",
                manifest_ok and full_predictions.shape == (contract["events"],),
                f"run={run}; manifest={manifest_ok}; shape={full_predictions.shape}",
            )

        neighbors, disagreement, alternative, has_alternative = local_uncertainty(
            eval_matrix, labels
        )
        np.save(dataset_dir / "neighbor_indices_k15.npy", neighbors)
        np.save(dataset_dir / "neighbor_disagreement.npy", disagreement)
        np.save(dataset_dir / "alternative_neighbor_label.npy", alternative)
        np.save(dataset_dir / "has_alternative_label.npy", has_alternative)
        neighbor_ok = (
            neighbors.shape == (len(labels), 15)
            and not np.any(neighbors == np.arange(len(labels))[:, None])
            and disagreement.min() >= 0
            and disagreement.max() <= 1
        )
        check(
            dataset,
            "local_uncertainty_contract",
            neighbor_ok,
            f"neighbors={neighbors.shape}; disagreement={float(disagreement.min())}-{float(disagreement.max())}; alternatives={int(has_alternative.sum())}",
        )

        scenarios: list[tuple[str, str, float, int | None, np.ndarray, np.ndarray]] = []
        scenarios.append(
            ("baseline", "unchanged", 0.0, None, labels.copy(), np.ones(len(labels), dtype=bool))
        )
        boundary_sets: dict[float, np.ndarray] = {}
        for fraction in (0.01, 0.05, 0.10):
            chosen = ranked_indices(labels, disagreement, has_alternative, fraction)
            boundary_sets[fraction] = chosen
            flipped = labels.copy()
            flipped[chosen] = alternative[chosen]
            scenarios.append(
                (
                    f"boundary_flip_{int(fraction * 100):02d}pct",
                    "boundary_flip",
                    fraction,
                    None,
                    flipped,
                    np.ones(len(labels), dtype=bool),
                )
            )
            excluded = ranked_indices(
                labels, disagreement, np.ones(len(labels), dtype=bool), fraction
            )
            keep = np.ones(len(labels), dtype=bool)
            keep[excluded] = False
            scenarios.append(
                (
                    f"boundary_exclude_{int(fraction * 100):02d}pct",
                    "boundary_exclude",
                    fraction,
                    None,
                    labels.copy(),
                    keep,
                )
            )
        counts5 = {
            str(label): int(np.sum(labels[boundary_sets[0.05]] == label))
            for label in np.unique(labels)
        }
        for seed in range(30):
            chosen = random_indices(labels, has_alternative, counts5, seed)
            flipped = labels.copy()
            flipped[chosen] = alternative[chosen]
            scenarios.append(
                (
                    f"random_control_05pct_seed{seed:02d}",
                    "random_control_flip",
                    0.05,
                    seed,
                    flipped,
                    np.ones(len(labels), dtype=bool),
                )
            )

        scenario_rows = []
        for name, kind, fraction, seed, scenario_labels, keep in scenarios:
            scenario_path = dataset_dir / "scenarios" / f"{name}_labels.npy"
            mask_path = dataset_dir / "scenarios" / f"{name}_keep_mask.npy"
            np.save(scenario_path, scenario_labels)
            np.save(mask_path, keep)
            changed = int(np.sum(scenario_labels != labels))
            excluded_count = int(np.sum(~keep))
            check(
                dataset,
                f"scenario_{name}",
                len(scenario_labels) == len(labels)
                and keep.shape == (len(labels),)
                and (kind != "boundary_flip" or changed > 0)
                and (kind != "boundary_exclude" or excluded_count > 0),
                f"changed={changed}; excluded={excluded_count}; retained={int(keep.sum())}",
            )
            scenario_rows.append(
                {
                    "scenario": name,
                    "kind": kind,
                    "nominal_fraction": fraction,
                    "seed": seed,
                    "changed_labels": changed,
                    "excluded_events": excluded_count,
                    "retained_events": int(keep.sum()),
                    "labels_sha256": sha256(scenario_path),
                    "mask_sha256": sha256(mask_path),
                }
            )
            for algorithm, predicted in predictions.items():
                metrics, population, _, _, _ = evaluate_multiclass(
                    scenario_labels[keep], predicted[keep], predicted[keep]
                )
                results.append(
                    {
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "scenario": name,
                        "kind": kind,
                        "nominal_fraction": fraction,
                        "seed": seed,
                        "changed_labels": changed,
                        "excluded_events": excluded_count,
                        **metrics,
                    }
                )
                population.insert(0, "scenario", name)
                population.insert(0, "algorithm", algorithm)
                population.insert(0, "dataset", dataset)
                population_results.append(population)
        pd.DataFrame(scenario_rows).to_csv(dataset_dir / "scenario_manifest.csv", index=False)
        write_json(
            dataset_dir / "source_links.json",
            {
                "data": str(source),
                "data_sha256": sha256(source),
                "prediction_runs": {
                    algorithm: {
                        "path": str(run),
                        "manifest_sha256": sha256(run / "run_manifest.json"),
                    }
                    for algorithm, run in prediction_runs[dataset].items()
                },
            },
        )
        write_json(
            dataset_dir / "artifact_hashes.json",
            {
                str(path): sha256(path)
                for path in dataset_dir.rglob("*")
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        )

    results_frame = pd.DataFrame(results)
    baseline = results_frame[results_frame.kind == "unchanged"].set_index(["dataset", "algorithm"])
    for metric in ("ari", "macro_precision", "macro_recall", "macro_f1", "hungarian_accuracy"):
        results_frame[f"delta_{metric}_from_baseline"] = [
            float(row[metric] - baseline.loc[(row.dataset, row.algorithm), metric])
            for row in results_frame.itertuples()
        ]
    results_frame.to_csv(output / "scenario_metrics.csv", index=False)
    pd.concat(population_results, ignore_index=True).to_csv(
        output / "scenario_population_metrics.csv", index=False
    )
    random_summary = (
        results_frame[results_frame.kind == "random_control_flip"]
        .groupby(["dataset", "algorithm"])[
            [
                f"delta_{metric}_from_baseline"
                for metric in ("ari", "macro_f1", "hungarian_accuracy")
            ]
        ]
        .agg(["mean", "std", "min", "max"])
    )
    random_summary.to_csv(output / "random_control_summary.csv")
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "verification_checks.csv", index=False)
    all_passed = bool(checks_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-019",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "config_sha256": sha256(config),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "prediction_runs": {
            dataset: {algorithm: str(path) for algorithm, path in runs.items()}
            for dataset, runs in prediction_runs.items()
        },
        "datasets": list(DATASETS),
        "algorithms": ["X-shift", "FlowSOM"],
        "knn_k": 15,
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
    paths = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps({str(path): sha256(path) for path in paths}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
