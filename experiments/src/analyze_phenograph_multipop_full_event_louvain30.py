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
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

ENDPOINTS = [
    "ari",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "hungarian_accuracy",
    "quality_q",
    "n_valid_communities",
    "overclustering_ratio_evaluable",
    "mean_population_effective_predicted_clusters",
    "weighted_evaluable_cluster_purity",
    "mean_cluster_effective_true_populations",
]
CORRELATION_ENDPOINTS = ["ari", "macro_f1", "hungarian_accuracy"]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_916


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_inventory(inventory_path: Path, base: Path) -> list[str]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    bad = []
    for name, expected in inventory.items():
        path = base / name
        if not path.is_file() or sha256(path) != expected:
            bad.append(name)
    return bad


def holm_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samusik", type=Path, required=True)
    parser.add_argument("--levine13", type=Path, required=True)
    parser.add_argument("--levine32", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parents = {
        "Levine_13dim": args.levine13.resolve(),
        "Levine_32dim": args.levine32.resolve(),
        "Samusik_01": args.samusik.resolve(),
    }
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks: list[dict] = []
    frames: list[pd.DataFrame] = []
    pair_rows: list[dict] = []
    source_links: list[dict] = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    for dataset, parent in parents.items():
        progress = json.loads((parent / "progress_manifest.json").read_text(encoding="utf-8"))
        index = pd.read_csv(parent / "run_index.csv").sort_values("repeat").reset_index(drop=True)
        passed = index["all_checks_passed"].astype(str).str.lower().eq("true")
        check(
            dataset,
            "parent_complete",
            progress.get("all_runs_complete_and_passed") is True
            and len(index) == 30
            and passed.all(),
            f"rows={len(index)}; passed={int(passed.sum())}",
        )
        check(
            dataset,
            "repeat_contract",
            index["repeat"].astype(int).tolist() == list(range(30)),
            str(index["repeat"].astype(int).tolist()),
        )
        check(
            dataset,
            "method_contract",
            index["algorithm"].eq("PhenoGraph_v1.5.7_default_Louvain").all()
            and index["k_neighbors"].astype(int).eq(30).all()
            and index["evaluation_scope"]
            .eq("all_events_fit_evaluable_reference_labels_only")
            .all(),
            "official_default_louvain_k30_full_event",
        )
        bad_top = verify_inventory(parent / "artifact_hashes.json", parent)
        check(dataset, "top_artifact_inventory", not bad_top, f"bad={len(bad_top)}")
        labels_by_repeat: dict[int, np.ndarray] = {}
        run_manifest_hashes = {}
        for repeat in range(30):
            run = parent / "runs" / f"repeat{repeat:03d}"
            labels_by_repeat[repeat] = np.load(
                run / "community_labels_all_events.npy", allow_pickle=False
            )
            run_manifest_hashes[str(repeat)] = sha256(run / "run_manifest.json")
        for first in range(30):
            for second in range(first + 1, 30):
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "repeat_a": first,
                        "repeat_b": second,
                        "partition_ari_all_events": float(
                            adjusted_rand_score(labels_by_repeat[first], labels_by_repeat[second])
                        ),
                    }
                )
        index.insert(0, "source_run", parent.name)
        frames.append(index)
        source_links.append(
            {
                "dataset": dataset,
                "path": str(parent),
                "progress_manifest_sha256": sha256(parent / "progress_manifest.json"),
                "run_index_sha256": sha256(parent / "run_index.csv"),
                "graph_sha256": sha256(parent / "fixed_full_event_graph.npz"),
                "neighbors_sha256": sha256(parent / "neighbor_indices_k30.npy"),
                "run_manifest_sha256_by_repeat": run_manifest_hashes,
            }
        )

    points = pd.concat(frames, ignore_index=True)
    pairs = (
        pd.DataFrame(pair_rows)
        .sort_values(["dataset", "repeat_a", "repeat_b"])
        .reset_index(drop=True)
    )
    points.to_csv(output / "all_run_points.csv", index=False)
    pairs.to_csv(output / "pairwise_partition_ari.csv", index=False)
    check(
        "experiment",
        "run_and_pair_counts",
        len(points) == 90 and len(pairs) == 1305 and pairs.groupby("dataset").size().eq(435).all(),
        f"runs={len(points)}; pairs={len(pairs)}",
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    descriptive_rows = []
    correlation_rows = []
    for dataset, group in points.groupby("dataset", sort=True):
        group = group.sort_values("repeat").reset_index(drop=True)
        for endpoint in ENDPOINTS:
            values = group[endpoint].to_numpy(dtype=float)
            resamples = values[
                rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
            ]
            bootstrap_means = resamples.mean(axis=1)
            q1, q3 = np.quantile(values, [0.25, 0.75])
            descriptive_rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "n_runs": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "mad_unscaled": float(np.median(np.abs(values - np.median(values)))),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": float(np.ptp(values)),
                    "bootstrap_mean_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                    "bootstrap_mean_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        for endpoint in CORRELATION_ENDPOINTS:
            x = group["quality_q"].to_numpy(dtype=float)
            y = group[endpoint].to_numpy(dtype=float)
            defined = bool(np.ptp(x) > 0 and np.ptp(y) > 0)
            if defined:
                result = spearmanr(x, y)
                rho, p_raw, reason = float(result.statistic), float(result.pvalue), ""
            else:
                rho, p_raw, reason = float("nan"), float("nan"), "constant_input"
            correlation_rows.append(
                {
                    "dataset": dataset,
                    "x": "quality_q",
                    "y": endpoint,
                    "n_runs": len(group),
                    "correlation_defined": defined,
                    "undefined_reason": reason,
                    "spearman_rho": rho,
                    "p_raw": p_raw,
                }
            )

    descriptive = pd.DataFrame(descriptive_rows)
    correlations = pd.DataFrame(correlation_rows)
    correlations["p_for_holm"] = correlations["p_raw"].where(correlations["p_raw"].notna(), 1.0)
    correlations["p_holm_across_nine"] = holm_adjust(
        correlations["p_for_holm"].to_numpy(dtype=float)
    )
    correlations["reject_0_05_holm"] = correlations["correlation_defined"] & correlations[
        "p_holm_across_nine"
    ].le(0.05)
    descriptive.to_csv(output / "endpoint_descriptive_statistics.csv", index=False)
    correlations.to_csv(output / "quality_external_correlations.csv", index=False)
    pair_summary = (
        pairs.groupby("dataset")["partition_ari_all_events"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    pair_summary["pairs"] = 435
    pair_summary.to_csv(output / "pairwise_partition_ari_summary.csv", index=False)
    check(
        "experiment",
        "descriptive_family",
        len(descriptive) == 36 and descriptive.groupby("dataset").size().eq(12).all(),
        f"rows={len(descriptive)}",
    )
    check(
        "experiment",
        "correlation_family",
        len(correlations) == 9 and correlations.groupby("dataset").size().eq(3).all(),
        f"rows={len(correlations)}",
    )
    check(
        "experiment",
        "metric_ranges",
        points["ari"].between(-1, 1).all()
        and points["macro_f1"].between(0, 1).all()
        and points["hungarian_accuracy"].between(0, 1).all()
        and pairs["partition_ari_all_events"].between(-1, 1).all(),
        "all legal",
    )
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(output / "analysis_checks.csv", index=False)
    manifest = {
        "experiment_id": "EXP-045D",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_links": source_links,
        "endpoints": ENDPOINTS,
        "correlation_endpoints": CORRELATION_ENDPOINTS,
        "repeat_not_seed": True,
        "evaluation_scope": "all_events_fit_evaluable_reference_labels_only",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "checks_passed": int(check_frame["passed"].sum()),
        "checks_total": len(check_frame),
        "all_checks_passed": bool(check_frame["passed"].all()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    write_json(output / "run_manifest.json", manifest)
    summary_lines = [
        "# EXP-045D PhenoGraph multiclass full-event summary",
        "",
        "The analysis includes 90 parent runs, 1,305 full-event partition pairs, 36 descriptive endpoints, and 9 associations between modularity Q and external metrics.",
        "",
    ]
    for row in pair_summary.itertuples(index=False):
        summary_lines.append(
            f"- {row.dataset}: pairwise ARI mean={row.mean:.6f}, range=[{row.min:.6f}, {row.max:.6f}]."
        )
    summary_lines.extend(
        [
            "",
            f"Checks passed: {manifest['checks_passed']}/{manifest['checks_total']}. Repeats are not controlled seeds, and no best repeat was selected.",
            "",
        ]
    )
    (output / "scientific_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    write_json(
        output / "artifact_hashes.json",
        {
            path.name: sha256(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
