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
from analyze_flowsom_stability30 import holm_adjust

SOURCES = {
    "FlowSOM": (
        "EXP-021C_flowsom_stability30_distribution_analysis_20260910",
        "controlled_seed_0_29",
        "all_events",
        "fixed_grid10_rlen30_metaclusters40",
    ),
    "GMM": (
        "EXP-022C_gmm_stability30_distribution_analysis_20260910",
        "controlled_seed_0_29",
        "all_events",
        "fixed_40_components_full_covariance",
    ),
    "Leiden_fixed_PhenoGraph_graph": (
        "EXP-024C_phenograph_leiden_stability30_distribution_analysis_20260910",
        "controlled_seed_0_29",
        "fixed_20000_training_events_only",
        "fixed_k15_jaccard_graph_resolution1",
    ),
    "PhenoGraph_default_Louvain": (
        "EXP-025C_phenograph_louvain_stability30_distribution_analysis_20260910",
        "uncontrolled_native_repeat_0_29",
        "fixed_20000_training_events_only",
        "fixed_k15_jaccard_graph_default_louvain",
    ),
    "KMeans": (
        "EXP-026_kmeans_uniform_stability_distribution_analysis_20260910",
        "controlled_seed_0_29",
        "all_events",
        "K_equals_true_population_count_sensitivity_only",
    ),
}
OBJECTIVES = {
    "FlowSOM": "median_bmu_distance",
    "GMM": "lower_bound",
    "Leiden_fixed_PhenoGraph_graph": "quality_q",
    "PhenoGraph_default_Louvain": "quality_q",
    "KMeans": "inertia_per_fit_cell",
}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, protocol, output = (
        args.runs_root.resolve(),
        args.protocol.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    source_links = []
    summary_rows = []
    test_frames = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    source_failures = []
    for algorithm, (name, repeat_type, scope, regime) in SOURCES.items():
        run = root / name
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        failures = inventory_failures(run / "artifact_hashes.json")
        source_failures.extend([f"{algorithm}:{x}" for x in failures])
        source_links.append(
            {
                "algorithm": algorithm,
                "run": str(run),
                "run_manifest_sha256": sha256(run / "run_manifest.json"),
                "artifact_hashes_sha256": sha256(run / "artifact_hashes.json"),
            }
        )
        desc = pd.read_csv(run / "endpoint_descriptive_statistics.csv")
        pairs = pd.read_csv(run / "pairwise_partition_ari_summary.csv").set_index("dataset")
        objective = OBJECTIVES[algorithm]
        for dataset in ["Levine_32dim", "Samusik_01"]:
            values = desc.loc[desc.dataset.eq(dataset)].set_index("endpoint")
            row = {
                "algorithm": algorithm,
                "dataset": dataset,
                "repeat_type": repeat_type,
                "n_runs": 30,
                "evaluation_scope": scope,
                "parameter_regime": regime,
                "objective": objective,
            }
            for endpoint, prefix in [
                ("ari", "ari"),
                ("macro_f1", "macro_f1"),
                ("hungarian_accuracy", "accuracy"),
                (objective, "objective"),
            ]:
                for stat in [
                    "mean",
                    "sd",
                    "min",
                    "max",
                    "bootstrap_mean_ci95_low",
                    "bootstrap_mean_ci95_high",
                ]:
                    row[f"{prefix}_{stat}"] = float(values.loc[endpoint, stat])
            row.update(
                {
                    "pairwise_partition_ari_mean": float(pairs.loc[dataset, "mean"]),
                    "pairwise_partition_ari_min": float(pairs.loc[dataset, "min"]),
                    "pairwise_partition_ari_max": float(pairs.loc[dataset, "max"]),
                    "pair_count": int(pairs.loc[dataset, "pairs"]),
                }
            )
            summary_rows.append(row)
        tests = pd.read_csv(run / "uniform_multimodality_tests.csv")
        tests.insert(0, "algorithm", algorithm)
        tests.insert(1, "source_run", name)
        test_frames.append(tests)
    check("source_analyses", not source_failures, f"failures={len(source_failures)}")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "harmonized_stability_evidence.csv", index=False)
    check(
        "harmonized_rows",
        len(summary) == 10
        and summary.groupby("algorithm").size().eq(2).all()
        and summary["pair_count"].eq(435).all(),
        f"rows={len(summary)}",
    )
    tests = pd.concat(test_frames, ignore_index=True)
    tests["empirical_p_holm_global_40"] = holm_adjust(tests["empirical_p_raw"].to_numpy(float))
    tests["reject_single_gaussian_global_0_05"] = tests["empirical_p_holm_global_40"] <= 0.05
    tests.to_csv(output / "global_40_endpoint_multimodality_tests.csv", index=False)
    check(
        "global_test_family",
        len(tests) == 40 and tests.groupby("algorithm").size().eq(8).all(),
        f"rows={len(tests)}; rejects={int(tests['reject_single_gaussian_global_0_05'].sum())}",
    )
    bl = json.loads(
        (root / "EXP-006_blflowsom_levine13_nodes_20260910" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    xs = json.loads(
        (root / "EXP-009B_xshift_repeat_comparison_20260910" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    repeat = pd.read_csv(
        root / "EXP-009B_xshift_repeat_comparison_20260910" / "repeat_comparison.csv"
    ).iloc[0]
    deterministic = pd.DataFrame(
        [
            {
                "algorithm": "BL-FlowSOM_Sony_PoC",
                "dataset": "Levine_13dim",
                "n_repeats": 2,
                "repeat_labels": "nominal_seed_1_2",
                "seed_control": "parameter_does_not_enter_current_training_path",
                "method_stage": "SOM_nodes_only_no_R_metaclustering",
                "labels_or_nodes_exact": bool(bl["seed_1_and_2_clusters_identical"]),
                "codes_exact": bool(bl["seed_1_and_2_codes_identical"]),
                "partition_ari": 1.0,
                "distribution_test_eligible": False,
            },
            {
                "algorithm": "X-shift_official",
                "dataset": "Levine_13dim",
                "n_repeats": 2,
                "repeat_labels": "same_fixed_K20_condition",
                "seed_control": "no_exposed_seed_in_used_CLI",
                "method_stage": "full_event_clustering",
                "labels_or_nodes_exact": bool(
                    np.isclose(float(repeat["raw_label_exact_fraction"]), 1.0)
                ),
                "codes_exact": np.nan,
                "partition_ari": float(repeat["partition_ari"]),
                "distribution_test_eligible": False,
            },
        ]
    )
    deterministic.to_csv(output / "deterministic_and_interface_limited_evidence.csv", index=False)
    check(
        "determinism_evidence",
        bl.get("all_structural_checks_passed") is True
        and xs.get("all_integrity_checks_passed") is True
        and deterministic["labels_or_nodes_exact"].all()
        and deterministic["partition_ari"].eq(1.0).all(),
        deterministic.to_json(orient="records"),
    )
    scope_counts = summary.groupby("evaluation_scope").size().to_dict()
    check(
        "scope_separation",
        scope_counts == {"all_events": 6, "fixed_20000_training_events_only": 4},
        json.dumps(scope_counts),
    )
    frame = pd.DataFrame(checks)
    frame.to_csv(output / "audit_checks.csv", index=False)
    manifest = {
        "experiment_id": "EXP-027",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_links": source_links,
        "global_test_family_size": 40,
        "global_rejections": int(tests["reject_single_gaussian_global_0_05"].sum()),
        "checks_passed": int(frame["passed"].sum()),
        "checks_total": len(frame),
        "all_checks_passed": bool(frame["passed"].all()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rejected = tests.loc[
        tests["reject_single_gaussian_global_0_05"],
        ["algorithm", "dataset", "endpoint", "empirical_p_holm_global_40"],
    ].to_dict(orient="records")
    (output / "scientific_summary.md").write_text(
        f"# EXP-027 Cross-method stability evidence audit\n\nThe global Holm family contains 40 tests across five methods, two datasets, and four endpoints. Rejections: {len(rejected)}; details: {json.dumps(rejected, ensure_ascii=False)}. The n=2 deterministic evidence for BL-FlowSOM and X-shift is reported separately and is not included in distributional tests.\n",
        encoding="utf-8",
    )
    artifacts = [p for p in output.iterdir() if p.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps({str(p): sha256(p) for p in artifacts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
