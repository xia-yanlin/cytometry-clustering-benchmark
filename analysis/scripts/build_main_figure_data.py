"""Build exact panel-data CSVs for the three planned main figures.

No plotting library is required. Rows retain their experimental unit and source ID.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
DEST = PACKAGE / "figure_data"


def read(source_id: str) -> list[dict[str, str]]:
    with (PACKAGE / "inputs" / f"{source_id}.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write(name: str, records: list[dict], columns: list[str], expected: int) -> None:
    if len(records) != expected:
        raise RuntimeError(f"{name}: expected {expected} rows, found {len(records)}")
    DEST.mkdir(exist_ok=True)
    with (DEST / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    simulation = [
        {
            "background_clusters": r["background_clusters"],
            "ari": r["binary_ari_using_raw_clusters"],
            "target_f1": r["target_f1"],
            "unit": "simulation_condition",
            "source_id": "metric_simulation",
        }
        for r in read("metric_simulation")
    ]
    write(
        "figure1a_reference_simulation.csv",
        simulation,
        ["background_clusters", "ari", "target_f1", "unit", "source_id"],
        7,
    )

    rare = []
    for r in read("gmm_rare_runs"):
        rare.append(
            {
                "dataset": r["dataset"],
                "method": "GMM",
                "regime": f"K={r['k']}",
                "label_informed_regime": r["label_information_used_for_regime_definition"],
                "repeat_type": "controlled_seed",
                "repeat_id": r["seed"],
                "ari": r["ari"],
                "precision": r["target_precision"],
                "recall": r["target_recall"],
                "f1": r["target_f1"],
                "f2": r["target_f2"],
                "source_id": "gmm_rare_runs",
            }
        )
    for r in read("xshift_k_sensitivity"):
        if r["dataset"] in ("Nilsson_rare", "Mosmann_rare") and r["knn_k"] == "20":
            rare.append(
                {
                    "dataset": r["dataset"],
                    "method": "official_Xshift",
                    "regime": "neighbor_K=20",
                    "label_informed_regime": "False",
                    "repeat_type": "single_fixed_partition",
                    "repeat_id": "0",
                    "ari": r["ari"],
                    "precision": r["target_precision"],
                    "recall": r["target_recall"],
                    "f1": r["target_f1"],
                    "f2": r["target_f2"],
                    "source_id": "xshift_k_sensitivity",
                }
            )
    columns = [
        "dataset",
        "method",
        "regime",
        "label_informed_regime",
        "repeat_type",
        "repeat_id",
        "ari",
        "precision",
        "recall",
        "f1",
        "f2",
        "source_id",
    ]
    gmm_cells = Counter((r["dataset"], r["regime"]) for r in rare if r["method"] == "GMM")
    if len(gmm_cells) != 6 or set(gmm_cells.values()) != {30}:
        raise RuntimeError(f"GMM rare-cell seed coverage changed: {gmm_cells}")
    write("figure1b_rare_run_points.csv", rare, columns, 182)

    population = [
        {
            key: r[key]
            for key in (
                "population",
                "support",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "overlapping_predicted_clusters",
                "effective_predicted_clusters",
            )
        }
        | {
            "dataset": "Levine_32dim",
            "method": "official_Xshift_K20",
            "unit": "reference_population",
            "source_id": "xshift_levine32_population",
        }
        for r in read("xshift_levine32_population")
    ]
    write(
        "figure1c_multiclass_population.csv",
        population,
        [
            "dataset",
            "method",
            "population",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "overlapping_predicted_clusters",
            "effective_predicted_clusters",
            "unit",
            "source_id",
        ],
        14,
    )

    inclusion = [
        {
            "dataset": "Levine_13dim",
            "condition": r["condition"],
            "fit_events": r["fit_events"],
            "ari": r["ari"],
            "macro_f1": r["macro_f1"],
            "reference_k_used": "24",
            "unit": "one_fixed_seed",
            "source_id": "flowsom_event_inclusion",
        }
        for r in read("flowsom_event_inclusion")
        if r["condition"] in ("labeled_direct_a", "all_direct")
    ]
    write(
        "figure2a_event_inclusion.csv",
        inclusion,
        [
            "dataset",
            "condition",
            "fit_events",
            "ari",
            "macro_f1",
            "reference_k_used",
            "unit",
            "source_id",
        ],
        2,
    )

    auto = [
        {
            "dataset": r["dataset"],
            "som_seed": r["som_seed"],
            "regime": r["regime"],
            "reference_informed_k": str(r["regime"] == "official_fixed_ktrue"),
            "selected_k": r["selected_k"],
            "ari": r["ari"],
            "macro_f1": r["macro_f1"],
            "unit": "paired_SOM_seed",
            "source_id": "flowsom_auto_multiclass_runs",
        }
        for r in read("flowsom_auto_multiclass_runs")
        if r["regime"] in ("official_auto_max40", "official_fixed_k40", "official_fixed_ktrue")
    ]
    auto_cells = Counter((r["dataset"], r["regime"]) for r in auto)
    if len(auto_cells) != 6 or set(auto_cells.values()) != {30}:
        raise RuntimeError(f"FlowSOM automatic-regime seed coverage changed: {auto_cells}")
    write(
        "figure2b_flowsom_metacluster_regimes.csv",
        auto,
        [
            "dataset",
            "som_seed",
            "regime",
            "reference_informed_k",
            "selected_k",
            "ari",
            "macro_f1",
            "unit",
            "source_id",
        ],
        180,
    )

    xshift_nilsson = []
    fixed = [
        r
        for r in read("xshift_k_sensitivity")
        if r["dataset"] == "Nilsson_rare" and r["knn_k"] == "20"
    ]
    if len(fixed) != 1:
        raise RuntimeError("Expected one Nilsson fixed K=20 X-shift row")
    for regime, r, source in (
        ("fixed_neighbor_K20", fixed[0], "xshift_k_sensitivity"),
        (
            "published_neighbor_K60",
            read("xshift_nilsson_published_k60")[0],
            "xshift_nilsson_published_k60",
        ),
        ("official_automatic_elbow", read("xshift_nilsson_auto")[0], "xshift_nilsson_auto"),
    ):
        xshift_nilsson.append(
            {
                "dataset": "Nilsson_rare",
                "regime": regime,
                "ari": r["ari"],
                "precision": r["target_precision"],
                "recall": r["target_recall"],
                "f1": r["target_f1"],
                "f2": r["target_f2"],
                "unit": "one_fixed_partition",
                "source_id": source,
            }
        )
    write(
        "figure2c_xshift_nilsson_regimes.csv",
        xshift_nilsson,
        ["dataset", "regime", "ari", "precision", "recall", "f1", "f2", "unit", "source_id"],
        3,
    )

    flowsom_runs = [
        {
            "dataset": r["dataset"],
            "method": "FlowSOM_R_2.18.0_full_event",
            "repeat_type": "controlled_seed",
            "repeat_id": r["seed"],
            "ari": r["ari"],
            "macro_f1": r["macro_f1"],
            "unit": "algorithm_run",
            "source_id": "flowsom_r_full_run_points",
        }
        for r in read("flowsom_r_full_run_points")
        if r["scheme"] == "R_full_event"
    ]
    pheno_runs = [
        {
            "dataset": r["dataset"],
            "method": "PhenoGraph_v1.5.7_default_Louvain",
            "repeat_type": "native_repeat",
            "repeat_id": r["repeat"],
            "ari": r["ari"],
            "macro_f1": r["macro_f1"],
            "unit": "algorithm_run",
            "source_id": "phenograph_full_run_points",
        }
        for r in read("phenograph_full_run_points")
    ]
    run_cols = [
        "dataset",
        "method",
        "repeat_type",
        "repeat_id",
        "ari",
        "macro_f1",
        "unit",
        "source_id",
    ]
    if set(Counter(r["dataset"] for r in flowsom_runs).values()) != {30}:
        raise RuntimeError("FlowSOM full-event run coverage changed")
    if set(Counter(r["dataset"] for r in pheno_runs).values()) != {30}:
        raise RuntimeError("PhenoGraph full-event repeat coverage changed")
    write("figure3a_flowsom_r_run_points.csv", flowsom_runs, run_cols, 60)
    write("figure3b_phenograph_run_points.csv", pheno_runs, run_cols, 90)

    partition = []
    for r in read("flowsom_r_full_partition"):
        if r["scheme"] == "R_full_event":
            partition.append(
                {
                    "dataset": r["dataset"],
                    "method": "FlowSOM_R_2.18.0_full_event",
                    "repeat_type": "controlled_seed",
                    "n_runs": 30,
                    "pair_count": r["count"],
                    "mean_pairwise_ari": r["mean"],
                    "min_pairwise_ari": r["min"],
                    "max_pairwise_ari": r["max"],
                    "source_id": "flowsom_r_full_partition",
                }
            )
    for r in read("phenograph_full_partition"):
        partition.append(
            {
                "dataset": r["dataset"],
                "method": "PhenoGraph_v1.5.7_default_Louvain",
                "repeat_type": "native_repeat",
                "n_runs": 30,
                "pair_count": r["pairs"],
                "mean_pairwise_ari": r["mean"],
                "min_pairwise_ari": r["min"],
                "max_pairwise_ari": r["max"],
                "source_id": "phenograph_full_partition",
            }
        )
    for dataset, source in (
        ("Nilsson_rare", "xshift_nilsson_repeat_pairs"),
        ("Samusik_01", "xshift_samusik_repeat_pairs"),
        ("Levine_13dim", "xshift_levine13_repeat_pairs"),
        ("Levine_32dim", "xshift_levine32_repeat_pairs"),
        ("Mosmann_rare", "xshift_mosmann_repeat_pairs"),
    ):
        values = [float(r["partition_ari"]) for r in read(source)]
        if len(values) != 435 or any(abs(v - 1.0) > 1e-12 for v in values):
            raise RuntimeError(f"Unexpected X-shift repeat partitions for {dataset}")
        partition.append(
            {
                "dataset": dataset,
                "method": "official_Xshift_fixed_neighbor_K20",
                "repeat_type": "native_repeat",
                "n_runs": 30,
                "pair_count": 435,
                "mean_pairwise_ari": 1.0,
                "min_pairwise_ari": 1.0,
                "max_pairwise_ari": 1.0,
                "source_id": source,
            }
        )
    write(
        "figure3c_within_method_partition_similarity.csv",
        partition,
        [
            "dataset",
            "method",
            "repeat_type",
            "n_runs",
            "pair_count",
            "mean_pairwise_ari",
            "min_pairwise_ari",
            "max_pairwise_ari",
            "source_id",
        ],
        10,
    )
    print("Built nine panel-data CSVs for Figures 1-3")


if __name__ == "__main__":
    main()
