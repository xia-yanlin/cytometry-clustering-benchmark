"""Build the main and supplementary tables from the analysis inputs."""

import json
import shutil
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "inputs"
FIG = ROOT / "figure_data"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)


def table1():
    d = pd.read_csv(IN / "dataset_metadata.csv")
    required = {
        "Dataset",
        "ExperimentHub",
        "Total events",
        "Evaluable events",
        "Markers",
        "Reference populations",
        "Reference contract",
        "Analysis transform",
        "Fit-event policy",
    }
    if set(d.columns) != required:
        raise ValueError("dataset_metadata.csv does not have the expected columns")
    expected_ids = ["EH2242", "EH2240", "EH2244", "EH2248", "EH2250"]
    if d["ExperimentHub"].tolist() != expected_ids:
        raise ValueError("Unexpected ExperimentHub accessions in dataset_metadata.csv")
    assert d["Total events"].tolist() == [167044, 265627, 86864, 44140, 396460]
    return d


def table2():
    rows = [
        (
            "Official X-shift",
            "Nolan Vortex standalone.Xshift 2017 rev2",
            "All events",
            "Fixed neighbor K=20; named K=60/auto contrasts",
            "Labels only for evaluation",
            "30 native repeats in stability panel",
            "Main F1–F3; not a common algorithm ranking",
        ),
        (
            "R FlowSOM",
            "FlowSOM 2.18.0",
            "All events in current stability analysis",
            "Fixed SOM/metaclustering regime",
            "Labels only for evaluation",
            "30 controlled seeds",
            "Main F3; two multiclass datasets",
        ),
        (
            "FlowSOM metaclustering contrast",
            "FlowSOM R, frozen SOM nodes",
            "Same SOM events within paired contrasts",
            "Official auto vs fixed 40; K=true sensitivity",
            "K=true defined using reference count",
            "30 paired SOM seeds",
            "Main F2; regime comparison only",
        ),
        (
            "Official PhenoGraph",
            "PhenoGraph 1.5.7 default Louvain",
            "All events in current run panel",
            "Fixed neighbor parameter per dataset",
            "Labels only for evaluation",
            "30 native repeats; seed not exposed",
            "Main F3; three multiclass datasets",
        ),
        (
            "GMM",
            "scikit-learn GaussianMixture 1.6.1",
            "20,000-event training; full-event prediction",
            "K=2/10/40 sensitivity",
            "K=2 uses reference count; K=10/40 fixed; labels excluded from fit",
            "30 controlled seeds per K",
            "Main F1; no unified rank",
        ),
        (
            "Official Deterministic-SPADE",
            "Qiu Lab MATLAB implementation",
            "Nilsson full-event endpoint only",
            "K=40 endpoint",
            "Labels only for post hoc evaluation",
            "Two accepted attempts, same partition",
            "Supplement only; one dataset",
        ),
        (
            "Sony PoC + R metaclustering",
            "Sony batch SOM PoC + Sony R MetaClust",
            "Levine 13dim endpoint",
            "Named K regime",
            "K=true endpoint is label-informed",
            "PoC seed did not change nodes in tested path",
            "Supplement only; no cloud-equivalence claim",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Method in manuscript",
            "Executed identity",
            "Fit scope",
            "K/parameter regime",
            "Label access",
            "Repeat interface",
            "Permitted role",
        ],
    )


def s1(t1):
    numeric = pd.read_csv(IN / "other4_identity.csv").set_index("dataset")
    lev = pd.read_csv(IN / "levine13_identity.csv").set_index("metric")["value"]
    rows = []
    for _, r in t1.iterrows():
        ds = r["Dataset"]
        max_delta = (
            float(lev["max_abs_delta_transformed"])
            if ds == "Levine_13dim"
            else float(numeric.loc[ds, "max_abs_delta"])
        )
        rows.append(
            {
                "Dataset": ds,
                "Resource": r["ExperimentHub"],
                "Events": r["Total events"],
                "Markers": r["Markers"],
                "Evaluable events": r["Evaluable events"],
                "Transform": r["Analysis transform"],
                "Max verified numeric delta": max_delta,
                "Label mismatches": 0,
            }
        )
    return pd.DataFrame(rows)


def s2():
    methods = pd.read_csv(IN / "method_identity.csv")
    methods = methods[~methods.method_label.str.startswith("ARCHIVED")]
    a = methods[
        [
            "method_label",
            "actual_execution_identity",
            "commit_or_version",
            "seed_interface",
            "label_information",
            "verified_endpoint_scope",
            "prohibited_inference",
        ]
    ].copy()
    a.columns = [
        "Method",
        "Executed identity",
        "Version/commit",
        "Random interface",
        "Label permission",
        "Verified scope",
        "Inference boundary",
    ]
    sp = a.Method.eq("official_Deterministic-SPADE")
    a.loc[sp, "Verified scope"] = (
        "Nilsson_rare full-event K=40 endpoint; EXP-044-R4 accepted attempts 3 and 4"
    )
    a.loc[sp, "Label permission"] = (
        "Labels excluded from fit/K selection; post hoc target matching only"
    )
    a.loc[sp, "Inference boundary"] = "One rare dataset; not a five-dataset method comparison"
    xs = a.Method.eq("official_X-shift")
    a.loc[xs, "Verified scope"] = (
        "Five datasets, full-event K=20 ×30 native repeats each; "
        "K=10/20/40 prespecified sensitivity"
    )
    a.loc[xs, "Inference boundary"] = (
        "Native repeats have no controlled seed; no common cross-method ranking"
    )
    pg = a.Method.eq("PhenoGraph_default_Louvain")
    a.loc[pg, "Verified scope"] = (
        "Three multiclass full-event datasets ×30 native Louvain repeats; "
        "two rare default endpoints separately"
    )
    a.loc[pg, "Inference boundary"] = (
        "Native Louvain repeats are not controlled seeds or fixed-graph Leiden"
    )
    a = pd.concat(
        [
            a,
            pd.DataFrame(
                [
                    {
                        "Method": "FlowSOM_R_2.18.0_full_event",
                        "Executed identity": "Bioconductor FlowSOM R 2.18.0 full-event fit",
                        "Version/commit": "2.18.0; EXP-046C-R2",
                        "Random interface": "Controlled seed 0–29",
                        "Label permission": "Labels excluded from fit; post hoc external evaluation",
                        "Verified scope": "Levine_32dim and Samusik_01, 30 runs each",
                        "Inference boundary": "Protocol-specific variability; not direct ranking against other method regimes",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    regimes = pd.read_csv(IN / "parameter_regimes.csv")
    b = regimes[
        [
            "record_id",
            "method_family",
            "datasets",
            "regime",
            "parameter_space",
            "training_policy",
            "label_permission",
            "selection_objective",
            "gap",
        ]
    ].copy()
    b.columns = [
        "Record",
        "Method",
        "Datasets",
        "Regime",
        "Parameter space",
        "Training",
        "Label access",
        "Selection objective",
        "Gap",
    ]
    b.loc[b.Record.eq("R01"), "Gap"] = (
        "Historical fixed endpoint; current five-dataset K=20 ×30 native repeats are R22–R26"
    )
    b.loc[b.Record.eq("R09"), "Gap"] = (
        "Historical K=true sensitivity; see R20–R21 for unlabeled fixed/automatic regimes; no cloud equivalence"
    )
    b.loc[b.Record.eq("R13"), "Gap"] = (
        "Historical 20k graph; current all-event default-Louvain repeats are R28; no controlled seed"
    )
    # Add the later EXP-042–046 runs to the earlier parameter registry.
    sp = b.Record.eq("R19")
    b.loc[
        sp,
        [
            "Datasets",
            "Regime",
            "Parameter space",
            "Training",
            "Label access",
            "Selection objective",
            "Gap",
        ],
    ] = [
        "Nilsson_rare",
        "main_no_label_fixed",
        "official Deterministic-SPADE; K=40",
        "all 44,140 events; two accepted full-pipeline attempts",
        "labels excluded from fit and K selection; target matched post hoc",
        "none; K fixed before evaluation",
        "EXP-044-R4: validated Nilsson endpoint only; two partitions identical; no five-dataset comparison",
    ]
    for record in ("R22", "R23", "R24", "R25"):
        b.loc[b.Record.eq(record), "Gap"] = (
            "EXP-042-R1 completed the five-dataset K=20 native-repeat set; "
            "30 empirical repeats per dataset, no controlled seed interface"
        )
    accepted_addendum = pd.DataFrame(
        [
            {
                "Record": "R26",
                "Method": "official_X-shift",
                "Datasets": "Mosmann_rare",
                "Regime": "main_no_label_fixed",
                "Parameter space": "K=20 fixed; 30 native Java-process repeats",
                "Training": "all 396,460 events in each repeat",
                "Label access": "labels excluded from clustering; rare target matched post hoc",
                "Selection objective": "none; fixed before evaluation",
                "Gap": "EXP-042-R1/V: 30/30 accepted, one empirical partition; native repeats are not controlled seeds",
            },
            {
                "Record": "R27",
                "Method": "official_X-shift",
                "Datasets": "all five datasets",
                "Regime": "fixed_K_sensitivity",
                "Parameter space": "neighbor K=10/20/40; 15 named dataset-K conditions",
                "Training": "all events; K=20 reuses accepted partition; 10 new K=10/40 runs",
                "Label access": "labels excluded from clustering and K choice; post hoc evaluation",
                "Selection objective": "none; prespecified scan, not outcome-selected best K",
                "Gap": "EXP-043-R1/V: one partition per condition; no repeat uncertainty for K=10/40",
            },
            {
                "Record": "R28",
                "Method": "PhenoGraph_default_Louvain",
                "Datasets": "Samusik_01;Levine_13dim;Levine_32dim",
                "Regime": "main_no_label_fixed",
                "Parameter space": "official v1.5.7; Euclidean k=30 Jaccard graph; default Louvain ×30",
                "Training": "all events per dataset; labels used only after partition freeze",
                "Label access": "no labels in neighbor, graph or Louvain steps",
                "Selection objective": "none; default graph and community settings",
                "Gap": "EXP-045A–D/V: 90 accepted runs; Louvain native repeats lack controlled seed",
            },
            {
                "Record": "R29",
                "Method": "FlowSOM_R_2.18.0_full_event",
                "Datasets": "Samusik_01;Levine_32dim",
                "Regime": "main_no_label_fixed",
                "Parameter space": "xdim=10; ydim=10; rlen=30; nClus=40; seed=0–29",
                "Training": "all 86,864 or 265,627 events, respectively",
                "Label access": "labels excluded from R fit; external evaluation after partition freeze",
                "Selection objective": "none; K=40 fixed before evaluation",
                "Gap": "EXP-046A–C/V: implementation and fit scope both differ from earlier Python 20k scheme",
            },
        ],
        columns=b.columns,
    )
    b = pd.concat([b, accepted_addendum], ignore_index=True)
    return a, b


def s3():
    d = pd.read_csv(IN / "xshift_levine32_population.csv")
    cols = [
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
    ]
    x = d[cols].copy()
    x.columns = [
        "Population",
        "Reference events",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "Overlapping predicted clusters",
        "Effective predicted clusters",
    ]
    return x.sort_values("Reference events", ascending=False)


def parse_spade_numeric():
    artifact = json.loads((IN / "spade_nilsson_evaluation.json").read_text(encoding="utf-8"))
    attempts = artifact["attempts"]
    if len(attempts) != 2 or {a["attempt"] for a in attempts} != {3, 4}:
        raise ValueError("Expected the two accepted SPADE attempts 3 and 4")
    first = next(a for a in attempts if a["attempt"] == 3)
    second = next(a for a in attempts if a["attempt"] == 4)
    fields = [
        "ari_two_class_reference_vs_40_clusters",
        "target_precision",
        "target_recall",
        "target_f1",
    ]
    for field in fields:
        if first[field] != second[field]:
            raise ValueError(f"SPADE accepted attempts disagree for {field}")
    return {field: float(first[field]) for field in fields}


def s4():
    points = pd.read_csv(FIG / "figure1b_rare_run_points.csv")
    rows = []
    for (ds, method, regime), g in points.groupby(["dataset", "method", "regime"]):
        rows.append(
            {
                "Dataset": ds,
                "Method": method,
                "Regime": regime,
                "n runs": len(g),
                "ARI mean": g.ari.mean(),
                "Precision mean": g.precision.mean(),
                "Recall mean": g.recall.mean(),
                "F1 mean": g.f1.mean(),
                "Repeat type": g.repeat_type.iloc[0],
                "Scope": (
                    "Reference-count K; labels excluded from fit; post hoc target matching"
                    if regime == "K=2"
                    else "Fixed K sensitivity; labels excluded from fit; post hoc target matching"
                )
                if method == "GMM"
                else "Unlabeled fixed regime; post hoc target matching",
            }
        )
    f = pd.read_csv(IN / "flowsom_rare_regimes.csv")
    for (ds, regime), g in f.groupby(["dataset", "regime"]):
        if regime not in ["official_auto_max40", "official_fixed_k40", "official_fixed_ktrue_2"]:
            continue
        m = g.set_index("metric")
        rows.append(
            {
                "Dataset": ds,
                "Method": "R FlowSOM",
                "Regime": regime,
                "n runs": int(m.loc["ari", "n"]),
                "ARI mean": m.loc["ari", "mean"],
                "Precision mean": m.loc["target_precision", "mean"],
                "Recall mean": m.loc["target_recall", "mean"],
                "F1 mean": m.loc["target_f1", "mean"],
                "Repeat type": "paired SOM seed",
                "Scope": "K=true is label-informed sensitivity"
                if "ktrue" in regime
                else "Named regime",
            }
        )
    p = pd.read_csv(IN / "phenograph_rare_summary.csv")
    for ds, g in p.groupby("dataset"):
        m = g.set_index("endpoint")
        rows.append(
            {
                "Dataset": ds,
                "Method": "PhenoGraph",
                "Regime": "default Louvain",
                "n runs": int(m.loc["ari", "n_runs"]),
                "ARI mean": m.loc["ari", "mean"],
                "Precision mean": m.loc["target_precision", "mean"],
                "Recall mean": m.loc["target_recall", "mean"],
                "F1 mean": m.loc["target_f1", "mean"],
                "Repeat type": "native repeat",
                "Scope": "Fixed graph/parameter regime",
            }
        )
    s = parse_spade_numeric()
    rows.append(
        {
            "Dataset": "Nilsson_rare",
            "Method": "Deterministic-SPADE",
            "Regime": "official K=40",
            "n runs": 2,
            "ARI mean": s["ari_two_class_reference_vs_40_clusters"],
            "Precision mean": s["target_precision"],
            "Recall mean": s["target_recall"],
            "F1 mean": s["target_f1"],
            "Repeat type": "accepted attempts 3/4, same partition",
            "Scope": "One dataset only",
        }
    )
    return pd.DataFrame(rows).sort_values(["Dataset", "Method", "Regime"])


def s5():
    older = pd.read_csv(IN / "stability_40cells.csv")
    rows = []
    for _, r in older.iterrows():
        rows.append(
            {
                "Evidence block": "Earlier fixed-regime family",
                "Dataset": r.dataset,
                "Method": r.algorithm,
                "Repeat type": r.repeat_type,
                "n runs": r.n_runs,
                "ARI mean": r.ari_mean,
                "ARI min": r.ari_min,
                "ARI max": r.ari_max,
                "Macro F1 mean": r.macro_f1_mean,
                "Pairwise partition ARI mean": r.pairwise_partition_ari_mean,
            }
        )
    c = pd.read_csv(FIG / "figure3c_within_method_partition_similarity.csv")
    a = pd.read_csv(FIG / "figure3a_flowsom_r_run_points.csv")
    b = pd.read_csv(FIG / "figure3b_phenograph_run_points.csv")
    runs = pd.concat([a, b], ignore_index=True)
    for _, r in c.iterrows():
        sub = runs[runs.dataset.eq(r.dataset) & runs.method.eq(r.method)]
        rows.append(
            {
                "Evidence block": "Current full-event/native-repeat result",
                "Dataset": r.dataset,
                "Method": r.method,
                "Repeat type": r.repeat_type,
                "n runs": r.n_runs,
                "ARI mean": sub.ari.mean() if len(sub) else None,
                "ARI min": sub.ari.min() if len(sub) else None,
                "ARI max": sub.ari.max() if len(sub) else None,
                "Macro F1 mean": sub.macro_f1.mean() if len(sub) else None,
                "Pairwise partition ARI mean": r.mean_pairwise_ari,
            }
        )
    return pd.DataFrame(rows)


def s5_tests_and_power():
    tests = pd.read_csv(IN / "stability_40tests.csv")
    tests = tests[
        [
            "algorithm",
            "dataset",
            "endpoint",
            "n_runs",
            "empirical_p_raw",
            "empirical_p_holm_global_40",
            "reject_single_gaussian_global_0_05",
        ]
    ]
    tests.columns = [
        "Method",
        "Dataset",
        "Endpoint",
        "n runs",
        "Raw empirical p",
        "Global Holm adjusted p",
        "Reject single Gaussian",
    ]
    power = pd.read_csv(IN / "multimodality_power.csv")
    power = power[
        ["n", "mean_separation_sd_units", "simulations", "independent_power", "original_power"]
    ]
    power.columns = [
        "Runs per simulation",
        "Mean separation (SD units)",
        "Simulations",
        "Independent rejection rate",
        "Original rejection rate",
    ]
    return tests, power


def s6():
    grid = pd.read_csv(IN / "flowsom_pareto.csv")
    grid = grid[
        [
            "dataset",
            "grid_side",
            "rlen",
            "median_bmu_distance_mean",
            "ari_mean",
            "macro_f1_mean",
            "pareto_front_mean",
        ]
    ]
    grid.columns = [
        "Dataset",
        "Grid side",
        "rlen",
        "Mean median BMU distance",
        "Mean ARI",
        "Mean Macro F1",
        "3-objective Pareto",
    ]
    label = pd.read_csv(IN / "label_perturbation.csv")
    label = label[label.kind.ne("unchanged")][
        [
            "dataset",
            "algorithm",
            "kind",
            "nominal_fraction",
            "changed_labels",
            "excluded_events",
            "delta_ari_from_baseline",
            "delta_macro_f1_from_baseline",
        ]
    ]
    label.columns = [
        "Dataset",
        "Method",
        "Scenario",
        "Nominal fraction",
        "Changed labels",
        "Excluded events",
        "Δ ARI",
        "Δ Macro F1",
    ]
    marker = pd.read_csv(IN / "marker_dimension.csv")
    marker = marker[
        [
            "n_markers",
            "n_marker_chains_contributing",
            "ari_mean_across_chain_mean",
            "ari_mean_across_chain_sd",
            "macro_f1_mean_across_chain_mean",
            "macro_f1_mean_across_chain_sd",
        ]
    ]
    marker.columns = [
        "Markers",
        "Independent marker chains",
        "ARI mean",
        "ARI SD across chains",
        "Macro F1 mean",
        "Macro F1 SD across chains",
    ]
    auto = pd.read_csv(IN / "flowsom_auto_multiclass.csv")
    auto = auto[auto.metric.isin(["ari", "macro_f1"])][
        ["dataset", "regime", "metric", "n", "mean", "sd", "min", "max"]
    ]
    auto.columns = ["Dataset", "Regime", "Metric", "n seeds", "Mean", "SD", "Min", "Max"]
    return grid, label, marker, auto


def write_csv(name, df):
    df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.10g")


def write_workbook(tables):
    path = OUT / "tables.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.book[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="17324D")
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[1].height = 30
            for col in ws.columns:
                letter = get_column_letter(col[0].column)
                max_len = max(len(str(c.value or "")) for c in list(col)[:100])
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 44)
                for c in list(col)[1:]:
                    c.alignment = Alignment(vertical="top", wrap_text=True)
    return path


def main():
    t1 = table1()
    t2 = table2()
    a, b = s2()
    grid, labels, markers, auto = s6()
    tests, power = s5_tests_and_power()
    tables = {
        "Table1_Data": t1,
        "Table2_Methods": t2,
        "TableS1_Identity": s1(t1),
        "TableS2_Methods": a,
        "TableS2_Regimes": b,
        "TableS3_Populations": s3(),
        "TableS4_Rare": s4(),
        "TableS5_Stability": s5(),
        "TableS5_Holm": tests,
        "TableS5_Power": power,
        "TableS6_Grid": grid,
        "TableS6_AutoK": auto,
        "TableS6_Labels": labels,
        "TableS6_Markers": markers,
        "TableS6_EventInclusion": pd.read_csv(IN / "kmeans_event_inclusion.csv"),
    }
    for name, df in tables.items():
        if df.empty:
            raise ValueError(f"Empty table {name}")
        if name == "TableS6_EventInclusion":
            shutil.copyfile(IN / "kmeans_event_inclusion.csv", OUT / (name + ".csv"))
        else:
            write_csv(name, df)
    path = write_workbook(tables)
    print("Built", len(tables), "table sheets; workbook:", path)


if __name__ == "__main__":
    main()
