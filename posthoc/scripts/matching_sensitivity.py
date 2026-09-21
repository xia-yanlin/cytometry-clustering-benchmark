"""Post hoc metric check of frozen contingency tables, 20 September 2026.
No clustering is re-run. Primary overlap-count Hungarian matching is retained.
Usage: python matching_sensitivity.py --source /path/to/extracted/source --out ./out
Dependencies: numpy, pandas, scipy. The input archive contains all required tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def measures(a: np.ndarray, objective: str) -> dict:
    nref = a.shape[0]
    rs, cs = a.sum(1), a.sum(0)
    f = np.divide(
        2 * a,
        rs[:, None] + cs[None, :],
        out=np.zeros_like(a, dtype=float),
        where=(rs[:, None] + cs[None, :]) != 0,
    )
    ri, ci = linear_sum_assignment(-(a if objective == "overlap" else f))
    p = np.zeros(nref)
    r = np.zeros(nref)
    fs = np.zeros(nref)
    p[ri] = np.divide(a[ri, ci], cs[ci], out=np.zeros(len(ri)), where=cs[ci] != 0)
    r[ri] = np.divide(a[ri, ci], rs[ri], out=np.zeros(len(ri)), where=rs[ri] != 0)
    fs[ri] = f[ri, ci]
    return {
        "macro_precision": p.mean(),
        "macro_recall": r.mean(),
        "macro_f1": fs.mean(),
        "matched_event_fraction": a[ri, ci].sum() / a.sum(),
        "assignment": list(zip(ri.tolist(), ci.tolist(), strict=True)),
    }


def write_manuscript_tables(
    summary: pd.DataFrame, runs: pd.DataFrame, met: pd.DataFrame, out: Path
) -> None:
    """Write the rounded, reader-facing versions of Tables S7A and S7B."""

    method_names = {
        "PhenoGraph_fixed_graph": "PhenoGraph fixed graph",
        "R_FlowSOM_full": "R FlowSOM",
        "Xshift_K20": "X-shift K20",
    }
    table_a = summary[summary.method.ne("Python_FlowSOM_inclusion")].copy()
    table_a["Analysis"] = table_a.method.map(method_names)
    if table_a["Analysis"].isna().any():
        raise ValueError("Table S7A contains an unmapped analysis name")
    table_a = table_a.assign(
        Dataset=table_a.dataset,
        **{
            "Primary F1": table_a.overlap_f1_mean,
            "Alternative F1": table_a.f1optimal_f1_mean,
            "Mean increase": table_a.delta_f1_mean,
            "Max increase": table_a.delta_f1_max,
        },
    )[
        [
            "Analysis",
            "Dataset",
            "n",
            "Primary F1",
            "Alternative F1",
            "Mean increase",
            "Max increase",
        ]
    ]

    inclusion_names = {
        "all_direct": "Python FlowSOM all-event",
        "labeled_direct_a": "Python FlowSOM labeled-only",
    }
    inclusion = runs[runs.method.eq("Python_FlowSOM_inclusion")].copy()
    inclusion["Analysis"] = inclusion.run.map(inclusion_names)
    if len(inclusion) != 2 or inclusion["Analysis"].isna().any():
        raise ValueError("Expected the two predefined Python FlowSOM inclusion runs")
    inclusion = inclusion.assign(
        Dataset=inclusion.dataset,
        n=1,
        **{
            "Primary F1": inclusion.overlap_macro_f1,
            "Alternative F1": inclusion.f1optimal_macro_f1,
            "Mean increase": inclusion.delta_macro_f1,
            "Max increase": inclusion.delta_macro_f1,
        },
    )[table_a.columns]
    table_a = pd.concat([table_a, inclusion], ignore_index=True)
    table_a.to_csv(
        out / "TableS7A_matching.csv", index=False, float_format="%.6f", encoding="utf-8-sig"
    )

    def display_range(values: pd.Series) -> str:
        low, high = int(values.min()), int(values.max())
        return str(low) if low == high else f"{low}–{high}"

    regime_names = {
        "official_auto_max40": "Automatic",
        "official_fixed_ktrue": "Reference count",
        "official_fixed_k40": "Fixed 40",
    }
    records = []
    for dataset in ("Levine_32dim", "Samusik_01"):
        for regime in regime_names:
            group = met[met.dataset.eq(dataset) & met.regime.eq(regime)]
            if len(group) != 30:
                raise ValueError(f"Expected 30 rows for {dataset}, {regime}; found {len(group)}")
            records.append(
                {
                    "Dataset": dataset,
                    "Metaclustering": regime_names[regime],
                    "Selected K": display_range(group.selected_k),
                    "Evaluable C": display_range(group.n_predicted_clusters_evaluable),
                    "Reference R": int(group.n_true_populations.iloc[0]),
                    "Mean ceiling": group.matching_ceiling.mean(),
                    "Mean F1": group.macro_f1.mean(),
                }
            )
    pd.DataFrame(records).to_csv(
        out / "TableS7B_ceiling.csv", index=False, float_format="%.6f", encoding="utf-8-sig"
    )


def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--source", type=Path, required=True)
    pa.add_argument("--out", type=Path, required=True)
    ar = pa.parse_args()
    ar.out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("EXP-045A_", "PhenoGraph_fixed_graph", "Samusik_01"),
        ("EXP-045B_", "PhenoGraph_fixed_graph", "Levine_13dim"),
        ("EXP-045C_", "PhenoGraph_fixed_graph", "Levine_32dim"),
        ("EXP-046A-R1_", "R_FlowSOM_full", "Samusik_01"),
        ("EXP-046B-R1_", "R_FlowSOM_full", "Levine_32dim"),
        ("EXP-015-R2_", "Python_FlowSOM_inclusion", "Levine_13dim"),
        ("EXP-011C_", "Xshift_K20", "Levine_32dim"),
    ]
    records = []
    for pref, method, dataset in specs:
        dirs = list((ar.source / "experiment_records/runs").glob(pref + "*"))
        if len(dirs) != 1:
            raise ValueError(f"Expected one directory for {pref}, found {dirs}")
        paths = sorted(dirs[0].rglob("contingency_true_by_predicted.csv"))
        for p in paths:
            if pref == "EXP-015-R2_" and p.parent.name not in ["labeled_direct_a", "all_direct"]:
                continue
            a = pd.read_csv(p, index_col=0).to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a < 0).any() or not np.equal(a, np.floor(a)).all():
                raise ValueError(str(p))
            x = measures(a, "overlap")
            y = measures(a, "f1")
            if y["macro_f1"] + 1e-12 < x["macro_f1"]:
                raise AssertionError("F1 optimum is lower than primary")
            rec = {
                "method": method,
                "dataset": dataset,
                "run": p.parent.name,
                "n_evaluable": int(a.sum()),
                "n_reference": a.shape[0],
                "n_predicted_evaluable": a.shape[1],
                "source": p.relative_to(ar.source).as_posix(),
            }
            for k in ["macro_precision", "macro_recall", "macro_f1", "matched_event_fraction"]:
                rec["overlap_" + k] = x[k]
                rec["f1optimal_" + k] = y[k]
                rec["delta_" + k] = y[k] - x[k]
            rec["assignment_changed"] = x["assignment"] != y["assignment"]
            records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(ar.out / "matching_run_results.csv", index=False)
    summary = (
        df.groupby(["method", "dataset"], sort=False)
        .agg(
            n=("run", "size"),
            overlap_f1_mean=("overlap_macro_f1", "mean"),
            f1optimal_f1_mean=("f1optimal_macro_f1", "mean"),
            delta_f1_mean=("delta_macro_f1", "mean"),
            delta_f1_min=("delta_macro_f1", "min"),
            delta_f1_max=("delta_macro_f1", "max"),
            assignments_changed=("assignment_changed", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(ar.out / "matching_summary.csv", index=False)
    # Confirm primary values against the frozen data plotted in Figure 3.
    checks = []
    for f, method in [
        ("figure3a_flowsom_r_run_points.csv", "R_FlowSOM_full"),
        ("figure3b_phenograph_run_points.csv", "PhenoGraph_fixed_graph"),
    ]:
        ref = pd.read_csv(ar.source / "package/figure_data" / f)
        for ds, g in df[df.method == method].groupby("dataset"):
            vals = ref[ref.dataset == ds].macro_f1.to_numpy()
            diff = float(np.max(np.abs(np.sort(vals) - np.sort(g.overlap_macro_f1.to_numpy()))))
            checks.append(
                {
                    "check": f"{method} {ds} primary macro-F1 reconstruction",
                    "max_abs_difference": diff,
                    "pass": diff < 1e-12,
                }
            )
    met = pd.read_csv(ar.source / "package/inputs/flowsom_auto_multiclass_runs.csv")
    met = met[
        (met.selector_seed == 12345)
        & met.regime.isin(["official_auto_max40", "official_fixed_ktrue", "official_fixed_k40"])
    ].copy()
    if len(met) != 180:
        print(
            "Meta filters found",
            len(met),
            "rows; regimes:",
            pd.read_csv(
                ar.source / "package/inputs/flowsom_auto_multiclass_runs.csv"
            ).regime.unique(),
        )
    met["matching_ceiling"] = (
        np.minimum(met.n_predicted_clusters_evaluable, met.n_true_populations)
        / met.n_true_populations
    )
    met["ceiling_satisfied"] = met.macro_f1 <= met.matching_ceiling + 1e-12
    met.to_csv(ar.out / "metacluster_matching_ceiling.csv", index=False)
    pd.DataFrame(checks).to_csv(ar.out / "reconstruction_checks.csv", index=False)
    if not all(c["pass"] for c in checks):
        raise AssertionError("Primary reconstruction failed")
    if not met.ceiling_satisfied.all():
        raise AssertionError("Matching ceiling check failed")
    write_manuscript_tables(summary, df, met, ar.out)
    print(summary.to_string(index=False))
    print(
        "Computed",
        len(df),
        "contingency-table comparisons and",
        len(met),
        "metaclustering ceilings",
    )


if __name__ == "__main__":
    main()
