"""Descriptive post hoc selection from frozen FlowSOM summaries (2026-09-20).
No clustering or held-out evaluation is performed.
Usage: python objective_selection.py --source /path/to/extracted/source --out ./out
Dependencies: pandas, numpy. Input: package/inputs/flowsom_pareto.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    relative = Path("package/inputs/flowsom_pareto.csv")
    path = args.source / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen input: {path}")
    data = pd.read_csv(path)
    columns = [
        "dataset",
        "configuration",
        "grid_side",
        "rlen",
        "ari_mean",
        "macro_f1_mean",
        "median_bmu_distance_mean",
    ]
    if any(c not in data.columns for c in columns):
        raise ValueError("Input lacks required objective or configuration columns")
    if len(data) != 18 or data.duplicated(["dataset", "configuration"]).any():
        raise ValueError("Expected 18 unique dataset/configuration rows")
    values = data[["ari_mean", "macro_f1_mean", "median_bmu_distance_mean"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite objective value")
    result = []
    for dataset, group in data.groupby("dataset", sort=True):
        if len(group) != 9:
            raise ValueError(f"Expected nine configurations in {dataset}")
        for objective, field, ascending in [
            ("maximize_ARI", "ari_mean", False),
            ("maximize_macro_F1", "macro_f1_mean", False),
            ("minimize_median_BMU_distance", "median_bmu_distance_mean", True),
        ]:
            selected = group.sort_values(
                [field, "grid_side", "rlen"], ascending=[ascending, True, True], kind="stable"
            ).iloc[0]
            result.append(
                {
                    "dataset": dataset,
                    "selection_objective": objective,
                    "configuration": selected.configuration,
                    "grid_side": int(selected.grid_side),
                    "rlen": int(selected.rlen),
                    "mean_ARI": selected.ari_mean,
                    "mean_macro_F1": selected.macro_f1_mean,
                    "mean_median_BMU_distance": selected.median_bmu_distance_mean,
                    "candidate_configurations": 9,
                    "seeds_per_configuration": 5,
                    "held_out_evaluation": False,
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(result)
    detailed.to_csv(args.out / "selection_objective_sensitivity.csv", index=False)

    objective_names = {
        "maximize_ARI": "Maximize ARI",
        "maximize_macro_F1": "Maximize macro F1",
        "minimize_median_BMU_distance": "Minimize BMU distance",
    }
    manuscript = detailed.assign(
        Dataset=detailed.dataset,
        **{
            "Selection objective": detailed.selection_objective.map(objective_names),
            "Grid / rlen": detailed.grid_side.astype(str) + " / " + detailed.rlen.astype(str),
            "Mean ARI": detailed.mean_ARI,
            "Mean macro F1": detailed.mean_macro_F1,
            "Mean median BMU": detailed.mean_median_BMU_distance,
        },
    )[
        [
            "Dataset",
            "Selection objective",
            "Grid / rlen",
            "Mean ARI",
            "Mean macro F1",
            "Mean median BMU",
        ]
    ]
    if manuscript["Selection objective"].isna().any():
        raise ValueError("Table S7C contains an unmapped selection objective")
    manuscript.to_csv(
        args.out / "TableS7C_objectives.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    print(detailed.to_string(index=False))


if __name__ == "__main__":
    main()
