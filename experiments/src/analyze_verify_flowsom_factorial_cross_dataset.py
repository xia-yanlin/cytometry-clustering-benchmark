from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CONTRASTS = [
    "grid15_vs10_at_rlen10",
    "grid20_vs10_at_rlen10",
    "rlen20_vs10_at_grid10",
    "rlen30_vs10_at_grid10",
]
METRICS = ["median_bmu_distance", "ari", "macro_f1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def raw_direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def outcome_direction(metric: str, value: float) -> str:
    if value == 0:
        return "no_change"
    improved = value < 0 if metric == "median_bmu_distance" else value > 0
    return "improved" if improved else "worsened"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-a", type=Path, required=True)
    parser.add_argument("--analysis-b", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-014C")
    args = parser.parse_args()

    inputs = [args.analysis_a.resolve(), args.analysis_b.resolve()]
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    output.mkdir(parents=True)

    checks: list[dict] = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    datasets: list[str] = []
    combined_cells: list[pd.DataFrame] = []
    combined_contrasts: list[pd.DataFrame] = []
    input_records: list[dict] = []

    expected_pairs = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
    for analysis in inputs:
        required = [
            analysis / "run_manifest.json",
            analysis / "verification_checks.csv",
            analysis / "artifact_hashes.json",
            analysis / "factorial_cell_summary.csv",
            analysis / "preregistered_paired_contrasts.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        check(analysis.name, "required_files", not missing, json.dumps(missing, ensure_ascii=False))
        if missing:
            continue

        manifest_path, verification_path, inventory_path, cells_path, contrasts_path = required
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = pd.read_csv(verification_path)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        manifest_dataset = str(manifest.get("dataset", "")).strip()
        parent_index_path = Path(str(manifest.get("parent_run", ""))) / "run_index.csv"
        if not parent_index_path.is_file():
            raise FileNotFoundError(f"Parent run index missing: {parent_index_path}")
        parent_index = pd.read_csv(parent_index_path, usecols=["dataset"])
        parent_datasets = parent_index["dataset"].astype(str).unique().tolist()
        if len(parent_datasets) != 1:
            raise ValueError(f"Expected one parent dataset, found {parent_datasets}")
        parent_dataset = str(parent_datasets[0])
        if manifest_dataset and manifest_dataset != parent_dataset:
            raise ValueError(
                f"Manifest/parent dataset mismatch: manifest={manifest_dataset}; parent={parent_dataset}"
            )
        dataset = manifest_dataset or parent_dataset
        datasets.append(dataset)

        check(
            dataset,
            "dataset_identity_resolved",
            bool(dataset) and len(parent_datasets) == 1,
            f"manifest={manifest_dataset or '<missing>'}; parent_index={parent_dataset}; resolved={dataset}",
        )

        check(
            dataset,
            "parent_analysis_passed",
            manifest.get("all_checks_passed") is True
            and int(manifest.get("checks_passed", -1)) == 281
            and int(manifest.get("checks_total", -2)) == 281
            and len(verification) == 281
            and verification["passed"].astype(str).str.lower().eq("true").all(),
            f"manifest={manifest.get('checks_passed')}/{manifest.get('checks_total')}; rows={len(verification)}",
        )
        hash_failures = []
        for path_text, expected_hash in inventory.items():
            path = Path(path_text)
            if not path.is_file() or sha256(path) != expected_hash:
                hash_failures.append(path_text)
        check(
            dataset,
            "parent_artifact_hashes",
            not hash_failures,
            json.dumps(hash_failures, ensure_ascii=False),
        )

        cells = pd.read_csv(cells_path)
        cell_pairs = set(
            zip(cells["grid_side"].astype(int), cells["rlen"].astype(int), strict=False)
        )
        check(
            dataset,
            "factorial_cells_complete",
            len(cells) == 9
            and cell_pairs == {(grid, rlen) for grid in (10, 15, 20) for rlen in (10, 20, 30)}
            and cells["n_seeds"].astype(int).eq(5).all(),
            f"rows={len(cells)}; pairs={sorted(cell_pairs)}",
        )
        cells.insert(0, "dataset", dataset)
        combined_cells.append(cells)

        all_contrasts = pd.read_csv(contrasts_path)
        selected = all_contrasts[
            all_contrasts["contrast"].isin(CONTRASTS) & all_contrasts["metric"].isin(METRICS)
        ].copy()
        actual_pairs = set(zip(selected["contrast"], selected["metric"], strict=False))
        check(
            dataset,
            "main_contrast_set_complete",
            len(selected) == 12
            and actual_pairs == expected_pairs
            and selected["n_paired_seeds"].astype(int).eq(5).all(),
            f"rows={len(selected)}; missing={sorted(expected_pairs - actual_pairs)}; extra={sorted(actual_pairs - expected_pairs)}",
        )
        selected.insert(0, "dataset", dataset)
        combined_contrasts.append(selected)
        input_records.append(
            {
                "analysis_directory": str(analysis),
                "dataset": dataset,
                "manifest_sha256": sha256(manifest_path),
                "artifact_inventory_sha256": sha256(inventory_path),
                "factorial_summary_sha256": sha256(cells_path),
                "paired_contrasts_sha256": sha256(contrasts_path),
            }
        )

    check(
        "experiment",
        "two_distinct_expected_datasets",
        set(datasets) == {"Levine_32dim", "Samusik_01"},
        json.dumps(datasets),
    )
    if len(combined_cells) != 2 or len(combined_contrasts) != 2:
        raise RuntimeError("Both verified input analyses are required")

    cells_out = pd.concat(combined_cells, ignore_index=True)
    contrasts_out = pd.concat(combined_contrasts, ignore_index=True)
    cells_out.to_csv(output / "cross_dataset_factorial_cells.csv", index=False)
    contrasts_out.to_csv(output / "cross_dataset_main_contrasts.csv", index=False)

    direction_rows = []
    for contrast in CONTRASTS:
        for metric in METRICS:
            rows = contrasts_out[
                (contrasts_out["contrast"] == contrast) & (contrasts_out["metric"] == metric)
            ]
            values = {
                str(row.dataset): float(row.mean_difference_target_minus_reference)
                for row in rows.itertuples(index=False)
            }
            a = values["Levine_32dim"]
            b = values["Samusik_01"]
            direction_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "levine32_mean_difference": a,
                    "samusik_mean_difference": b,
                    "levine32_raw_direction": raw_direction(a),
                    "samusik_raw_direction": raw_direction(b),
                    "raw_direction_concordant": raw_direction(a) == raw_direction(b)
                    and raw_direction(a) != "zero",
                    "levine32_outcome_direction": outcome_direction(metric, a),
                    "samusik_outcome_direction": outcome_direction(metric, b),
                    "both_improved": outcome_direction(metric, a)
                    == outcome_direction(metric, b)
                    == "improved",
                    "interpretation_role": "descriptive_cross_dataset_check_not_meta_analysis_or_parameter_selection",
                }
            )
    directions = pd.DataFrame(direction_rows)
    directions.to_csv(output / "cross_dataset_direction_concordance.csv", index=False)
    check(
        "experiment",
        "direction_table_complete",
        len(directions) == 12 and not directions.isna().any().any(),
        f"rows={len(directions)}",
    )

    verification_out = pd.DataFrame(checks)
    verification_out.to_csv(output / "verification_checks.csv", index=False)

    concordant = int(directions["raw_direction_concordant"].sum())
    both_improved = int(directions["both_improved"].sum())
    per_metric = directions.groupby("metric", sort=False).agg(
        concordant=("raw_direction_concordant", "sum"), both_improved=("both_improved", "sum")
    )
    summary_lines = [
        "# EXP-014C Descriptive cross-dataset check of FlowSOM factorial effects",
        "",
        f"Verification and artifact hashes passed for both parent analyses; the new checks passed {int(verification_out['passed'].sum())}/{len(verification_out)}.",
        f"Across four preregistered primary contrasts and three endpoints (12 comparisons), the raw effect direction agreed between datasets in {concordant}/12, and both datasets improved in the endpoint-specific direction in {both_improved}/12.",
        "",
    ]
    for metric, row in per_metric.iterrows():
        summary_lines.append(
            f"- {metric}: direction agreed in {int(row['concordant'])}/4; both datasets improved in {int(row['both_improved'])}/4."
        )
    summary_lines.extend(
        [
            "",
            "These results describe replication direction only for the two fixed datasets, the fixed 20,000-event training regime, and 40 metaclusters. Two datasets do not support population-level inference; no cross-dataset significance test was performed, and external labels must not be used to choose a purportedly optimal parameter setting.",
        ]
    )
    (output / "scientific_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": input_records,
        "datasets": sorted(datasets),
        "factorial_cells": int(len(cells_out)),
        "dataset_level_main_contrasts": int(len(contrasts_out)),
        "cross_dataset_direction_checks": int(len(directions)),
        "raw_direction_concordant": concordant,
        "both_datasets_improved": both_improved,
        "checks_passed": int(verification_out["passed"].sum()),
        "checks_total": int(len(verification_out)),
        "all_checks_passed": bool(verification_out["passed"].all()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_paths = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {str(path): sha256(path) for path in artifact_paths}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
