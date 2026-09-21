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


DATASETS = {
    "Levine_32dim": ("EXP-014A_flowsom_grid_rlen_factorial_levine32_20260910", "EXP-014A-V_flowsom_grid_rlen_factorial_analysis_20260910"),
    "Samusik_01": ("EXP-014B_flowsom_grid_rlen_factorial_samusik_20260910", "EXP-014B-V_flowsom_grid_rlen_factorial_analysis_20260910"),
}
UTILITY_COLUMNS = ["neg_median_bmu_distance", "ari", "macro_f1"]
TOLERANCE = 1e-12
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_910


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_inventory(path: Path) -> list[str]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, expected in inventory.items() if not Path(name).is_file() or sha256(Path(name)) != expected]


def pareto_mask(values: np.ndarray) -> np.ndarray:
    result = np.ones(len(values), dtype=bool)
    for candidate in range(len(values)):
        for challenger in range(len(values)):
            if candidate == challenger:
                continue
            no_worse = np.all(values[challenger] >= values[candidate] - TOLERANCE)
            strictly_better = np.any(values[challenger] > values[candidate] + TOLERANCE)
            if no_worse and strictly_better:
                result[candidate] = False
                break
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, protocol, output = args.runs_root.resolve(), args.protocol.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks: list[dict] = []
    run_frames: list[pd.DataFrame] = []
    source_links: list[dict] = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    for dataset, (parent_name, analysis_name) in DATASETS.items():
        parent, analysis = root / parent_name, root / analysis_name
        progress = json.loads((parent / "progress_manifest.json").read_text(encoding="utf-8"))
        analysis_manifest = json.loads((analysis / "run_manifest.json").read_text(encoding="utf-8"))
        check(dataset, "parent_complete", progress.get("all_runs_complete_and_passed") is True and progress.get("completed_runs") == 45, json.dumps(progress))
        check(dataset, "analysis_complete", analysis_manifest.get("all_checks_passed") is True, json.dumps(analysis_manifest))
        check(dataset, "analysis_hashes", not (failures := verify_inventory(analysis / "artifact_hashes.json")), json.dumps(failures))
        frame = pd.read_csv(parent / "run_index.csv")
        frame = frame[["dataset", "grid_side", "rlen", "seed", "all_checks_passed", "median_bmu_distance", "ari", "macro_f1"]].copy()
        exact = len(frame) == 45 and frame["all_checks_passed"].astype(str).str.lower().eq("true").all()
        exact = exact and set(frame["grid_side"]) == {10, 15, 20} and set(frame["rlen"]) == {10, 20, 30} and set(frame["seed"]) == set(range(5))
        exact = exact and frame.groupby(["grid_side", "rlen"]).size().eq(5).all() and set(frame["dataset"]) == {dataset}
        check(dataset, "balanced_design", exact, f"rows={len(frame)}; cells={frame.groupby(['grid_side','rlen']).size().to_dict()}")
        frame["configuration"] = frame.apply(lambda row: f"grid{int(row.grid_side)}_rlen{int(row.rlen)}", axis=1)
        frame["neg_median_bmu_distance"] = -frame["median_bmu_distance"]
        run_frames.append(frame)
        source_links.append({
            "dataset": dataset, "parent": str(parent), "progress_manifest_sha256": sha256(parent / "progress_manifest.json"),
            "run_index_sha256": sha256(parent / "run_index.csv"), "analysis": str(analysis),
            "analysis_manifest_sha256": sha256(analysis / "run_manifest.json"), "analysis_inventory_sha256": sha256(analysis / "artifact_hashes.json"),
        })

    runs = pd.concat(run_frames, ignore_index=True)
    runs.to_csv(output / "frozen_run_level_input.csv", index=False)
    summaries: list[pd.DataFrame] = []
    dominance_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for dataset, dataset_runs in runs.groupby("dataset", sort=True):
        summary = dataset_runs.groupby(["configuration", "grid_side", "rlen"])[["median_bmu_distance", "ari", "macro_f1"]].agg(["mean", "std", "median", "min", "max"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary["neg_median_bmu_distance"] = -summary["median_bmu_distance_mean"]
        summary["ari"] = summary["ari_mean"]
        summary["macro_f1"] = summary["macro_f1_mean"]
        values = summary[UTILITY_COLUMNS].to_numpy(float)
        summary["pareto_front_mean"] = pareto_mask(values)
        summary.insert(0, "dataset", dataset)
        summaries.append(summary)

        configs = summary["configuration"].tolist()
        by_config = {config: dataset_runs.loc[dataset_runs["configuration"] == config].sort_values("seed") for config in configs}
        for left_index, left in enumerate(configs):
            for right_index, right in enumerate(configs):
                if left == right:
                    continue
                left_values = by_config[left][UTILITY_COLUMNS].to_numpy(float)
                right_values = by_config[right][UTILITY_COLUMNS].to_numpy(float)
                per_seed = np.all(left_values >= right_values - TOLERANCE, axis=1) & np.any(left_values > right_values + TOLERANCE, axis=1)
                mean_dom = bool(np.all(values[left_index] >= values[right_index] - TOLERANCE) and np.any(values[left_index] > values[right_index] + TOLERANCE))
                dominance_rows.append({"dataset": dataset, "configuration_a": left, "configuration_b": right, "a_dominates_b_on_means": mean_dom, "paired_seed_dominance_count": int(per_seed.sum()), "unanimous_seed_dominance": bool(per_seed.all())})

        bootstrap_counts = dict.fromkeys(configs, 0)
        matrices = {config: by_config[config][UTILITY_COLUMNS].to_numpy(float) for config in configs}
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled_means = np.vstack([matrices[config][rng.integers(0, 5, 5)].mean(axis=0) for config in configs])
            for config, is_front in zip(configs, pareto_mask(sampled_means)):
                bootstrap_counts[config] += int(is_front)
        for config in configs:
            bootstrap_rows.append({"dataset": dataset, "configuration": config, "pareto_inclusion_count": bootstrap_counts[config], "pareto_inclusion_frequency": bootstrap_counts[config] / BOOTSTRAP_REPLICATES})

    summary_frame = pd.concat(summaries, ignore_index=True)
    dominance = pd.DataFrame(dominance_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary_frame.to_csv(output / "configuration_objective_summary.csv", index=False)
    dominance.to_csv(output / "ordered_pair_dominance.csv", index=False)
    bootstrap.to_csv(output / "pareto_bootstrap_inclusion.csv", index=False)
    front_sets = {dataset: set(group.loc[group["pareto_front_mean"], "configuration"]) for dataset, group in summary_frame.groupby("dataset")}
    union = sorted(set.union(*front_sets.values()))
    intersection = sorted(set.intersection(*front_sets.values()))
    cross = pd.DataFrame({"configuration": sorted(set(summary_frame["configuration"])), **{f"pareto_{dataset}": [config in front_sets[dataset] for config in sorted(set(summary_frame["configuration"]))] for dataset in DATASETS}})
    cross["pareto_both_datasets"] = cross[[column for column in cross if column.startswith("pareto_") and column != "pareto_both_datasets"]].all(axis=1)
    cross.to_csv(output / "cross_dataset_pareto_membership.csv", index=False)

    for dataset, group in summary_frame.groupby("dataset"):
        values = group[UTILITY_COLUMNS].to_numpy(float)
        mask = group["pareto_front_mean"].to_numpy(bool)
        internal_ok = not any(np.all(values[j] >= values[i] - TOLERANCE) and np.any(values[j] > values[i] + TOLERANCE) for i in np.flatnonzero(mask) for j in range(len(values)) if i != j)
        covered_ok = all(any(np.all(values[j] >= values[i] - TOLERANCE) and np.any(values[j] > values[i] + TOLERANCE) for j in range(len(values)) if j != i) for i in np.flatnonzero(~mask))
        check(dataset, "pareto_front_contract", internal_ok and covered_ok, f"front={sorted(group.loc[mask, 'configuration'])}; internal={internal_ok}; nonfront_covered={covered_ok}")
        frequencies = bootstrap.loc[bootstrap["dataset"] == dataset, "pareto_inclusion_frequency"]
        check(dataset, "bootstrap_contract", len(frequencies) == 9 and frequencies.between(0, 1).all() and int(bootstrap.loc[bootstrap["dataset"] == dataset, "pareto_inclusion_count"].max()) <= BOOTSTRAP_REPLICATES, f"replicates={BOOTSTRAP_REPLICATES}; range={frequencies.min()}-{frequencies.max()}")

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "analysis_checks.csv", index=False)
    manifest = {
        "experiment_id": "EXP-020", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol), "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "source_links": source_links, "objectives": {"minimize": ["median_bmu_distance"], "maximize": ["ari", "macro_f1"]},
        "tolerance": TOLERANCE, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "mean_pareto_fronts": {dataset: sorted(front) for dataset, front in front_sets.items()}, "front_union": union, "front_intersection": intersection,
        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame["passed"].all()),
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unanimous = dominance.loc[dominance["unanimous_seed_dominance"]]
    multi_front = [dataset for dataset, front in front_sets.items() if len(front) > 1]
    lines = ["# EXP-020 FlowSOM多目标Pareto分析", "", f"配置均值前沿：{json.dumps(manifest['mean_pareto_fronts'], ensure_ascii=False)}", f"两数据集前沿交集：{intersection or '空'}；并集：{union}。", f"存在多个均值前沿配置的数据集：{multi_front or '无'}；5/5 seed一致支配的有序配置对共{len(unanimous)}个。", "", "三个目标未加权且未合成为单一分数。本结果是已有FlowSOM配置的描述性Pareto审计，不是新的多目标优化算法，也不提供跨算法排名。"]
    (output / "scientific_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(json.dumps({str(path): sha256(path) for path in artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
