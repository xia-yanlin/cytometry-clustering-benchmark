from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CYTOMETRY_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "raw")))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


DATASETS = {
    "Nilsson_rare": (DATA_ROOT / "Nilsson_rare" / "Nilsson_rare_notransform.csv", "HSCs", 358),
    "Mosmann_rare": (DATA_ROOT / "Mosmann_rare" / "Mosmann_rare_notransform.csv", "activated", 109),
}
MAIN_REGIMES = ("official_auto_max40", "official_fixed_ktrue_2", "official_fixed_k10", "official_fixed_k40")
SENSITIVITY_REGIME = "official_auto_max40_selector_seed_sensitivity"
METRICS = ("ari", "target_precision", "target_recall", "target_f1", "target_f2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def effective_categories(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0].astype(float)
    probabilities /= probabilities.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def evaluate_rare(labels: np.ndarray, clusters: np.ndarray, target: str) -> tuple[dict[str, object], pd.DataFrame]:
    actual = labels == target
    target_events = int(actual.sum())
    rows: list[dict[str, object]] = []
    for cluster in np.unique(clusters):
        predicted = clusters == cluster
        tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted)); fn = target_events - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / target_events if target_events else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
        rows.append({"predicted_cluster": int(cluster), "cluster_size": int(predicted.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "f2": f2})
    frame = pd.DataFrame(rows).sort_values(["f1", "recall", "predicted_cluster"], ascending=[False, False, True]).reset_index(drop=True)
    best = frame.iloc[0]
    target_counts = frame.tp.to_numpy(dtype=np.int64)
    metrics = {
        "n_total_events": len(labels), "n_predicted_metaclusters": int(np.unique(clusters).size),
        "target": target, "target_events": target_events, "target_prevalence": float(actual.mean()),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "selected_target_cluster": int(best.predicted_cluster), "selected_cluster_size": int(best.cluster_size),
        "target_tp": int(best.tp), "target_fp": int(best.fp), "target_fn": int(best.fn),
        "target_precision": float(best.precision), "target_recall": float(best.recall),
        "target_f1": float(best.f1), "target_f2": float(best.f2),
        "target_overlapping_clusters": int(np.sum(target_counts > 0)),
        "target_effective_predicted_clusters": effective_categories(target_counts),
        "target_dominant_cluster_capture": float(target_counts.max() / target_events),
    }
    return metrics, frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent, parent_verification, protocol, output = args.parent.resolve(), args.parent_verification.resolve(), args.protocol.resolve(), args.output.resolve()
    progress = json.loads((parent / "progress_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((parent_verification / "run_manifest.json").read_text(encoding="utf-8"))
    if progress.get("all_runs_complete_and_passed") is not True or verification.get("all_checks_passed") is not True:
        raise RuntimeError("EXP-033A/033AV parent is not eligible")
    output.mkdir(parents=True, exist_ok=False)
    codes_dir = output / "codes_inputs"; r_dir = output / "r_outputs"; event_dir = output / "event_partitions"
    codes_dir.mkdir(); r_dir.mkdir(); event_dir.mkdir()
    exp_root = Path(__file__).resolve().parent.parent
    r_script = exp_root / "src" / "run_flowsom_r_rare_metaclustering_batch.R"
    r_library = exp_root / "environments" / "blflowsom_r452" / "library"

    input_rows, parent_hashes = [], []
    nodes_by_run: dict[tuple[str, int], np.ndarray] = {}
    labels_by_dataset: dict[str, np.ndarray] = {}
    for dataset, (data_path, target, target_events) in DATASETS.items():
        labels = pd.read_csv(data_path, usecols=["label"], low_memory=False)["label"].astype(str).to_numpy()
        if int(np.sum(labels == target)) != target_events:
            raise ValueError(f"target count failed: {dataset}")
        labels_by_dataset[dataset] = labels
        for seed in range(30):
            run_dir = parent / "runs" / dataset / f"seed{seed:03d}"
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("all_checks_passed") is not True:
                raise RuntimeError(f"ineligible parent run: {run_dir}")
            codes_path = run_dir / "som_codes.npy"; nodes_path = run_dir / "node_labels_all_events.npy"
            codes = np.load(codes_path, allow_pickle=False); nodes = np.load(nodes_path, allow_pickle=False)
            if codes.shape[0] != 100 or nodes.shape != labels.shape:
                raise ValueError(f"shape failed: {dataset}:{seed}")
            csv_path = codes_dir / f"{dataset}_som_seed{seed:03d}.csv"
            pd.DataFrame(codes, columns=[f"feature_{i+1:02d}" for i in range(codes.shape[1])]).to_csv(csv_path, index=False)
            input_rows.append({"dataset": dataset, "som_seed": seed, "codes_path": str(csv_path.resolve())})
            nodes_by_run[(dataset, seed)] = nodes.astype(np.int16, copy=False)
            for role, path in (("parent_manifest", run_dir / "run_manifest.json"), ("som_codes", codes_path), ("node_labels", nodes_path)):
                parent_hashes.append({"dataset": dataset, "som_seed": seed, "role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(input_rows).to_csv(output / "r_input_manifest.csv", index=False)
    pd.DataFrame(parent_hashes).to_csv(output / "parent_input_hashes.csv", index=False)

    env = os.environ.copy(); env["R_LIBS_USER"] = str(r_library)
    command = [str(args.rscript.resolve()), str(r_script.resolve()), str((output / "r_input_manifest.csv").resolve()), str(r_dir.resolve())]
    with (output / "r_stdout.log").open("w", encoding="utf-8") as stdout, (output / "r_stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, text=True, stdout=stdout, stderr=stderr, env=env)
    if completed.returncode != 0:
        write_json(output / "failure.json", {"stage": "R rare metaclustering", "exit_code": completed.returncode, "scientific_output_eligible": False})
        return 1

    r_summary = pd.read_csv(r_dir / "r_run_summary.csv")
    mappings = pd.read_csv(r_dir / "node_metacluster_mappings.csv")
    metric_rows, cluster_frames = [], []
    event_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    for row in r_summary.itertuples(index=False):
        selector_seed = int(row.selector_seed)
        subset = mappings[(mappings.dataset == row.dataset) & (mappings.som_seed == row.som_seed) & (mappings.regime == row.regime) & (mappings.selector_seed == selector_seed)].sort_values("node_index_0based")
        node_map = subset.metacluster_label_1based.to_numpy(dtype=np.int16)
        if node_map.shape != (100,): raise ValueError(f"mapping shape failed: {row.dataset}:{row.som_seed}:{row.regime}")
        event_labels = node_map[nodes_by_run[(row.dataset, int(row.som_seed))]]
        labels = labels_by_dataset[row.dataset]
        rare_metrics, cluster_frame = evaluate_rare(labels, event_labels, DATASETS[row.dataset][1])
        uses_label_k = row.regime == "official_fixed_ktrue_2"
        metric_rows.append({
            "dataset": row.dataset, "som_seed": int(row.som_seed), "regime": row.regime, "selector_seed": selector_seed,
            "requested_k": int(row.requested_k), "selected_k": int(row.selected_k),
            "occupied_node_metaclusters": int(row.occupied_node_metaclusters), "occupied_event_metaclusters": int(np.unique(event_labels).size),
            "r_runtime_seconds": float(row.r_runtime_seconds), "label_information_used_for_regime_definition": uses_label_k,
            "labels_used_by_r_metaclustering": False, "labels_used_only_for_posthoc_evaluation": True, **rare_metrics,
        })
        cluster_frames.append(cluster_frame.assign(dataset=row.dataset, som_seed=int(row.som_seed), regime=row.regime, selector_seed=selector_seed))
        event_arrays.setdefault((row.dataset, row.regime), []).append(event_labels.astype(np.int16, copy=False))
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "run_level_metrics.csv", index=False)
    pd.concat(cluster_frames, ignore_index=True).to_csv(output / "metacluster_target_diagnostics.csv", index=False)
    for (dataset, regime), arrays in event_arrays.items():
        np.savez_compressed(event_dir / f"{dataset}__{regime}.npz", partitions=np.stack(arrays))

    distributions = []
    for (dataset, regime), group in metrics_frame.groupby(["dataset", "regime"], sort=True):
        for metric in ("selected_k", *METRICS):
            values = group[metric].to_numpy(float)
            distributions.append({"dataset": dataset, "regime": regime, "metric": metric, "n": len(values), "mean": float(values.mean()), "sd": float(values.std(ddof=1)), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()), "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))})
    pd.DataFrame(distributions).to_csv(output / "distribution_summary.csv", index=False)

    paired_rows, contrast_rows = [], []
    rng = np.random.default_rng(20_260_911)
    main = metrics_frame[metrics_frame.regime.isin(MAIN_REGIMES)]
    for dataset in DATASETS:
        pivot = main[main.dataset == dataset].pivot(index="som_seed", columns="regime", values=list(METRICS))
        for comparator in ("official_fixed_ktrue_2", "official_fixed_k10", "official_fixed_k40"):
            for metric in METRICS:
                diffs = pivot[(metric, "official_auto_max40")].to_numpy() - pivot[(metric, comparator)].to_numpy()
                for seed, value in zip(pivot.index, diffs, strict=True):
                    paired_rows.append({"dataset": dataset, "contrast": f"auto_minus_{comparator}", "metric": metric, "som_seed": int(seed), "difference": float(value)})
                draws = rng.choice(diffs, size=(10_000, len(diffs)), replace=True).mean(axis=1)
                contrast_rows.append({"dataset": dataset, "contrast": f"auto_minus_{comparator}", "metric": metric, "n": len(diffs), "mean_difference": float(diffs.mean()), "sd_difference": float(diffs.std(ddof=1)), "bootstrap_ci_low": float(np.quantile(draws, .025)), "bootstrap_ci_high": float(np.quantile(draws, .975)), "bootstrap_replicates": 10_000})
    pd.DataFrame(paired_rows).to_csv(output / "paired_differences.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output / "paired_contrast_summary.csv", index=False)

    pair_rows = []
    for dataset in DATASETS:
        archive = np.load(event_dir / f"{dataset}__{SENSITIVITY_REGIME}.npz", allow_pickle=False)["partitions"]
        for i in range(30):
            for j in range(i + 1, 30):
                pair_rows.append({"dataset": dataset, "selector_seed_a": i, "selector_seed_b": j, "partition_ari": float(adjusted_rand_score(archive[i], archive[j]))})
    pair_frame = pd.DataFrame(pair_rows); pair_frame.to_csv(output / "selector_seed_pairwise_partition_ari.csv", index=False)

    checks: list[dict[str, object]] = []
    def check(name: str, passed: bool, detail: str) -> None: checks.append({"check": name, "passed": bool(passed), "detail": detail})
    r_checks = pd.read_csv(r_dir / "r_checks.csv")
    check("r_checks", len(r_checks) == 9 and bool(r_checks.passed.all()), f"{int(r_checks.passed.sum())}/{len(r_checks)}")
    check("row_counts", len(metrics_frame) == 300 and len(main) == 240, f"total={len(metrics_frame)}; main={len(main)}")
    check("regime_counts", all(len(group) == 30 for _, group in metrics_frame.groupby(["dataset", "regime"])), str(metrics_frame.groupby(["dataset", "regime"]).size().to_dict()))
    check("target_counts", set(metrics_frame[metrics_frame.dataset == "Nilsson_rare"].target_events) == {358} and set(metrics_frame[metrics_frame.dataset == "Mosmann_rare"].target_events) == {109}, "358;109")
    check("fixed_k_contract", set(metrics_frame[metrics_frame.regime == "official_fixed_ktrue_2"].selected_k) == {2} and set(metrics_frame[metrics_frame.regime == "official_fixed_k10"].selected_k) == {10} and set(metrics_frame[metrics_frame.regime == "official_fixed_k40"].selected_k) == {40}, "2/10/40")
    check("label_permission", metrics_frame.loc[metrics_frame.regime == "official_fixed_ktrue_2", "label_information_used_for_regime_definition"].all() and not metrics_frame.loc[metrics_frame.regime != "official_fixed_ktrue_2", "label_information_used_for_regime_definition"].any() and not metrics_frame.labels_used_by_r_metaclustering.any(), "only Ktrue regime definition uses label count")
    check("metric_ranges", metrics_frame[list(METRICS)].apply(lambda col: col.between(-1 if col.name == "ari" else 0, 1).all()).all(), "legal")
    check("partition_archives", len(list(event_dir.glob("*.npz"))) == 10 and all(np.load(path, allow_pickle=False)["partitions"].shape[0] == 30 for path in event_dir.glob("*.npz")), "10 x 30")
    check("paired_design", len(paired_rows) == 900 and len(contrast_rows) == 30, f"{len(paired_rows)};{len(contrast_rows)}")
    check("selector_pairs", len(pair_frame) == 870 and pair_frame.partition_ari.between(-1, 1).all(), str(len(pair_frame)))
    check("parent_hashes", len(parent_hashes) == 180 and all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"] for row in parent_hashes), str(len(parent_hashes)))
    checks_frame = pd.DataFrame(checks); checks_frame.to_csv(output / "checks.csv", index=False)
    result = {
        "experiment_id": args.experiment_id, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "r_script": str(r_script), "r_script_sha256": sha256(r_script), "rscript": str(args.rscript.resolve()),
        "parent": str(parent), "parent_progress_sha256": sha256(parent / "progress_manifest.json"),
        "parent_verification": str(parent_verification), "parent_verification_manifest_sha256": sha256(parent_verification / "run_manifest.json"),
        "r_flowsom_version": "2.18.0", "auto_max": 40, "fixed_metaclustering_seed": 12345,
        "main_rows": len(main), "selector_sensitivity_rows": int((metrics_frame.regime == SENSITIVITY_REGIME).sum()),
        "checks_passed": int(checks_frame.passed.sum()), "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame.passed.all()),
        "python": sys.version, "python_executable": sys.executable, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
    }
    write_json(output / "run_manifest.json", result)
    auto_counts = metrics_frame[metrics_frame.regime == "official_auto_max40"].groupby(["dataset", "selected_k"]).size().to_dict()
    (output / "scientific_summary.md").write_text(f"# {args.experiment_id} 稀有数据集FlowSOM元簇制度\n\n自动K计数：{auto_counts}。\n\n300套事件分区；检查{result['checks_passed']}/{result['checks_total']}。完整目标P/R/F1/F2及配对差见机器可读表。\n", encoding="utf-8")
    artifacts = [p for p in output.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"]
    write_json(output / "artifact_hashes.json", {str(p.relative_to(output)): sha256(p) for p in artifacts})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
