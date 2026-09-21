from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import flowsom as fs
import numpy as np
import pandas as pd
from evaluate_xshift_cross_dataset import evaluate_multiclass, metrics_are_valid
from flowsom.models import map_data_to_codes

EXPECTED_DATA_SHA = "941c6328909f0abf3bd18801fd9ca95779356b225ba422c1f2daf0d2a737f354"
PAPER_ARI = 0.8936
PAPER_MACRO_F1 = 0.5356


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_hashes(directory: Path) -> None:
    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    ]
    (directory / "artifact_hashes.json").write_text(
        json.dumps({str(path): sha256(path) for path in paths}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--archive-data", type=Path, required=True)
    parser.add_argument("--historical-loader", type=Path, required=True)
    parser.add_argument("--historical-flowsom", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--qualification-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-015")
    args = parser.parse_args()

    data = args.data.resolve()
    archive_data = args.archive_data.resolve()
    historical_loader = args.historical_loader.resolve()
    historical_flowsom = args.historical_flowsom.resolve()
    historical_summary = args.historical_summary.resolve()
    qualification = args.qualification_run.resolve()
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    required = [
        data,
        archive_data,
        historical_loader,
        historical_flowsom,
        historical_summary,
        protocol,
        qualification / "run_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    (output / "runs").mkdir()

    checks: list[dict] = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    source_inventory = []
    for role, path in [
        ("formal_data", data),
        ("archived_data_copy", archive_data),
        ("historical_loader", historical_loader),
        ("historical_flowsom_script", historical_flowsom),
        ("historical_summary", historical_summary),
        ("protocol", protocol),
    ]:
        source_inventory.append(
            {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    pd.DataFrame(source_inventory).to_csv(output / "source_inventory.csv", index=False)

    data_hash = sha256(data)
    archive_hash = sha256(archive_data)
    check("source", "formal_data_expected_hash", data_hash == EXPECTED_DATA_SHA, data_hash)
    check(
        "source",
        "archive_copy_byte_identical",
        data_hash == archive_hash and data.stat().st_size == archive_data.stat().st_size,
        f"formal={data_hash}; archive={archive_hash}",
    )

    loader_text = historical_loader.read_text(encoding="utf-8")
    flowsom_text = historical_flowsom.read_text(encoding="utf-8")
    check(
        "source",
        "historical_loader_filters_before_return",
        "df = _filter_labeled(df, name)" in loader_text,
        "literal source audit",
    )
    check(
        "source",
        "historical_loader_skips_levine_transform",
        "'Levine_13dim': None" in loader_text,
        "literal source audit",
    )
    check(
        "source",
        "historical_flowsom_uses_true_k",
        "n_clusters=n_clusters" in flowsom_text and "k = TRUE_K[name]" in flowsom_text,
        "literal source audit",
    )
    check(
        "source",
        "historical_flowsom_grid_seed",
        "xdim=10" in flowsom_text and "ydim=10" in flowsom_text and "seed=42" in flowsom_text,
        "literal source audit",
    )

    qualification_manifest_path = qualification / "run_manifest.json"
    qualification_manifest = json.loads(qualification_manifest_path.read_text(encoding="utf-8"))
    check(
        "source",
        "qualified_official_flowsom",
        qualification_manifest.get("all_checks_passed") is True
        and qualification_manifest.get("flowsom_version") == "0.2.2",
        json.dumps(
            {
                key: qualification_manifest.get(key)
                for key in ("flowsom_version", "checks_passed", "checks_total")
            },
            ensure_ascii=False,
        ),
    )

    frame = pd.read_csv(data, sep="\t")
    markers = [column for column in frame.columns if column != "label"]
    labels_numeric = pd.to_numeric(frame["label"], errors="coerce").to_numpy()
    evaluable = (
        np.isfinite(labels_numeric)
        & (labels_numeric >= 1)
        & (labels_numeric <= 24)
        & (labels_numeric == np.floor(labels_numeric))
    )
    labels = np.array(
        [
            str(int(value)) if ok else "__unassigned__"
            for value, ok in zip(labels_numeric, evaluable, strict=False)
        ]
    )
    matrix_direct = frame[markers].to_numpy(np.float64)
    matrix_extra = np.arcsinh(matrix_direct / 5.0)
    check(
        "data",
        "dataset_contract",
        frame.shape == (167044, 14)
        and len(markers) == 13
        and int(evaluable.sum()) == 81747
        and len(np.unique(labels[evaluable])) == 24,
        f"shape={frame.shape}; markers={len(markers)}; evaluable={int(evaluable.sum())}; populations={len(np.unique(labels[evaluable]))}",
    )
    check(
        "data",
        "finite_matrices",
        np.isfinite(matrix_direct).all() and np.isfinite(matrix_extra).all(),
        f"direct={matrix_direct.min()}..{matrix_direct.max()}; extra={matrix_extra.min()}..{matrix_extra.max()}",
    )

    input_rows = []
    marker_rows = []
    for transform_name, matrix in [
        ("direct_file_values", matrix_direct),
        ("extra_asinh_cofactor5", matrix_extra),
    ]:
        for population_scope, mask in [
            ("all_events", np.ones(len(matrix), dtype=bool)),
            ("evaluable_events", evaluable),
        ]:
            subset = matrix[mask]
            input_rows.append(
                {
                    "transform": transform_name,
                    "scope": population_scope,
                    "events": len(subset),
                    "markers": subset.shape[1],
                    "minimum": float(subset.min()),
                    "maximum": float(subset.max()),
                    "mean": float(subset.mean()),
                    "sd": float(subset.std()),
                }
            )
        for marker_index, marker in enumerate(markers):
            values = matrix[:, marker_index]
            marker_rows.append(
                {
                    "transform": transform_name,
                    "marker_index": marker_index,
                    "marker": marker,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                }
            )
    pd.DataFrame(input_rows).to_csv(output / "input_stage_summary.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(output / "marker_stage_summary.csv", index=False)
    (output / "marker_order.json").write_text(
        json.dumps(markers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    conditions = [
        ("labeled_direct_a", matrix_direct, evaluable),
        ("labeled_direct_b", matrix_direct, evaluable),
        ("all_direct", matrix_direct, np.ones(len(matrix_direct), dtype=bool)),
        ("labeled_extra_asinh5", matrix_extra, evaluable),
        ("all_extra_asinh5", matrix_extra, np.ones(len(matrix_extra), dtype=bool)),
    ]
    result_rows = []
    core_outputs: dict[str, dict[str, np.ndarray]] = {}
    for condition, full_matrix, fit_mask in conditions:
        run_dir = output / "runs" / condition
        run_dir.mkdir()
        fit_matrix = np.asarray(full_matrix[fit_mask], dtype=np.float64)
        started = time.perf_counter()
        model = fs.FlowSOM(
            ad.AnnData(X=pd.DataFrame(fit_matrix, columns=markers)),
            n_clusters=24,
            cols_to_use=markers,
            xdim=10,
            ydim=10,
            rlen=10,
            seed=42,
        )
        codes = np.asarray(model.model.codes, dtype=np.float64)
        node_to_meta = np.asarray(model.model._y_codes, dtype=np.int16)
        node_float, distances = map_data_to_codes(fit_matrix, codes)
        node_labels = np.asarray(node_float, dtype=np.int16)
        meta_labels = np.asarray(node_to_meta[node_labels], dtype=np.int16)
        distances = np.asarray(distances, dtype=np.float32)
        runtime = time.perf_counter() - started

        if fit_mask is evaluable:
            evaluation_labels = meta_labels
        else:
            evaluation_labels = meta_labels[evaluable]
        metrics, population, clusters, mapping, contingency = evaluate_multiclass(
            labels[evaluable], evaluation_labels, meta_labels
        )
        valid_metrics, metric_detail = metrics_are_valid(metrics)

        arrays = {
            "node_labels_fit": node_labels,
            "metacluster_labels_fit": meta_labels,
            "bmu_distances_fit": distances,
            "som_codes": codes,
            "node_to_metacluster": node_to_meta,
            "metacluster_labels_evaluable": evaluation_labels,
        }
        for name, array in arrays.items():
            np.save(run_dir / f"{name}.npy", array)
        population.to_csv(run_dir / "population_level_metrics.csv", index=False)
        clusters.to_csv(run_dir / "cluster_level_metrics.csv", index=False)
        mapping.to_csv(run_dir / "hungarian_mapping.csv", index=False)
        contingency.to_csv(run_dir / "contingency_true_by_predicted.csv")

        run_checks = []

        def run_check(name: str, passed: bool, detail: str) -> None:
            run_checks.append({"check": name, "passed": bool(passed), "detail": detail})

        run_check(
            "fit_shape",
            node_labels.shape == meta_labels.shape == distances.shape == (int(fit_mask.sum()),),
            f"fit={int(fit_mask.sum())}; arrays={node_labels.shape}",
        )
        run_check(
            "model_shape",
            codes.shape == (100, 13) and node_to_meta.shape == (100,),
            f"codes={codes.shape}; mapping={node_to_meta.shape}",
        )
        run_check(
            "node_mapping",
            np.array_equal(meta_labels, node_to_meta[node_labels.astype(int)])
            and len(np.unique(node_to_meta)) == 24,
            f"node_metas={len(np.unique(node_to_meta))}; occupied={len(np.unique(meta_labels))}",
        )
        run_check(
            "evaluation_shape", evaluation_labels.shape == (81747,), str(evaluation_labels.shape)
        )
        run_check(
            "finite_outputs",
            np.isfinite(codes).all() and np.isfinite(distances).all() and np.all(distances >= 0),
            f"distance={float(distances.min())}..{float(distances.max())}",
        )
        run_check(
            "evaluation_contract",
            metrics["n_evaluable_events"] == 81747 and metrics["n_true_populations"] == 24,
            f"events={metrics['n_evaluable_events']}; populations={metrics['n_true_populations']}",
        )
        run_check("metric_ranges", valid_metrics, metric_detail)
        pd.DataFrame(run_checks).to_csv(run_dir / "qualification_checks.csv", index=False)
        check(
            condition,
            "run_contract",
            all(item["passed"] for item in run_checks),
            json.dumps(run_checks, ensure_ascii=False),
        )

        run_manifest = {
            "condition": condition,
            "fit_events": int(fit_mask.sum()),
            "evaluation_events": 81747,
            "markers": markers,
            "transform": "extra_arcsinh_cofactor5"
            if "extra_asinh5" in condition
            else "direct_file_values",
            "fit_policy": "evaluable_only" if condition.startswith("labeled") else "all_events",
            "parameters": {"xdim": 10, "ydim": 10, "rlen": 10, "n_clusters": 24, "seed": 42},
            "label_used_for_fit": bool(condition.startswith("labeled")),
            "n_clusters_label_informed": True,
            "runtime_seconds": runtime,
            "metrics": metrics,
            "checks_passed": int(sum(item["passed"] for item in run_checks)),
            "checks_total": len(run_checks),
            "all_checks_passed": bool(all(item["passed"] for item in run_checks)),
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        save_hashes(run_dir)
        core_outputs[condition] = arrays
        legacy_staged_macro_f1 = round(
            float(np.mean([round(float(value), 4) for value in population["f1"]])), 4
        )
        result_rows.append(
            {
                "condition": condition,
                "fit_events": int(fit_mask.sum()),
                "transform": run_manifest["transform"],
                "fit_policy": run_manifest["fit_policy"],
                "runtime_seconds": runtime,
                "occupied_nodes": int(len(np.unique(node_labels))),
                "occupied_metaclusters_fit": int(len(np.unique(meta_labels))),
                "occupied_metaclusters_evaluable": int(len(np.unique(evaluation_labels))),
                "ari": float(metrics["ari"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "legacy_staged_macro_f1": legacy_staged_macro_f1,
                "hungarian_accuracy": float(metrics["hungarian_accuracy"]),
                "delta_ari_from_paper": float(metrics["ari"] - PAPER_ARI),
                "delta_macro_f1_from_paper": float(metrics["macro_f1"] - PAPER_MACRO_F1),
            }
        )
        print(json.dumps(result_rows[-1], ensure_ascii=False), flush=True)

    exact_keys = [
        "node_labels_fit",
        "metacluster_labels_fit",
        "som_codes",
        "node_to_metacluster",
        "metacluster_labels_evaluable",
    ]
    exact_repeat = {
        key: bool(
            np.array_equal(
                core_outputs["labeled_direct_a"][key], core_outputs["labeled_direct_b"][key]
            )
        )
        for key in exact_keys
    }
    check(
        "reproduction",
        "historical_condition_same_seed_exact",
        all(exact_repeat.values()),
        json.dumps(exact_repeat),
    )
    results = pd.DataFrame(result_rows)
    results.to_csv(output / "condition_comparison.csv", index=False)

    historical = results[results.condition == "labeled_direct_a"].iloc[0]
    check(
        "reproduction",
        "paper_ari_four_decimal_match",
        round(float(historical.ari), 4) == PAPER_ARI,
        f"recomputed={historical.ari}; paper={PAPER_ARI}",
    )
    check(
        "reproduction",
        "paper_macro_f1_legacy_staged_rounding_match",
        float(historical.legacy_staged_macro_f1) == PAPER_MACRO_F1,
        f"full_precision={historical.macro_f1}; legacy_staged={historical.legacy_staged_macro_f1}; paper={PAPER_MACRO_F1}",
    )

    verification = pd.DataFrame(checks)
    verification.to_csv(output / "verification_checks.csv", index=False)
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    (output / "pip_check.txt").write_text(
        (pip_check.stdout or "") + (pip_check.stderr or ""), encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True
    )
    (output / "environment_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    check_frame = pd.read_csv(output / "verification_checks.csv")

    all_direct = results[results.condition == "all_direct"].iloc[0]
    labeled_extra = results[results.condition == "labeled_extra_asinh5"].iloc[0]
    audit = [
        f"# {args.experiment_id} Minimal reproduction audit for Levine_13dim FlowSOM",
        "",
        f"The final dataset and the first-draft copy are byte-identical, SHA-256={data_hash}.",
        f"Under the historical conditions (labeled events only, values used directly from file, 24 metaclusters, 10×10 grid, rlen=10, seed=42), the recomputed ARI is {historical.ari:.6f} and the full-precision macro F1 is {historical.macro_f1:.6f}. Averaging population-level F1 values after first rounding them, as in the initial draft, gives {historical.legacy_staged_macro_f1:.4f}, reproducing the displayed manuscript value 0.5356.",
        f"Changing only the fitting set to all events gives ARI={all_direct.ari:.6f} and macro F1={all_direct.macro_f1:.6f}.",
        f"Applying an additional arcsinh(x/5) transform to labeled events gives ARI={labeled_extra.ari:.6f} and macro F1={labeled_extra.macro_f1:.6f}.",
        "Both full-precision macro F1 and the historical staged-rounding value are retained. The latter explains only the final displayed digit and does not replace the full-precision calculation.",
        "These conditions localize reproducibility divergences; they do not determine whether the upstream data should be transformed again. Both 24-metacluster fitting and labeled-only fitting use label information and cannot serve as a new unlabeled primary regime.",
    ]
    (output / "audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "qualification_manifest": str(qualification_manifest_path),
        "qualification_manifest_sha256": sha256(qualification_manifest_path),
        "data_sha256": data_hash,
        "archive_data_sha256": archive_hash,
        "conditions": len(results),
        "checks_passed": int(check_frame["passed"].astype(str).str.lower().eq("true").sum()),
        "checks_total": int(len(check_frame)),
        "all_checks_passed": bool(check_frame["passed"].astype(str).str.lower().eq("true").all()),
        "pip_check_exit_code": pip_check.returncode,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_hashes(output)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] and pip_check.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
