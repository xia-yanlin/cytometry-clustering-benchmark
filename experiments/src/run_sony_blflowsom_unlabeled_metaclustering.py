from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from cell_inclusion_kmeans import align_and_score
from sklearn.metrics import adjusted_rand_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean_ci(
    values: np.ndarray, seed: int = 20260911, n_boot: int = 10000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 1000):
        stop = min(start + 1000, n_boot)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rscript = args.rscript.resolve()
    r_library = args.r_library.resolve()
    parent = args.parent.resolve()
    data_path = args.data.resolve()
    protocol = args.protocol.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    r_output = output / "r_outputs"
    r_output.mkdir()

    parent_manifest = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    parent_summary = pd.read_csv(parent / "run_summary.csv").set_index("seed")
    input_rows = []
    parent_nodes: dict[int, np.ndarray] = {}
    parent_codes: dict[int, np.ndarray] = {}
    parent_hash_rows = []
    for seed in (1, 2):
        nodes_path = parent / f"seed_{seed}" / "nodes.csv"
        codes_path = parent / f"seed_{seed}" / "codes.csv"
        nodes = pd.read_csv(nodes_path, header=None).iloc[:, 0].to_numpy(dtype=np.int64)
        codes = pd.read_csv(codes_path, header=None).to_numpy(dtype=np.float64)
        if (
            nodes.shape != (167044,)
            or codes.shape != (100, 13)
            or nodes.min() < 1
            or nodes.max() > 100
        ):
            raise ValueError(f"parent shape contract failed for seed {seed}")
        expected_nodes = str(parent_summary.loc[seed, "clusters_sha256"])
        expected_codes = str(parent_summary.loc[seed, "codes_sha256"])
        if sha256(nodes_path) != expected_nodes or sha256(codes_path) != expected_codes:
            raise ValueError(f"parent hash contract failed for seed {seed}")
        parent_nodes[seed] = nodes
        parent_codes[seed] = codes
        input_rows.append(
            {
                "parent_seed": seed,
                "codes_path": str(codes_path.resolve()),
                "nodes_path": str(nodes_path.resolve()),
            }
        )
        for role, path in (("nodes", nodes_path), ("codes", codes_path)):
            parent_hash_rows.append(
                {
                    "parent_seed": seed,
                    "role": role,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(input_rows).to_csv(output / "r_input_manifest.csv", index=False)
    pd.DataFrame(parent_hash_rows).to_csv(output / "parent_input_hashes.csv", index=False)

    r_script = Path(__file__).resolve().with_name("run_sony_blflowsom_unlabeled_metaclustering.R")
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(r_library)
    env["LANG"] = "English_United States.1252"
    started = time.perf_counter()
    result = subprocess.run(
        [str(rscript), str(r_script), str(output / "r_input_manifest.csv"), str(r_output)],
        env=env,
        cwd=output,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    (output / "r_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "r_stderr.log").write_text(result.stderr, encoding="utf-8")
    (output / "r_execution.json").write_text(
        json.dumps(
            {
                "command": [
                    str(rscript),
                    str(r_script),
                    str(output / "r_input_manifest.csv"),
                    str(r_output),
                ],
                "exit_code": result.returncode,
                "runtime_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])

    r_checks = pd.read_csv(r_output / "r_checks.csv")
    r_summary = pd.read_csv(r_output / "r_run_summary.csv")
    mappings = pd.read_csv(r_output / "node_metacluster_mappings.csv")
    data = pd.read_csv(data_path, sep="\t")
    truth_numeric = pd.to_numeric(data["label"], errors="coerce")
    evaluable = (truth_numeric.between(1, 24) & truth_numeric.mod(1).eq(0)).to_numpy()
    y_true = truth_numeric[evaluable].astype(int).astype(str).to_numpy()

    run_rows = []
    population_rows = []
    partitions: dict[str, np.ndarray] = {}
    for row in r_summary.itertuples(index=False):
        subset = mappings[
            mappings.parent_seed.eq(row.parent_seed)
            & mappings.regime.eq(row.regime)
            & mappings.selector_seed.eq(row.selector_seed)
        ].sort_values("node_index_1based")
        node_map = subset.metacluster_label_1based.to_numpy(dtype=np.int16)
        if node_map.shape != (100,):
            raise ValueError(
                f"mapping shape failed: {row.parent_seed} {row.regime} {row.selector_seed}"
            )
        event_labels = node_map[parent_nodes[int(row.parent_seed)] - 1]
        key = f"parent{int(row.parent_seed)}__{row.regime}__selector{int(row.selector_seed):05d}"
        partitions[key] = event_labels.astype(np.int16, copy=False)
        metrics, per_pop = align_and_score(y_true, event_labels[evaluable].astype(str))
        base = {
            "dataset": "Levine_13dim",
            "parent_seed": int(row.parent_seed),
            "regime": row.regime,
            "selector_seed": int(row.selector_seed),
            "requested_k": int(row.requested_k),
            "selected_k": int(row.selected_k),
            "occupied_node_metaclusters": int(row.occupied_node_metaclusters),
            "occupied_event_metaclusters": int(np.unique(event_labels).size),
            "r_runtime_seconds": float(row.r_runtime_seconds),
            "label_information_used_for_regime_definition": bool(
                row.label_information_used_for_regime_definition
            ),
            "labels_used_by_r_metaclustering": bool(row.labels_used_by_r_metaclustering),
            "labels_used_only_for_posthoc_evaluation": True,
            "n_total_events": len(data),
            "n_evaluable_events": int(evaluable.sum()),
            **metrics,
        }
        run_rows.append(base)
        for item in per_pop:
            population_rows.append(
                {
                    "parent_seed": int(row.parent_seed),
                    "regime": row.regime,
                    "selector_seed": int(row.selector_seed),
                    **item,
                }
            )
    np.savez_compressed(output / "event_partitions.npz", **partitions)
    run_frame = pd.DataFrame(run_rows)
    pop_frame = pd.DataFrame(population_rows)
    run_frame.to_csv(output / "run_level_metrics.csv", index=False)
    pop_frame.to_csv(output / "population_level_metrics.csv", index=False)
    main_frame = run_frame[
        ~run_frame.regime.eq("official_auto_max40_selector_seed_sensitivity")
    ].copy()
    main_frame.to_csv(output / "main_regime_comparison.csv", index=False)

    selector = run_frame[
        run_frame.regime.eq("official_auto_max40_selector_seed_sensitivity")
    ].sort_values("selector_seed")
    summary_rows = []
    for metric in (
        "selected_k",
        "occupied_event_metaclusters",
        "ari",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "unmatched_cells",
    ):
        values = selector[metric].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values)
        summary_rows.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": np.median(values),
                "min": values.min(),
                "max": values.max(),
                "bootstrap_mean_ci_low": low,
                "bootstrap_mean_ci_high": high,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "selector_distribution_summary.csv", index=False)
    selector_partitions = [
        partitions[f"parent1__official_auto_max40_selector_seed_sensitivity__selector{seed:05d}"]
        for seed in range(30)
    ]
    pair_rows = []
    for i in range(30):
        for j in range(i + 1, 30):
            pair_rows.append(
                {
                    "selector_seed_a": i,
                    "selector_seed_b": j,
                    "partition_ari": adjusted_rand_score(
                        selector_partitions[i], selector_partitions[j]
                    ),
                }
            )
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output / "selector_pairwise_partition_ari.csv", index=False)

    exp028 = parent.parent / "EXP-028_blflowsom_r_metaclustering_qualification_levine13_20260910"
    anchor = pd.read_csv(exp028 / "evaluation_metrics.csv").iloc[0]
    ktrue = main_frame[
        (main_frame.parent_seed == 1) & main_frame.regime.eq("official_fixed_ktrue_24")
    ].iloc[0]
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check(
        "parent_manifest",
        parent_manifest["all_structural_checks_passed"],
        str(parent_manifest["experiment_id"]),
    )
    check(
        "parent_seed_outputs_identical",
        np.array_equal(parent_nodes[1], parent_nodes[2])
        and np.array_equal(parent_codes[1], parent_codes[2]),
        "nodes and codes exact",
    )
    check(
        "r_checks",
        len(r_checks) == 10 and r_checks.passed.astype(str).str.lower().eq("true").all(),
        f"{r_checks.passed.astype(str).str.lower().eq('true').sum()}/10",
    )
    check(
        "run_rows",
        len(run_frame) == 38 and len(main_frame) == 8 and len(selector) == 30,
        f"total={len(run_frame)} main={len(main_frame)} selector={len(selector)}",
    )
    check("population_rows", len(pop_frame) == 38 * 24, f"rows={len(pop_frame)}")
    check("pairwise_rows", len(pair_frame) == 435, f"rows={len(pair_frame)}")
    check(
        "event_partition_shapes",
        len(partitions) == 38 and all(value.shape == (167044,) for value in partitions.values()),
        f"partitions={len(partitions)}",
    )
    check(
        "label_permission",
        not run_frame.labels_used_by_r_metaclustering.any()
        and run_frame.label_information_used_for_regime_definition.eq(
            run_frame.regime.eq("official_fixed_ktrue_24")
        ).all(),
        "only Ktrue definition uses reference count",
    )
    check(
        "parent_seed_regime_partitions_exact",
        all(
            np.array_equal(
                partitions[f"parent1__{regime}__selector12345"],
                partitions[f"parent2__{regime}__selector12345"],
            )
            for regime in (
                "official_auto_max40",
                "official_fixed_k10",
                "official_fixed_k40",
                "official_fixed_ktrue_24",
            )
        ),
        "4/4 exact",
    )
    anchor_cols = [
        "ari",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "n_clusters",
        "unmatched_cells",
    ]
    anchor_diff = max(abs(float(ktrue[col]) - float(anchor[col])) for col in anchor_cols)
    check("exp028_ktrue_anchor", anchor_diff <= 1e-12, f"max_abs_diff={anchor_diff:.3e}")
    check(
        "selected_k_range",
        selector.selected_k.between(2, 40).all(),
        f"range={selector.selected_k.min()}-{selector.selected_k.max()}",
    )
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "checks.csv", index=False)

    k_counts = selector.selected_k.value_counts().sort_index().to_dict()
    main_compact = main_frame[main_frame.parent_seed.eq(1)][
        [
            "regime",
            "selected_k",
            "ari",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "unmatched_cells",
        ]
    ].to_dict(orient="records")
    manifest = {
        "experiment_id": "EXP-037",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "r_script": str(r_script),
        "r_script_sha256": sha256(r_script),
        "rscript": str(rscript),
        "rscript_sha256": sha256(rscript),
        "r_library": str(r_library),
        "r_flowsom_version": "2.18.0",
        "parent": str(parent),
        "parent_manifest_sha256": sha256(parent / "run_manifest.json"),
        "exp028_anchor": str(exp028),
        "exp028_manifest_sha256": sha256(exp028 / "run_manifest.json"),
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "main_rows": len(main_frame),
        "selector_rows": len(selector),
        "selector_k_counts": {str(k): int(v) for k, v in k_counts.items()},
        "main_seed1_results": main_compact,
        "checks_passed": int(checks_frame.passed.sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame.passed.all()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    auto = main_frame[
        (main_frame.parent_seed == 1) & main_frame.regime.eq("official_auto_max40")
    ].iloc[0]
    pair_values = pair_frame.partition_ari.to_numpy()
    (output / "scientific_summary.md").write_text(
        "# EXP-037 Unlabeled metaclustering for Sony BL-FlowSOM\n\n"
        f"Primary checks passed: {int(checks_frame.passed.sum())}/{len(checks_frame)}. Automatic max40 selected K={int(auto.selected_k)} at selector seed 12345, "
        f"ARI={auto.ari:.6f}, macro P/R/F1={auto.macro_precision:.6f}/{auto.macro_recall:.6f}/{auto.macro_f1:.6f}. "
        f"K counts across 30 selector seeds were {json.dumps(k_counts, ensure_ascii=False)}; the mean/range of pairwise event-partition ARI was "
        f"{pair_values.mean():.6f}/[{pair_values.min():.6f}, {pair_values.max():.6f}]. "
        "This result supplies an unlabeled endpoint for the hybrid Sony PoC plus R FlowSOM pipeline. It does not establish equivalence to Sony's cloud service and is not included in a cross-algorithm ranking.\n",
        encoding="utf-8",
    )
    artifacts = [path for path in output.rglob("*") if path.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {str(path.relative_to(output)): sha256(path) for path in artifacts},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
