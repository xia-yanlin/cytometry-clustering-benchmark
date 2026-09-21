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
from sklearn.metrics import adjusted_rand_score

SOURCES = {
    ("Levine_32dim", "R_full_event"): ("EXP-046B-R1_flowsom_r_levine32_full_event_20260916", "raw"),
    ("Levine_32dim", "Python_20k_train"): ("EXP-021A_flowsom_levine32_stability30_20260910", "npy"),
    ("Samusik_01", "R_full_event"): ("EXP-046A-R1_flowsom_r_samusik_full_event_20260916", "raw"),
    ("Samusik_01", "Python_20k_train"): ("EXP-021B_flowsom_samusik_stability30_20260910", "npy"),
}
ENDPOINTS = [
    "ari",
    "macro_f1",
    "weighted_f1",
    "hungarian_accuracy",
    "median_bmu_distance",
    "runtime_seconds",
]
N_BOOT = 10_000
RNG_SEED = 20_260_916


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inventory_failures(path: Path) -> list[str]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for name, expected in inventory.items():
        target = Path(name)
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file() or sha256(target) != expected:
            failures.append(name)
    return failures


def read_labels(run_dir: Path, kind: str) -> np.ndarray:
    if kind == "raw":
        return np.fromfile(run_dir / "event_metacluster_labels.int32le", dtype="<i4")
    return np.load(run_dir / "metacluster_labels_all_events.npy", allow_pickle=False)


def describe(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "n": len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "median": np.median(values),
        "q1": q1,
        "q3": q3,
        "min": values.min(),
        "max": values.max(),
        "bootstrap_mean_ci95_low": np.quantile(boot, 0.025),
        "bootstrap_mean_ci95_high": np.quantile(boot, 0.975),
    }


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
    checks: list[dict] = []
    source_links: list[dict] = []
    point_frames: list[pd.DataFrame] = []
    labels: dict[tuple[str, str, int], np.ndarray] = {}

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    for (dataset, scheme), (run_name, kind) in SOURCES.items():
        run = root / run_name
        progress = json.loads((run / "progress_manifest.json").read_text(encoding="utf-8"))
        index = pd.read_csv(run / "run_index.csv").sort_values("seed").reset_index(drop=True)
        design_ok = len(index) == 30 and index["seed"].tolist() == list(range(30))
        design_ok = (
            design_ok and index["all_checks_passed"].astype(str).str.lower().eq("true").all()
        )
        design_ok = design_ok and progress.get("all_runs_complete_and_passed") is True
        check(
            f"{dataset}:{scheme}",
            "batch_contract",
            design_ok,
            f"rows={len(index)}; progress={progress.get('all_runs_complete_and_passed')}",
        )
        top_inventory = run / "artifact_hashes.json"
        top_failures = inventory_failures(top_inventory) if top_inventory.is_file() else []
        check(
            f"{dataset}:{scheme}",
            "top_level_hashes_or_legacy_format",
            not top_failures,
            f"inventory_present={top_inventory.is_file()}; failures={len(top_failures)}",
        )
        run_hashes: dict[str, str] = {}
        failures: list[str] = []
        for seed in range(30):
            seed_dir = run / "runs" / f"seed{seed:03d}"
            failures.extend(
                f"seed{seed:03d}:{item}"
                for item in inventory_failures(seed_dir / "artifact_hashes.json")
            )
            manifest_path = seed_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("seed") != seed or manifest.get("all_checks_passed") is not True:
                failures.append(f"seed{seed:03d}:manifest_contract")
            current = read_labels(seed_dir, kind)
            expected_n = int(index.loc[index["seed"] == seed, "n_total_events"].iloc[0])
            if len(current) != expected_n:
                failures.append(f"seed{seed:03d}:label_count={len(current)} expected={expected_n}")
            labels[(dataset, scheme, seed)] = current
            run_hashes[f"seed{seed:03d}"] = sha256(manifest_path)
        check(
            f"{dataset}:{scheme}",
            "run_hashes_manifests_and_labels",
            not failures,
            f"failures={len(failures)}",
        )
        normalized = index[
            [
                "seed",
                *ENDPOINTS,
                "n_total_events",
                "n_evaluable_events",
                "n_true_populations",
                "n_predicted_clusters_all_events",
            ]
        ].copy()
        normalized.insert(0, "source_run", run_name)
        normalized.insert(0, "scheme", scheme)
        normalized.insert(0, "dataset", dataset)
        point_frames.append(normalized)
        source_links.append(
            {
                "dataset": dataset,
                "scheme": scheme,
                "run": str(run),
                "progress_manifest_sha256": sha256(run / "progress_manifest.json"),
                "run_index_sha256": sha256(run / "run_index.csv"),
                "artifact_hashes_sha256": sha256(top_inventory)
                if top_inventory.is_file()
                else None,
                "run_manifest_sha256_by_seed": run_hashes,
            }
        )

    points = (
        pd.concat(point_frames, ignore_index=True)
        .sort_values(["dataset", "scheme", "seed"])
        .reset_index(drop=True)
    )
    points.to_csv(output / "all_run_points.csv", index=False)
    check("analysis", "run_point_count", len(points) == 120, f"rows={len(points)}")

    within_rows: list[dict] = []
    for dataset in sorted({key[0] for key in SOURCES}):
        for scheme in ["Python_20k_train", "R_full_event"]:
            for left in range(30):
                for right in range(left + 1, 30):
                    within_rows.append(
                        {
                            "dataset": dataset,
                            "scheme": scheme,
                            "seed_a": left,
                            "seed_b": right,
                            "partition_ari_all_events": adjusted_rand_score(
                                labels[(dataset, scheme, left)], labels[(dataset, scheme, right)]
                            ),
                        }
                    )
    within = pd.DataFrame(within_rows)
    within.to_csv(output / "within_scheme_pairwise_partition_ari.csv", index=False)
    check("analysis", "within_partition_pair_count", len(within) == 1740, f"rows={len(within)}")

    cross_rows: list[dict] = []
    for dataset in sorted({key[0] for key in SOURCES}):
        for seed in range(30):
            cross_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "partition_ari_r_full_vs_python_20k": adjusted_rand_score(
                        labels[(dataset, "R_full_event", seed)],
                        labels[(dataset, "Python_20k_train", seed)],
                    ),
                }
            )
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output / "same_seed_cross_scheme_partition_ari.csv", index=False)
    check("analysis", "cross_partition_pair_count", len(cross) == 60, f"rows={len(cross)}")

    rng = np.random.default_rng(RNG_SEED)
    descriptive_rows: list[dict] = []
    for (dataset, scheme), group in points.groupby(["dataset", "scheme"], sort=True):
        for endpoint in ENDPOINTS:
            descriptive_rows.append(
                {
                    "dataset": dataset,
                    "scheme": scheme,
                    "endpoint": endpoint,
                    **describe(group.sort_values("seed")[endpoint].to_numpy(float), rng),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(output / "endpoint_descriptive_statistics.csv", index=False)

    delta_rows: list[dict] = []
    for dataset in sorted(points["dataset"].unique()):
        subset = points.loc[points["dataset"] == dataset]
        r = subset.loc[subset["scheme"] == "R_full_event"].sort_values("seed")
        py = subset.loc[subset["scheme"] == "Python_20k_train"].sort_values("seed")
        for endpoint in ENDPOINTS:
            delta_rows.append(
                {
                    "dataset": dataset,
                    "contrast": "R_full_event_minus_Python_20k_train",
                    "endpoint": endpoint,
                    **describe(r[endpoint].to_numpy(float) - py[endpoint].to_numpy(float), rng),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output / "paired_seed_descriptive_differences.csv", index=False)

    within_summary = (
        within.groupby(["dataset", "scheme"])["partition_ari_all_events"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    within_summary.to_csv(output / "within_scheme_partition_ari_summary.csv", index=False)
    cross_summary = (
        cross.groupby("dataset")["partition_ari_r_full_vs_python_20k"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    cross_summary.to_csv(output / "cross_scheme_partition_ari_summary.csv", index=False)

    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(output / "analysis_checks.csv", index=False)
    all_passed = bool(check_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-046C",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_links": source_links,
        "endpoints": ENDPOINTS,
        "bootstrap_replicates": N_BOOT,
        "rng_seed": RNG_SEED,
        "checks_passed": int(check_frame["passed"].sum()),
        "checks_total": len(check_frame),
        "all_checks_passed": all_passed,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "interpretation_boundary": "implementation_version_and_training_scope_changed_together; descriptive_extrapolation_audit_only",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    key = deltas[deltas["endpoint"].isin(["ari", "macro_f1", "hungarian_accuracy"])][
        ["dataset", "endpoint", "mean", "bootstrap_mean_ci95_low", "bootstrap_mean_ci95_high"]
    ]
    stability = within_summary[["dataset", "scheme", "mean", "min", "max"]]
    agreement = cross_summary[["dataset", "mean", "min", "max"]]
    summary = [
        "# EXP-046C R FlowSOM full-event extrapolation analysis",
        "",
        "All 120 runs from the four source batches were included. Implementation, version, and training scope changed together, so the differences below are descriptive extrapolations and do not support single-factor attribution or cross-algorithm ranking.",
        "",
        "## Same-seed descriptive differences: R full-event minus Python 20,000-event training",
        "",
        key.to_csv(index=False).strip(),
        "",
        "## Within-method event-partition ARI across 30 seeds",
        "",
        stability.to_csv(index=False).strip(),
        "",
        "## Same-seed event-partition ARI across schemes",
        "",
        agreement.to_csv(index=False).strip(),
        "",
        "Runtime values are local execution records, not general performance benchmarks.",
    ]
    (output / "scientific_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    inventory = {str(path): sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "artifact_hashes.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
