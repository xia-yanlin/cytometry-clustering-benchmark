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
from sklearn.mixture import GaussianMixture

RUNS = {
    "Levine_32dim": "EXP-021A_flowsom_levine32_stability30_20260910",
    "Samusik_01": "EXP-021B_flowsom_samusik_stability30_20260910",
}
ENDPOINTS = ["ari", "macro_f1", "hungarian_accuracy", "median_bmu_distance"]
N_BOOT_MEAN = 10_000
N_NULL = 5_000
N_POWER = 5_000
RNG_SEED = 20_260_910


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_inventory(path: Path) -> list[str]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    return [
        name
        for name, expected in inventory.items()
        if not Path(name).is_file() or sha256(Path(name)) != expected
    ]


def bic_difference(values: np.ndarray) -> tuple[float, GaussianMixture]:
    x = np.asarray(values, dtype=float).reshape(-1, 1)
    one = GaussianMixture(
        n_components=1, covariance_type="full", n_init=1, random_state=104729
    ).fit(x)
    two = GaussianMixture(
        n_components=2, covariance_type="full", n_init=20, random_state=104729
    ).fit(x)
    return float(one.bic(x) - two.bic(x)), two


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


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
    checks = []
    indices = []
    pairwise_rows = []
    source_links = []

    def check(scope: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    for dataset, name in RUNS.items():
        run = root / name
        progress = json.loads((run / "progress_manifest.json").read_text(encoding="utf-8"))
        index = pd.read_csv(run / "run_index.csv")
        balanced = (
            progress.get("all_runs_complete_and_passed") is True
            and len(index) == 30
            and set(index["seed"]) == set(range(30))
            and index["all_checks_passed"].astype(str).str.lower().eq("true").all()
        )
        check(
            dataset,
            "batch_and_seed_contract",
            balanced,
            f"rows={len(index)}; seeds={sorted(index['seed'].tolist())}",
        )
        hash_failures = []
        labels = {}
        manifest_hashes = {}
        for seed in range(30):
            run_dir = run / "runs" / f"seed{seed:03d}"
            hash_failures.extend(
                [
                    f"seed{seed:03d}:{item}"
                    for item in verify_inventory(run_dir / "artifact_hashes.json")
                ]
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("all_checks_passed") is not True or manifest.get("seed") != seed:
                hash_failures.append(f"seed{seed:03d}:manifest_contract")
            manifest_hashes[f"seed{seed:03d}"] = sha256(run_dir / "run_manifest.json")
            labels[seed] = np.load(
                run_dir / "metacluster_labels_all_events.npy", allow_pickle=False
            )
        check(dataset, "all_run_hashes_and_manifests", not hash_failures, json.dumps(hash_failures))
        for left in range(30):
            for right in range(left + 1, 30):
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "seed_a": left,
                        "seed_b": right,
                        "partition_ari_all_events": float(
                            adjusted_rand_score(labels[left], labels[right])
                        ),
                    }
                )
        index.insert(0, "source_run", name)
        indices.append(index)
        source_links.append(
            {
                "dataset": dataset,
                "run": str(run),
                "progress_manifest_sha256": sha256(run / "progress_manifest.json"),
                "run_index_sha256": sha256(run / "run_index.csv"),
                "run_manifest_sha256_by_seed": manifest_hashes,
            }
        )

    run_points = pd.concat(indices, ignore_index=True)
    run_points.to_csv(output / "all_run_points.csv", index=False)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(output / "pairwise_partition_ari.csv", index=False)
    check(
        "experiment",
        "pairwise_count",
        len(pairwise) == 870 and pairwise.groupby("dataset").size().eq(435).all(),
        str(pairwise.groupby("dataset").size().to_dict()),
    )

    rng = np.random.default_rng(RNG_SEED)
    descriptive_rows = []
    for dataset, group in run_points.groupby("dataset", sort=True):
        for endpoint in ENDPOINTS:
            values = group.sort_values("seed")[endpoint].to_numpy(float)
            boot_means = values[rng.integers(0, len(values), size=(N_BOOT_MEAN, len(values)))].mean(
                axis=1
            )
            q1, q3 = np.quantile(values, [0.25, 0.75])
            descriptive_rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "n_runs": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "mad_unscaled": float(np.median(np.abs(values - np.median(values)))),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": float(np.ptp(values)),
                    "bootstrap_mean_ci95_low": float(np.quantile(boot_means, 0.025)),
                    "bootstrap_mean_ci95_high": float(np.quantile(boot_means, 0.975)),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(output / "endpoint_descriptive_statistics.csv", index=False)

    null_statistics = np.empty(N_NULL)
    for iteration in range(N_NULL):
        null_statistics[iteration] = bic_difference(rng.normal(size=30))[0]
    np.save(output / "gmm_bic_null_statistics.npy", null_statistics)
    critical = float(np.quantile(null_statistics, 0.95))
    test_rows = []
    for dataset, group in run_points.groupby("dataset", sort=True):
        for endpoint in ENDPOINTS:
            values = group.sort_values("seed")[endpoint].to_numpy(float)
            statistic, mixture = bic_difference(values)
            means = mixture.means_.ravel()
            order = np.argsort(means)
            weights = mixture.weights_[order]
            pooled_sd = float(
                np.sqrt(np.average(mixture.covariances_.ravel()[order], weights=weights))
            )
            separation = (
                float((means[order[1]] - means[order[0]]) / pooled_sd) if pooled_sd > 0 else np.inf
            )
            pvalue = float((1 + np.sum(null_statistics >= statistic)) / (N_NULL + 1))
            test_rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "n_runs": 30,
                    "bic_one_minus_two": statistic,
                    "empirical_p_raw": pvalue,
                    "component_1_mean": float(means[order[0]]),
                    "component_2_mean": float(means[order[1]]),
                    "component_1_weight": float(weights[0]),
                    "component_2_weight": float(weights[1]),
                    "standardized_mean_separation": separation,
                }
            )
    tests = pd.DataFrame(test_rows)
    tests["empirical_p_holm"] = holm_adjust(tests["empirical_p_raw"].to_numpy(float))
    tests["reject_unimodal_gaussian_family_0_05"] = tests["empirical_p_holm"] <= 0.05
    tests.to_csv(output / "uniform_multimodality_tests.csv", index=False)
    check(
        "experiment",
        "uniform_test_family",
        len(tests) == 8
        and set(map(tuple, tests[["dataset", "endpoint"]].to_numpy()))
        == {(dataset, endpoint) for dataset in RUNS for endpoint in ENDPOINTS},
        f"rows={len(tests)}",
    )

    power_rows = []
    for separation in [1.0, 2.0, 3.0, 4.0]:
        rejections = 0
        statistics = []
        for _ in range(N_POWER):
            components = rng.integers(0, 2, size=30)
            sample = rng.normal(loc=(components - 0.5) * separation, scale=1.0)
            statistic = bic_difference(sample)[0]
            statistics.append(statistic)
            rejections += int(statistic > critical)
        power_rows.append(
            {
                "n": 30,
                "equal_component_weights": True,
                "within_component_sd": 1.0,
                "mean_separation_sd_units": separation,
                "null_critical_value_95pct": critical,
                "simulations": N_POWER,
                "empirical_power_uncorrected_single_test": rejections / N_POWER,
                "median_bic_difference": float(np.median(statistics)),
            }
        )
    power = pd.DataFrame(power_rows)
    power.to_csv(output / "multimodality_power_calibration.csv", index=False)
    check(
        "experiment",
        "power_contract",
        len(power) == 4
        and power["simulations"].eq(N_POWER).all()
        and power["empirical_power_uncorrected_single_test"].between(0, 1).all(),
        power.to_json(orient="records"),
    )

    pair_summary = (
        pairwise.groupby("dataset")["partition_ari_all_events"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    pair_summary["pairs"] = 435
    pair_summary.to_csv(output / "pairwise_partition_ari_summary.csv", index=False)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "analysis_checks.csv", index=False)
    all_passed = bool(checks_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-021C",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_links": source_links,
        "endpoints": ENDPOINTS,
        "n_runs_per_dataset": 30,
        "n_pairwise_partitions_per_dataset": 435,
        "mean_bootstrap_replicates": N_BOOT_MEAN,
        "multimodality_null_replicates": N_NULL,
        "multimodality_power_replicates_per_separation": N_POWER,
        "rng_seed": RNG_SEED,
        "familywise_method": "Holm",
        "familywise_alpha": 0.05,
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": all_passed,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    significant = tests.loc[
        tests["reject_unimodal_gaussian_family_0_05"], ["dataset", "endpoint", "empirical_p_holm"]
    ].to_dict(orient="records")
    lines = [
        "# EXP-021C FlowSOM 30-seed distribution analysis",
        "",
        f"Each dataset has 30 runs, producing 435 pairwise ARI values for each set of all-event partitions. Rejections after Holm correction across the unified family of eight GMM bootstrap tests: {json.dumps(significant, ensure_ascii=False)}.",
        f"Empirical single-test power at n=30 for ideal equal-weight two-normal mixtures: {json.dumps(dict(zip(power['mean_separation_sd_units'], power['empirical_power_uncorrected_single_test'], strict=False)), ensure_ascii=False)}.",
        "",
        "No subjective Stable/Variable/Unstable categories are used. The power simulation is an idealized calibration; a nonsignificant result is not evidence of unimodality. The 435 pairwise ARI values are dependent and were not used for multimodality significance tests.",
    ]
    (output / "scientific_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_inventory = {str(path): sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "artifact_hashes.json").write_text(
        json.dumps(write_inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
