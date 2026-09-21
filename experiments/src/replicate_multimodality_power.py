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
import sklearn
from sklearn.mixture import GaussianMixture

N = 30
N_NULL = 5_000
N_POWER = 5_000
SEPARATIONS = [1.0, 2.0, 3.0, 4.0]
RNG_SEED = 20_260_916


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bic_difference(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1, 1)
    one = GaussianMixture(
        n_components=1, covariance_type="full", n_init=1, random_state=104729
    ).fit(x)
    two = GaussianMixture(
        n_components=2, covariance_type="full", n_init=20, random_state=104729
    ).fit(x)
    return float(one.bic(x) - two.bic(x))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-table", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_path, protocol, output = (
        args.original_table.resolve(),
        args.protocol.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    original = (
        pd.read_csv(original_path).sort_values("mean_separation_sd_units").reset_index(drop=True)
    )
    if (
        original["mean_separation_sd_units"].tolist() != SEPARATIONS
        or not original["simulations"].eq(N_POWER).all()
    ):
        raise RuntimeError("Original power table contract mismatch")

    rng = np.random.default_rng(RNG_SEED)
    null = np.empty(N_NULL)
    for iteration in range(N_NULL):
        null[iteration] = bic_difference(rng.normal(size=N))
        if (iteration + 1) % 500 == 0:
            print(f"null {iteration + 1}/{N_NULL}", flush=True)
    np.save(output / "independent_null_bic_statistics.npy", null)
    critical = float(np.quantile(null, 0.95))

    rows = []
    alternative_statistics = {}
    for separation in SEPARATIONS:
        statistics = np.empty(N_POWER)
        for iteration in range(N_POWER):
            components = rng.integers(0, 2, size=N)
            sample = rng.normal(loc=(components - 0.5) * separation, scale=1.0)
            statistics[iteration] = bic_difference(sample)
            if (iteration + 1) % 500 == 0:
                print(f"separation={separation:g} {iteration + 1}/{N_POWER}", flush=True)
        alternative_statistics[f"separation_{separation:g}"] = statistics
        rejections = int(np.sum(statistics > critical))
        power = rejections / N_POWER
        original_power = float(
            original.loc[
                original["mean_separation_sd_units"] == separation,
                "empirical_power_uncorrected_single_test",
            ].iloc[0]
        )
        pooled = (rejections + int(round(original_power * N_POWER))) / (2 * N_POWER)
        tolerance = 4 * np.sqrt(pooled * (1 - pooled) * (1 / N_POWER + 1 / N_POWER)) + 1 / N_POWER
        rows.append(
            {
                "n": N,
                "equal_component_weights": True,
                "within_component_sd": 1.0,
                "mean_separation_sd_units": separation,
                "independent_null_critical_value_95pct": critical,
                "simulations": N_POWER,
                "independent_rejections": rejections,
                "independent_power": power,
                "original_power": original_power,
                "absolute_power_difference": abs(power - original_power),
                "preregistered_monte_carlo_tolerance": tolerance,
                "compatible_with_original": abs(power - original_power) <= tolerance,
                "independent_median_bic_difference": float(np.median(statistics)),
            }
        )
    np.savez_compressed(
        output / "independent_alternative_bic_statistics.npz", **alternative_statistics
    )
    results = pd.DataFrame(rows)
    results.to_csv(output / "independent_power_replication.csv", index=False)

    checks = pd.DataFrame(
        [
            {
                "check": "null_contract",
                "passed": len(null) == N_NULL and np.isfinite(null).all(),
                "detail": f"n={len(null)}; critical={critical}",
            },
            {
                "check": "alternative_contract",
                "passed": len(results) == 4 and results["simulations"].eq(N_POWER).all(),
                "detail": f"rows={len(results)}",
            },
            {
                "check": "integer_rejection_counts",
                "passed": np.allclose(
                    results["independent_power"] * N_POWER,
                    results["independent_rejections"],
                    rtol=0,
                    atol=1e-12,
                ),
                "detail": results[["mean_separation_sd_units", "independent_rejections"]].to_json(
                    orient="records"
                ),
            },
            {
                "check": "monte_carlo_compatibility",
                "passed": results["compatible_with_original"].all(),
                "detail": results[
                    [
                        "mean_separation_sd_units",
                        "absolute_power_difference",
                        "preregistered_monte_carlo_tolerance",
                    ]
                ].to_json(orient="records"),
            },
        ]
    )
    checks.to_csv(output / "checks.csv", index=False)
    all_passed = bool(checks["passed"].all())
    manifest = {
        "experiment_id": "EXP-047",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "original_table": str(original_path),
        "original_table_sha256": sha256(original_path),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "n": N,
        "null_replicates": N_NULL,
        "power_replicates_per_separation": N_POWER,
        "rng_seed": RNG_SEED,
        "null_critical_value_95pct": critical,
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
        "all_checks_passed": all_passed,
        "python": sys.version,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# EXP-047 Independent rerun of the power simulation",
        "",
        f"Independent null-distribution critical value={critical:.6f}.",
        "",
        results.to_csv(index=False).strip(),
        "",
        "This calibration applies only to an ideal equal-weight Gaussian mixture with n=30. Failure to reject is not evidence of unimodality.",
    ]
    (output / "scientific_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {str(path): sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "artifact_hashes.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
