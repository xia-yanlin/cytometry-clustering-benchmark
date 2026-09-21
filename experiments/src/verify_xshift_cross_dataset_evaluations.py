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

RUNS = {
    "Samusik_01": "EXP-011A_xshift_samusik_evaluation_20260910",
    "Nilsson_rare": "EXP-011B_xshift_nilsson_evaluation_20260910",
    "Levine_32dim": "EXP-011C_xshift_levine32_evaluation_20260910",
    "Mosmann_rare": "EXP-011D_xshift_mosmann_evaluation_20260910",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def truthy_csv(series: pd.Series) -> bool:
    return bool(series.astype(str).str.lower().eq("true").all())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    protocol = args.protocol.resolve()
    if not runs_root.is_dir() or not protocol.is_file():
        raise FileNotFoundError(f"Missing runs root or protocol: {runs_root}; {protocol}")

    checks: list[dict] = []
    summaries: list[dict] = []

    def check(dataset: str, name: str, passed: bool, detail: str) -> None:
        checks.append({"dataset": dataset, "check": name, "passed": bool(passed), "detail": detail})

    for dataset, run_name in RUNS.items():
        run = runs_root / run_name
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        evaluation_checks = pd.read_csv(run / "evaluation_checks.csv")
        metrics = pd.read_csv(run / "run_level_metrics.csv").iloc[0]
        contingency = pd.read_csv(run / "contingency_true_by_predicted.csv", index_col=0)
        cluster_diagnostics = pd.read_csv(run / "cluster_level_diagnostics.csv")
        indices = np.load(run / "evaluation_indices.npy", allow_pickle=False)

        check(
            dataset,
            "manifest_and_checks_pass",
            bool(manifest["all_checks_passed"]) and truthy_csv(evaluation_checks["passed"]),
            f"manifest={manifest['all_checks_passed']}, checks={len(evaluation_checks)}",
        )
        check(
            dataset,
            "contingency_sum",
            int(contingency.to_numpy().sum()) == int(metrics["n_evaluable_events"]),
            f"contingency={int(contingency.to_numpy().sum())}, metrics={int(metrics['n_evaluable_events'])}",
        )
        indices_valid = bool(
            len(indices) == int(metrics["n_evaluable_events"])
            and np.all(np.diff(indices) > 0)
            and int(indices[0]) >= 0
            and int(indices[-1]) < int(metrics["n_total_events"])
        )
        check(
            dataset,
            "evaluation_indices",
            indices_valid,
            f"n={len(indices)}, first={int(indices[0])}, last={int(indices[-1])}",
        )

        if str(metrics["evaluation_mode"]).startswith("multiclass"):
            populations = pd.read_csv(run / "population_level_metrics.csv")
            mapping = pd.read_csv(run / "hungarian_mapping.csv")
            cluster_total = int(cluster_diagnostics["all_event_count"].sum())
            structure_valid = bool(
                int(populations["support"].sum()) == int(metrics["n_evaluable_events"])
                and len(populations) == int(metrics["n_true_populations"])
                and len(mapping) == int(metrics["matched_clusters"])
                and contingency.shape
                == (
                    int(metrics["n_true_populations"]),
                    int(metrics["n_predicted_clusters_evaluable"]),
                )
                and cluster_total == int(metrics["n_total_events"])
            )
            detail = f"pop_support={int(populations['support'].sum())}, mapping={len(mapping)}, contingency={contingency.shape}, cluster_total={cluster_total}"
        else:
            target = str(metrics["target"])
            ordered = cluster_diagnostics.sort_values(
                ["f1", "recall", "predicted_cluster"],
                ascending=[False, False, True],
            )
            best = ordered.iloc[0]
            structure_valid = bool(
                int(cluster_diagnostics["cluster_size"].sum()) == int(metrics["n_total_events"])
                and int(cluster_diagnostics["tp"].sum()) == int(metrics["target_events"])
                and int(best["predicted_cluster"]) == int(metrics["selected_target_cluster"])
                and target in contingency.index.astype(str)
            )
            detail = f"cluster_total={int(cluster_diagnostics['cluster_size'].sum())}, target_tp_sum={int(cluster_diagnostics['tp'].sum())}, best={int(best['predicted_cluster'])}"
        check(dataset, "derived_tables_conserve_counts", structure_valid, detail)

        artifact_inventory = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
        artifact_failures = [
            path
            for path, expected_hash in artifact_inventory.items()
            if not Path(path).is_file() or sha256(Path(path)) != expected_hash
        ]
        check(dataset, "artifact_hashes", not artifact_failures, str(artifact_failures))
        summaries.append(
            {
                "dataset": dataset,
                "evaluation_mode": metrics["evaluation_mode"],
                "n_total_events": int(metrics["n_total_events"]),
                "n_evaluable_events": int(metrics["n_evaluable_events"]),
                "n_predicted_clusters": int(metrics["n_predicted_clusters_all_events"]),
                "ari": float(metrics["ari"]),
                "primary_f1": float(
                    metrics["macro_f1"]
                    if str(metrics["evaluation_mode"]).startswith("multiclass")
                    else metrics["target_f1"]
                ),
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "verification_checks.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "cross_dataset_metrics_summary.csv", index=False)
    manifest = {
        "experiment_id": "EXP-011V",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "verified_runs": RUNS,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": int(len(checks_frame)),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = [
        "# EXP-011V Independent verification of cross-dataset X-shift evaluations",
        "",
        f"Across four evaluation runs, {len(checks_frame)} conservation and hash checks were performed; {'all passed' if manifest['all_checks_passed'] else 'one or more failed'}.",
        "Verification covers manifests, original evaluation checks, contingency-table totals, evaluation indices, derived-table counts, and artifact SHA-256 hashes.",
    ]
    (output / "audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
