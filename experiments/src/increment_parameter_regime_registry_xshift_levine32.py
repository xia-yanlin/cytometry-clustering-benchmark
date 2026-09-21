from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--verification-correction", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent = Path(args.parent).resolve()
    experiment = Path(args.experiment).resolve()
    verification = Path(args.verification).resolve()
    correction = Path(args.verification_correction).resolve()
    protocol = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    pm = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    em = json.loads((experiment / "run_manifest.json").read_text(encoding="utf-8"))
    vm = json.loads((verification / "run_manifest.json").read_text(encoding="utf-8"))
    cm = json.loads((correction / "run_manifest.json").read_text(encoding="utf-8"))
    if pm["experiment_id"] != "EXP-036-R6" or not pm["all_integrity_checks_passed"]:
        raise RuntimeError("bad R6 parent")
    if (
        em["experiment_id"] != "EXP-041"
        or em["runs_successful"] != 30
        or em["runs_failed"] != 0
        or em["unique_partition_hashes"] != 1
        or not em["all_checks_passed"]
    ):
        raise RuntimeError("bad EXP041")
    if (
        vm["experiment_id"] != "EXP-041V"
        or vm["checks_passed"] != 19
        or vm["checks_total"] != 20
        or vm["all_checks_passed"]
    ):
        raise RuntimeError("bad EXP041V failure evidence")
    if (
        cm["experiment_id"] != "EXP-041V-R1"
        or cm["checks_passed"] != cm["checks_total"]
        or cm["checks_total"] != 8
        or not cm["all_checks_passed"]
    ):
        raise RuntimeError("bad EXP041V-R1")

    evidence_hashes = json.loads((parent / "evidence_hashes.json").read_text(encoding="utf-8"))
    drift = [
        path
        for path, expected in evidence_hashes.items()
        if not Path(path).is_file() or sha256(Path(path)) != expected
    ]
    if drift:
        raise RuntimeError(f"evidence drift {drift}")
    registry = pd.read_csv(parent / "parameter_regime_registry.csv", dtype={"record_id": str})
    if len(registry) != 24 or "R25" in set(registry.record_id):
        raise RuntimeError("bad registry")
    row = {
        "record_id": "R25",
        "method_family": "official_X-shift",
        "execution_identity": "Nolan Vortex standalone.Xshift 29-Jun-2017 rev2",
        "datasets": "Levine_32dim",
        "regime": "main_no_label_fixed",
        "parameter_space": "K=20 fixed; 30 independent Java-process native repeats; MST bypass after validated cluster output",
        "candidate_count_per_dataset": 1,
        "repeats_per_candidate": "30 native repeats (not controlled seeds)",
        "total_observed_runs": 30,
        "random_interface": "CLI exposes no seed; source/JAR use default java.util.Random()",
        "training_policy": "all 265,627 events in every repeat",
        "label_permission": "labels excluded from clustering; 104,184 labeled events used posthoc only",
        "selection_objective": "none; fixed before evaluation",
        "tie_rule": "rectangular Hungarian posthoc",
        "failure_rule": "retain every attempted run and verification failure; MST bypass only after output/log dual guard",
        "reconstruction_evidence": f"{experiment.name}/run_level_metrics.csv;{verification.name}/run_manifest.json;{correction.name}/run_manifest.json",
        "same_regime_ranking_allowed": False,
        "gap": "30 native repeats cover 4/5 fixed-K datasets (Nilsson, Samusik, Levine_13dim, Levine_32dim); Mosmann remains single-run; cross-dataset K sensitivity remains missing",
    }
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry.to_csv(output / "parameter_regime_registry.csv", index=False)
    (output / "parameter_regime_registry.json").write_text(
        json.dumps(registry.to_dict("records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coverage = pd.read_csv(parent / "four_regime_coverage_matrix.csv")
    coverage.to_csv(output / "four_regime_coverage_matrix.csv", index=False)
    registry.groupby(["method_family", "regime"], as_index=False).agg(
        records=("record_id", "count"), observed_runs=("total_observed_runs", "sum")
    ).to_csv(output / "regime_inventory_summary.csv", index=False)

    fairness = pd.read_csv(parent / "fairness_checks.csv")
    fairness.loc[fairness.criterion.eq("main_fixed_repetition_budget_equal"), "observed"] = (
        "per-dataset repetitions remain unequal: 1, 2 exact parent repeats, or 30; X-shift has 30 native repeats for 4/5 fixed-K datasets"
    )
    fairness.loc[fairness.criterion.eq("main_repeat_contract_values_frozen"), "observed"] = (
        "observed main repeat budgets remain {1,2,30}; X-shift repeats are native, not controlled seeds"
    )
    fairness.to_csv(output / "fairness_checks.csv", index=False)
    gaps = pd.read_csv(parent / "fairness_gap_register.csv")
    mask = gaps.gap.str.startswith("Main no-label repeat budgets")
    gaps.loc[mask, "gap"] = (
        "Main no-label repeat budgets and coverage differ (X-shift 30-repeat on 4/5 datasets; Sony exact duplicate parents)"
    )
    gaps.loc[mask, "required_next_evidence"] = (
        "extend X-shift repeats to Mosmann and add cross-dataset K sensitivity or retain explicit dataset-specific uncertainty"
    )
    gaps.to_csv(output / "fairness_gap_register.csv", index=False)

    new_evidence = (
        experiment / "run_level_metrics.csv",
        verification / "run_manifest.json",
        correction / "run_manifest.json",
    )
    for path in new_evidence:
        evidence_hashes[str(path)] = sha256(path)
    (output / "evidence_hashes.json").write_text(
        json.dumps(evidence_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checks = pd.read_csv(parent / "audit_checks.csv")
    additions = pd.DataFrame(
        [
            {
                "check": "r6_parent_snapshot",
                "passed": True,
                "detail": f"records=24 manifest={sha256(parent / 'run_manifest.json')}",
            },
            {
                "check": "levine32_repeat_parent",
                "passed": em["runs_successful"] == 30
                and em["unique_partition_hashes"] == 1
                and em["pairwise_partition_ari_min"] == 1,
                "detail": "30/30 unique=1 pairwise_min=1",
            },
            {
                "check": "levine32_first_verification_preserved",
                "passed": vm["checks_passed"] == 19
                and vm["checks_total"] == 20
                and not vm["all_checks_passed"],
                "detail": "19/20; dtype false negative retained",
            },
            {
                "check": "levine32_formal_verification_correction",
                "passed": cm["checks_passed"] == cm["checks_total"] == 8,
                "detail": "R1=8/8",
            },
            {
                "check": "r25_scope_semantics",
                "passed": "4/5" in row["gap"]
                and "not controlled seeds" in row["repeats_per_candidate"]
                and "MST bypass" in row["parameter_space"],
                "detail": row["gap"],
            },
            {
                "check": "evidence_count",
                "passed": len(evidence_hashes) == 33,
                "detail": f"files={len(evidence_hashes)}",
            },
        ]
    )
    checks = pd.concat([checks, additions], ignore_index=True)
    checks.to_csv(output / "audit_checks.csv", index=False)
    (output / "scientific_summary.md").write_text(
        "# EXP-036-R7 Parameter-regime registry increment\n\nThe registry contains 25 records and 33 frozen evidence items. Thirty native X-shift repeats cover 4 of 5 fixed-K datasets. Mosmann still has one run, the interface exposes no seed, and cross-dataset K sensitivity remains incomplete. None of the seven method families covers all four regimes, so a unified ranking remains prohibited.\n",
        encoding="utf-8",
    )
    passed = bool(checks.passed.astype(str).str.lower().eq("true").all())
    manifest = {
        "experiment_id": "EXP-036-R7",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "parent": str(parent),
        "parent_manifest_sha256": sha256(parent / "run_manifest.json"),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "registry_records": len(registry),
        "eligible_method_families": 7,
        "families_with_all_four_regimes": int(
            coverage.all_four_present.astype(str).str.lower().eq("true").sum()
        ),
        "frozen_evidence_files": len(evidence_hashes),
        "audit_checks_passed": int(checks.passed.astype(str).str.lower().eq("true").sum()),
        "audit_checks_total": len(checks),
        "all_integrity_checks_passed": passed,
        "unified_cross_method_ranking_ready": False,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {
                p.name: sha256(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "artifact_hashes.json"
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
