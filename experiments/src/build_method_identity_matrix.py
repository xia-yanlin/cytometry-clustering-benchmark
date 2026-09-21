from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_rev(repository: Path, revision: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", revision],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-029")
    args = parser.parse_args()
    workspace, protocol, output = (
        args.workspace.resolve(),
        args.protocol.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    runs = workspace / "revision_experiments_20260910" / "runs"
    refs = workspace / "reference_implementations"
    archive = workspace / "archive" / "legacy_initial_analysis_20260910"

    evidence = {
        "xshift": load(runs / "EXP-007-R3_xshift_qualification_20260910" / "run_manifest.json"),
        "spade": load(runs / "EXP-008_spade_source_qualification_20260910" / "run_manifest.json"),
        "spade_source": load(
            runs / "EXP-008_spade_source_qualification_20260910" / "source_identity.json"
        ),
        "bl": load(
            runs
            / "EXP-028_blflowsom_r_metaclustering_qualification_levine13_20260910"
            / "run_manifest.json"
        ),
        "bl_verify": load(
            runs
            / "EXP-028V_blflowsom_r_metaclustering_independent_verification_20260910"
            / "run_manifest.json"
        ),
        "flowsom": load(
            runs / "EXP-013-R2_flowsom_python_v022_qualification_20260910" / "run_manifest.json"
        ),
        "phenograph": load(
            runs / "EXP-016-R2_phenograph_v157_louvain_qualification_20260910" / "run_manifest.json"
        ),
        "leiden": load(
            runs / "EXP-023_phenograph_leiden_path_qualification_20260910" / "run_manifest.json"
        ),
        "kmeans": load(
            runs / "EXP-003_cell_inclusion_kmeans_30seeds_20260910" / "run_manifest.json"
        ),
        "gmm": load(
            next(
                (runs / "EXP-022A_gmm_levine32_stability30_20260910" / "runs").glob(
                    "seed000/run_manifest.json"
                )
            )
        ),
    }
    rows = [
        {
            "method_label": "official_X-shift",
            "actual_execution_identity": "Nolan Lab Vortex standalone.Xshift, 29-Jun-2017 rev2",
            "source_or_package": "https://github.com/nolanlab/vortex.git",
            "commit_or_version": "fda75cf79980222da663185e2a6a72b442b9aff3",
            "qualification_evidence": "EXP-007-R3; EXP-007E; EXP-009-R2; EXP-010A/B/C-R1/D",
            "seed_interface": "used CLI exposes no seed",
            "metaclustering_or_postprocess": "native X-shift clusters; MST layout may be bypassed only after FCS cluster output validation",
            "label_information": "fixed/default K and auto elbow do not read labels; K=60 public anchor separate",
            "verified_endpoint_scope": "five datasets full-event clustering; four external evaluations",
            "current_use": "qualified fixed/default and named parameter regimes",
            "prohibited_inference": "two identical runs are not a 30-run random distribution",
            "main_ranking_ready": False,
        },
        {
            "method_label": "official_Deterministic-SPADE",
            "actual_execution_identity": "Qiu Lab Deterministic-SPADE MATLAB source/Windows standalone",
            "source_or_package": "https://github.com/pqiu/Deterministic-SPADE.git",
            "commit_or_version": evidence["spade_source"]["commit"],
            "qualification_evidence": "EXP-008 source qualification only",
            "seed_interface": "deterministic design",
            "metaclustering_or_postprocess": "official density downsampling, clustering, MST, upsampling",
            "label_information": "not yet tested end-to-end",
            "verified_endpoint_scope": "source components and artifact only; no event endpoint",
            "current_use": "blocked: MATLAB Runtime 8.5 absent",
            "prohibited_inference": "must not use archived Python substitute as SPADE",
            "main_ranking_ready": False,
        },
        {
            "method_label": "Sony_BL-FlowSOM_PoC_plus_R_MetaClust",
            "actual_execution_identity": "Sony batch SOM PoC followed by Sony eval MetaClust.R",
            "source_or_package": "sony/bl-flowsom_poc + sony/bl-flowsom_eval",
            "commit_or_version": f"PoC {git_rev(refs / 'bl-flowsom_poc')}; eval {git_rev(refs / 'bl-flowsom_eval')}; R FlowSOM 2.18.0",
            "qualification_evidence": "EXP-005-R2; EXP-006; EXP-028; EXP-028V",
            "seed_interface": "CLI seed logged but does not change PoC nodes/codes in tested path",
            "metaclustering_or_postprocess": "R metaClustering_consensus(k=24, seed=12345)",
            "label_information": "current event endpoint uses K=true=24 sensitivity",
            "verified_endpoint_scope": "Levine_13dim 167,044 events; node+metacluster labels",
            "current_use": "qualified identity/K=true endpoint only",
            "prohibited_inference": "no Sony cloud event truth; not cloud equivalence or deployable main setting",
            "main_ranking_ready": False,
        },
        {
            "method_label": "FlowSOM_Python",
            "actual_execution_identity": "Saeys Lab FlowSOM_Python v0.2.2",
            "source_or_package": evidence["flowsom"]["official_repo_remote"],
            "commit_or_version": evidence["flowsom"]["official_repo_commit"],
            "qualification_evidence": "EXP-013-R2; EXP-014A/B; EXP-021A/B",
            "seed_interface": "seed enters SOM initialization/training",
            "metaclustering_or_postprocess": "Python ConsensusCluster class currently returns agglomerative clustering at requested n_clusters",
            "label_information": "fixed 40 in main stability; K=true only in explicitly named reproduction/sensitivity",
            "verified_endpoint_scope": "two datasets 30 seeds/full-event mapping plus Levine_13 qualification",
            "current_use": "qualified for specified Python path",
            "prohibited_inference": "not stepwise equivalent to R consensus metaclustering or automatic K",
            "main_ranking_ready": False,
        },
        {
            "method_label": "PhenoGraph_default_Louvain",
            "actual_execution_identity": "Dana Pe'er Lab PhenoGraph v1.5.7 default Louvain community.exe",
            "source_or_package": evidence["phenograph"]["official_remote"],
            "commit_or_version": evidence["phenograph"]["official_commit"],
            "qualification_evidence": "EXP-016-R2; EXP-017-R1/V-R1; EXP-025A/B/C/V",
            "seed_interface": "official external Louvain interface receives no seed",
            "metaclustering_or_postprocess": "native Louvain on official Jaccard graph",
            "label_information": "fixed k values or historical label-selected scan are explicitly separated",
            "verified_endpoint_scope": "qualified path; Levine reproduction; two fixed 20k graphs x30 native repeats",
            "current_use": "qualified for default Louvain path only",
            "prohibited_inference": "full package has unresolved compatibility tests; repeat is not controlled seed",
            "main_ranking_ready": False,
        },
        {
            "method_label": "Leiden_fixed_PhenoGraph_graph",
            "actual_execution_identity": "PhenoGraph v1.5.7 Jaccard graph + leidenalg RBConfigurationVertexPartition",
            "source_or_package": evidence["leiden"]["official_remote"],
            "commit_or_version": f"PhenoGraph {evidence['leiden']['official_commit']}; leidenalg {evidence['leiden']['packages']['leidenalg']}",
            "qualification_evidence": "EXP-023; EXP-024A/B-R1/C/V",
            "seed_interface": "explicit Leiden seed 0-29",
            "metaclustering_or_postprocess": "fixed weighted graph, resolution=1, n_iterations=-1",
            "label_information": "no labels for graph or optimization; posthoc evaluation only",
            "verified_endpoint_scope": "two fixed 20,000-event training graphs",
            "current_use": "qualified experimental Leiden path",
            "prohibited_inference": "not the legacy Scanpy Leiden all-event implementation",
            "main_ranking_ready": False,
        },
        {
            "method_label": "sklearn_KMeans",
            "actual_execution_identity": "scikit-learn KMeans",
            "source_or_package": "scikit-learn",
            "commit_or_version": evidence["kmeans"]["packages"]["scikit-learn"],
            "qualification_evidence": "EXP-003/003A; EXP-004/004A-R1; EXP-026/V-R1",
            "seed_interface": "controlled random_state 0-29",
            "metaclustering_or_postprocess": "none",
            "label_information": "current unified stability uses K=true; explicit sensitivity only",
            "verified_endpoint_scope": "multi-population and rare datasets; full-event runs",
            "current_use": "qualified sensitivity method",
            "prohibited_inference": "K=true cannot represent deployable unsupervised selection",
            "main_ranking_ready": False,
        },
        {
            "method_label": "sklearn_GMM",
            "actual_execution_identity": "scikit-learn GaussianMixture full covariance",
            "source_or_package": "scikit-learn",
            "commit_or_version": evidence["gmm"]["sklearn_version"],
            "qualification_evidence": "EXP-022A/B/C/V",
            "seed_interface": "controlled random_state 0-29, n_init=1",
            "metaclustering_or_postprocess": "fixed 40 mixture components",
            "label_information": "K=40 fixed without labels; posthoc evaluation only",
            "verified_endpoint_scope": "two datasets, fixed 20k training then full-event prediction",
            "current_use": "qualified fixed-regime stability evidence",
            "prohibited_inference": "one fixed component count is not all GMM regimes",
            "main_ranking_ready": False,
        },
    ]
    legacy_specs = [
        (
            "ARCHIVED_custom_X-shift",
            archive / "02_extended" / "flow_gating_xshift.py",
            "custom inverse-kNN density climbing; not Nolan X-shift",
        ),
        (
            "ARCHIVED_custom_SPADE_style",
            archive / "02_extended" / "flow_gating_spade_v2.py",
            "random sampling and substitute clustering; no official MST identity",
        ),
        (
            "ARCHIVED_custom_BL-FlowSOM",
            archive / "01_baseline" / "flow_gating_BLFlowSOM.py",
            "author reimplementation without event/intermediate equivalence to Sony",
        ),
    ]
    for name, path, reason in legacy_specs:
        rows.append(
            {
                "method_label": name,
                "actual_execution_identity": reason,
                "source_or_package": str(path),
                "commit_or_version": sha256(path),
                "qualification_evidence": "legacy source audit; archived",
                "seed_interface": "not admissible for new evidence",
                "metaclustering_or_postprocess": "custom",
                "label_information": "historical contracts not inherited",
                "verified_endpoint_scope": "none in restart",
                "current_use": "archived and prohibited",
                "prohibited_inference": "must not be relabeled as the named official algorithm",
                "main_ranking_ready": False,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output / "method_identity_and_admissibility_matrix.csv", index=False)

    def markdown_cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    header = "| " + " | ".join(table.columns) + " |"
    separator = "|" + "|".join(["---"] * len(table.columns)) + "|"
    body = [
        "| " + " | ".join(markdown_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    markdown = [
        "# Method identity and current admissible scope",
        "",
        "This table is an internal evidence contract, not a manuscript table or algorithm ranking.",
        "",
        header,
        separator,
        *body,
        "",
    ]
    (output / "method_identity_and_admissibility_matrix.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check(
        "expected_rows", len(table) == 11 and table["method_label"].is_unique, f"rows={len(table)}"
    )
    check(
        "official_commits",
        git_rev(refs / "vortex", "29-Jun-2017^{}") == "fda75cf79980222da663185e2a6a72b442b9aff3"
        and git_rev(refs / "deterministic-spade") == evidence["spade_source"]["commit"]
        and git_rev(refs / "FlowSOM_Python-v0.2.2") == evidence["flowsom"]["official_repo_commit"]
        and git_rev(refs / "PhenoGraph") == evidence["phenograph"]["official_commit"],
        "all expected commits match",
    )
    check(
        "qualification_manifests",
        evidence["xshift"]["all_checks_passed"]
        and evidence["flowsom"]["all_checks_passed"]
        and evidence["phenograph"]["all_checks_passed"]
        and evidence["leiden"]["all_checks_passed"]
        and evidence["bl"]["all_checks_passed"]
        and evidence["bl_verify"]["all_checks_passed"]
        and evidence["kmeans"]["completed"]
        and evidence["gmm"]["all_checks_passed"],
        "all admitted paths have successful evidence",
    )
    check(
        "spade_boundary",
        evidence["spade"]["official_source_components_complete"]
        and not evidence["spade"]["official_standalone_runnable_here"]
        and not evidence["spade"]["legacy_candidate_method_identity_passed"],
        "source qualified; endpoint blocked; legacy rejected",
    )
    check(
        "legacy_archived",
        all(path.is_file() and archive in path.parents for _, path, _ in legacy_specs),
        "all three legacy sources remain in archive",
    )
    check(
        "scope_labels",
        "not the legacy Scanpy Leiden"
        in table.loc[
            table.method_label.eq("Leiden_fixed_PhenoGraph_graph"), "prohibited_inference"
        ].iloc[0]
        and "K=true"
        in table.loc[
            table.method_label.eq("Sony_BL-FlowSOM_PoC_plus_R_MetaClust"), "label_information"
        ].iloc[0],
        "Leiden and BL boundaries explicit",
    )
    check(
        "no_main_ranking_claim",
        not table["main_ranking_ready"].any(),
        "uniform main comparison not yet frozen",
    )
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "identity_checks.csv", index=False)
    source_files = [
        Path(value["script"])
        for key, value in evidence.items()
        if isinstance(value, dict) and "script" in value and Path(value["script"]).is_file()
    ]
    source_files += [path for _, path, _ in legacy_specs]
    (output / "source_hashes.json").write_text(
        json.dumps({str(path): sha256(path) for path in source_files}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "rows": len(table),
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
    (output / "scientific_summary.md").write_text(
        f"# {args.experiment_id} Method-identity audit\n\nThe registry contains {len(table)} identities (8 current reference paths and 3 archived prohibited paths). Checks passed: {manifest['checks_passed']}/{manifest['checks_total']}. No method is prematurely marked as ready for a unified primary ranking.\n",
        encoding="utf-8",
    )
    artifacts = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(
        json.dumps({str(path): sha256(path) for path in artifacts}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
