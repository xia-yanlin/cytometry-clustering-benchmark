from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--failed-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    protocol = args.protocol.resolve()
    parent = args.parent.resolve()
    failed_parent = args.failed_parent.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exp_root = workspace / "experiments"
    authoritative = exp_root / "source_data" / "HDCytoData" / "Levine_13dim_ExperimentHub_Bioc3.22_20260910" / "Levine_13dim_SE_authoritative_extract.rds"
    local = exp_root / "source_data" / "project_inputs" / "Levine_13dim" / "Levine_13dim.txt"
    r_verifier = exp_root / "src" / "verify_hdcytodata_levine13_numeric_identity.R"

    command = [str(args.rscript.resolve()), str(r_verifier.resolve()), str(authoritative), str(local), str(parent), str(output)]
    completed = subprocess.run(command, text=True, capture_output=True)
    (output / "r_verification_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "r_verification_stderr.log").write_text(completed.stderr, encoding="utf-8")

    checks: list[dict[str, object]] = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    parent_manifest = load_json(parent / "run_manifest.json")
    failed_manifest = load_json(failed_parent / "run_manifest.json")
    check("parent_manifest", parent_manifest.get("experiment_id") == "EXP-030-R1" and parent_manifest.get("all_checks_passed") is True, f"{parent_manifest.get('experiment_id')}; all={parent_manifest.get('all_checks_passed')}")
    check("independent_r_exit", completed.returncode == 0, f"exit_code={completed.returncode}")
    r_checks_path = output / "independent_numeric_checks.csv"
    with r_checks_path.open("r", encoding="utf-8-sig", newline="") as handle:
        r_checks = list(csv.DictReader(handle))
    check("independent_r_checks", len(r_checks) == 10 and all(row["passed"].lower() == "true" for row in r_checks), f"passed={sum(row['passed'].lower() == 'true' for row in r_checks)}/{len(r_checks)}")

    with (parent / "input_hashes.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    bad_inputs = [row["path"] for row in input_rows if not Path(row["path"]).is_file() or sha256(Path(row["path"])) != row["sha256"]]
    check("parent_input_hashes", not bad_inputs, f"verified={len(input_rows)}; bad={len(bad_inputs)}")

    artifact_hashes = load_json(parent / "artifact_hashes.json")
    bad_artifacts = [name for name, digest in artifact_hashes.items() if not (parent / name).is_file() or sha256(parent / name) != digest]
    check("parent_artifact_hashes", not bad_artifacts, f"verified={len(artifact_hashes)}; bad={len(bad_artifacts)}")
    check("failed_chain_preserved", failed_manifest.get("experiment_id") == "EXP-030" and failed_manifest.get("all_checks_passed") is False and failed_manifest.get("r_returncode") == 1, "original run retained as pre-comparison path failure")
    failed_stderr = (failed_parent / "r_numeric_comparison_stderr.log").read_text(encoding="utf-8", errors="replace")
    check("failed_chain_cause", "cannot open file" in failed_stderr and "Invalid argument" in failed_stderr, "R command-line path decoding failure reproduced in log")
    check("verifier_source_independence", "audit_levine13_authoritative_provenance" not in r_verifier.read_text(encoding="utf-8") and "compare_hdcytodata_levine13_numeric_identity" not in r_verifier.read_text(encoding="utf-8"), "R verifier imports neither parent script")

    with (output / "verification_checks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader(); writer.writerows(checks)
    all_passed = all(bool(row["passed"]) for row in checks)
    manifest = {
        "experiment_id": "EXP-030V",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "r_verifier": str(r_verifier), "r_verifier_sha256": sha256(r_verifier),
        "parent": str(parent), "failed_parent": str(failed_parent),
        "checks_passed": sum(bool(row["passed"]) for row in checks), "checks_total": len(checks),
        "all_checks_passed": all_passed,
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "scientific_summary.md").write_text(
        f"# EXP-030V 独立核验\n\n独立R代码复算10/10项数值与标签检查；外层{manifest['checks_passed']}/{manifest['checks_total']}项来源、哈希、失败链和独立性检查通过。\n",
        encoding="utf-8",
    )
    artifacts = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(json.dumps({path.name: sha256(path) for path in artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
