from __future__ import annotations

import argparse
import ast
import csv
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


SPECS = {
    "Levine_32dim": (DATA_ROOT / "Levine_32dim" / "Levine_32dim_notransform.txt", 265627, 39),
    "Samusik_01": (DATA_ROOT / "Samusik_01" / "Samusik_01_notransform.txt", 86864, 51),
    "Nilsson_rare": (DATA_ROOT / "Nilsson_rare" / "Nilsson_rare_notransform.csv", 44140, 19),
    "Mosmann_rare": (DATA_ROOT / "Mosmann_rare" / "Mosmann_rare_notransform.csv", 396460, 22),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-031V")
    args = parser.parse_args()
    workspace, protocol, parent, output = args.workspace.resolve(), args.protocol.resolve(), args.parent.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exp_root = workspace / "experiments"
    source_root = exp_root / "source_data" / "HDCytoData" / "remaining_four_ExperimentHub_Bioc3.22_20260910"
    r_verifier = exp_root / "src" / "verify_hdcytodata_remaining_four_exports.R"
    completed = subprocess.run([str(args.rscript.resolve()), str(r_verifier), str(source_root), str(output)], text=True, capture_output=True)
    (output / "r_verification_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "r_verification_stderr.log").write_text(completed.stderr, encoding="utf-8")

    checks: list[dict[str, object]] = []
    def check(dataset: str, name: str, passed: bool, detail: str) -> None: checks.append({"dataset": dataset, "check": name, "passed": bool(passed), "detail": detail})
    manifest = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    check("__global__", "parent_manifest", manifest.get("experiment_id") == "EXP-031" and manifest.get("all_checks_passed") is True, f"id={manifest.get('experiment_id')}; all={manifest.get('all_checks_passed')}")
    check("__global__", "r_export_verifier_exit", completed.returncode == 0, f"exit={completed.returncode}")
    r_checks = pd.read_csv(output / "r_export_verification_checks.csv")
    check("__global__", "r_export_checks", len(r_checks) == 16 and bool(r_checks["passed"].all()), f"passed={int(r_checks['passed'].sum())}/{len(r_checks)}")

    parent_summary = pd.read_csv(parent / "dataset_numeric_identity_summary.csv").set_index("dataset")
    independent_rows: list[dict[str, object]] = []
    for dataset, (local_path, rows, columns) in SPECS.items():
        directory = source_root / dataset
        official_columns = pd.read_csv(directory / "column_data.csv")["marker_name"].astype(str).tolist()
        official_labels = pd.read_csv(directory / "row_data.csv", low_memory=False)["population_id"].astype(str).to_numpy()
        authoritative = np.fromfile(directory / "expression_column_major_f64.bin", dtype="<f8").reshape((rows, columns), order="F")
        local = pd.read_csv(local_path, low_memory=False)
        local_expression = local.iloc[:, :-1].to_numpy(dtype=np.float64, copy=False)
        local_labels = local.iloc[:, -1].astype(str).to_numpy()
        delta = np.abs(authoritative - local_expression)
        values = {
            "dataset": dataset, "elements": int(delta.size), "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()), "exact_fraction": float(np.mean(delta == 0)),
            "within_5e_8_fraction": float(np.mean(delta <= 5e-8)), "label_mismatches": int(np.count_nonzero(official_labels != local_labels)),
        }
        independent_rows.append(values)
        reported = parent_summary.loc[dataset]
        check(dataset, "dimensions_and_columns", local_expression.shape == authoritative.shape == (rows, columns) and local.columns[:-1].astype(str).tolist() == official_columns, f"shape={authoritative.shape}; columns={len(official_columns)}")
        check(dataset, "numeric_tolerance", bool(np.all(delta <= 5e-8)), f"max={values['max_abs_delta']:.17g}")
        check(dataset, "labels_rowwise", values["label_mismatches"] == 0, f"mismatches={values['label_mismatches']}")
        numeric_match = all(np.isclose(float(values[name]), float(reported[name]), rtol=0, atol=1e-15) for name in ("max_abs_delta", "mean_abs_delta", "exact_fraction", "within_5e_8_fraction"))
        check(dataset, "parent_summary_reproduced", numeric_match and values["elements"] == int(reported["elements"]) and values["label_mismatches"] == int(reported["label_mismatches"]), "six reported quantities reproduced")
        del local, local_expression, authoritative, delta

    with (parent / "input_hashes.csv").open("r", encoding="utf-8-sig", newline="") as handle: input_rows = list(csv.DictReader(handle))
    bad_inputs = [row["path"] for row in input_rows if not Path(row["path"]).is_file() or sha256(Path(row["path"])) != row["sha256"]]
    check("__global__", "parent_input_hashes", not bad_inputs, f"verified={len(input_rows)}; bad={len(bad_inputs)}")
    artifacts = json.loads((parent / "artifact_hashes.json").read_text(encoding="utf-8"))
    bad_artifacts = [name for name, digest in artifacts.items() if not (parent / name).is_file() or sha256(parent / name) != digest]
    check("__global__", "parent_artifact_hashes", not bad_artifacts, f"verified={len(artifacts)}; bad={len(bad_artifacts)}")
    comparison_commit = subprocess.run(["git", "-C", str(workspace / "reference_implementations" / "cytometry-clustering-comparison"), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    hdcyto_commit = subprocess.run(["git", "-C", str(workspace / "reference_implementations" / "HDCytoData"), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    check("__global__", "source_commits", comparison_commit == "5b705de4e156652219765ccda005eed82c1d70e8" and hdcyto_commit == "ca8c9781c7b502430bc497d3c0beac16f2ca078f", f"{comparison_commit}; {hdcyto_commit}")
    verifier_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(verifier_tree):
        if isinstance(node, ast.Import): imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imported_modules.add(node.module)
    check("__global__", "source_independence", "audit_remaining_four_dataset_identity" not in imported_modules and "audit_remaining_four_dataset_identity.py" not in r_verifier.read_text(encoding="utf-8"), "main script neither imported nor called")

    pd.DataFrame(independent_rows).to_csv(output / "independent_dataset_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "verification_checks.csv", index=False)
    all_passed = all(bool(row["passed"]) for row in checks)
    out_manifest = {
        "experiment_id": args.experiment_id, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol), "parent": str(parent),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "r_verifier": str(r_verifier), "r_verifier_sha256": sha256(r_verifier),
        "checks_passed": sum(bool(row["passed"]) for row in checks), "checks_total": len(checks), "all_checks_passed": all_passed,
        "python": sys.version, "python_executable": sys.executable, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "scientific_summary.md").write_text(f"# EXP-031V 独立核验\n\nRDS—二进制逐值检查16/16；独立项目原件比较及来源/哈希检查{out_manifest['checks_passed']}/{out_manifest['checks_total']}通过。\n", encoding="utf-8")
    out_artifacts = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(json.dumps({path.name: sha256(path) for path in out_artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out_manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__": raise SystemExit(main())
