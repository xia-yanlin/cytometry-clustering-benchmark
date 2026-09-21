"""Unified checks and reproducibility entry points for this repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = (
    ROOT / "analysis" / "scripts",
    ROOT / "posthoc" / "scripts",
    ROOT / "experiments" / "src",
    ROOT / "tools",
)
POSTHOC_TABLES = (
    "TableS7A_matching.csv",
    "TableS7B_ceiling.csv",
    "TableS7C_objectives.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_release() -> None:
    manifest = csv_rows(ROOT / "CODE_MANIFEST.csv")
    listed = {row["path"] for row in manifest}
    actual = {
        path.relative_to(ROOT).as_posix()
        for code_root in CODE_ROOTS
        for path in code_root.iterdir()
        if path.suffix in {".py", ".R"}
    }
    if len(manifest) != len(listed):
        raise RuntimeError("CODE_MANIFEST.csv contains duplicate paths")
    if listed != actual:
        missing = sorted(actual - listed)
        stale = sorted(listed - actual)
        raise RuntimeError(f"Code manifest mismatch; missing={missing}, stale={stale}")

    contract = csv_rows(ROOT / "data" / "input_manifest.csv")
    config = json.loads((ROOT / "experiments" / "config" / "datasets.json").read_text())
    dataset_rows = csv_rows(ROOT / "data" / "datasets.csv")
    contract_names = {row["dataset"] for row in contract}
    if len(contract) != 5 or contract_names != set(config["datasets"]):
        raise RuntimeError("The raw-input manifest and dataset configuration disagree")
    if contract_names != {row["dataset"] for row in dataset_rows}:
        raise RuntimeError("data/datasets.csv and data/input_manifest.csv disagree")
    for row in contract:
        spec = config["datasets"][row["dataset"]]
        if row["filename"] != spec["filename"] or row["label_policy"] != spec["label_policy"]:
            raise RuntimeError(f"Dataset contract mismatch: {row['dataset']}")
        if len(row["sha256"]) != 64 or int(row["bytes"]) <= 0:
            raise RuntimeError(f"Incomplete input identity: {row['dataset']}")

    provenance = csv_rows(ROOT / "RESULT_PROVENANCE.csv")
    for row in provenance:
        sources = list((ROOT / "analysis" / "inputs").glob(f"{row['analysis_input']}.*"))
        if len(sources) != 1:
            raise RuntimeError(f"Missing or ambiguous frozen input: {row['analysis_input']}")
        if not (ROOT / row["generator_script"]).is_file():
            raise RuntimeError(f"Missing generator: {row['generator_script']}")

    if len(list((ROOT / "analysis" / "figure_data").glob("*.csv"))) != 9:
        raise RuntimeError("Expected nine main-figure data files")
    if len(list((ROOT / "results" / "figures").glob("Figure_*.pdf"))) != 9:
        raise RuntimeError("Expected nine released figure PDFs")
    if len(list((ROOT / "results" / "tables").glob("*.csv"))) != 18:
        raise RuntimeError("Expected 18 released table parts")
    print(
        f"PASS: {len(manifest)} code files, {len(provenance)} frozen inputs, "
        "five raw-input contracts, nine figures and 18 table parts"
    )


def selected_contracts(datasets: list[str] | None) -> list[dict[str, str]]:
    records = csv_rows(ROOT / "data" / "input_manifest.csv")
    by_name = {row["dataset"]: row for row in records}
    names = datasets or list(by_name)
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def verify_inputs(data_root: Path, datasets: list[str] | None = None) -> None:
    data_root = data_root.resolve()
    failures = []
    for record in selected_contracts(datasets):
        path = data_root / record["dataset"] / record["filename"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        observed_size = path.stat().st_size
        if observed_size != int(record["bytes"]):
            failures.append(
                f"size mismatch: {path} (expected {record['bytes']}, observed {observed_size})"
            )
            continue
        observed_hash = sha256(path)
        if observed_hash != record["sha256"]:
            failures.append(f"SHA-256 mismatch: {path}")
            continue
        print(f"PASS input: {record['dataset']} ({observed_size} bytes)")
    if failures:
        raise RuntimeError("Raw-input verification failed:\n- " + "\n- ".join(failures))


def reproduce_paper() -> None:
    figure_data = sorted((ROOT / "analysis" / "figure_data").glob("*.csv"))
    before = {path.name: sha256(path) for path in figure_data}
    for script in (
        "build_main_figure_data.py",
        "render_main_figures.py",
        "render_supplement_figures.py",
        "build_tables.py",
    ):
        run([sys.executable, str(ROOT / "analysis" / "scripts" / script)])
    after = {path.name: sha256(path) for path in figure_data}
    if after != before:
        changed = sorted(name for name in before if before[name] != after.get(name))
        raise RuntimeError(f"Rebuilt figure data differ from the release: {changed}")
    if len(list((ROOT / "analysis" / "figures").glob("*.pdf"))) != 9:
        raise RuntimeError("Paper rebuild did not produce nine PDF figures")
    print("PASS: paper figure data are unchanged; nine figures and all tables rebuilt")


def reproduce_posthoc(output: Path | None) -> None:
    if output is None:
        with tempfile.TemporaryDirectory(prefix="cytometry-posthoc-") as temporary:
            reproduce_posthoc(Path(temporary))
        return
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(ROOT / "posthoc" / "scripts" / "matching_sensitivity.py"),
            "--source",
            str(ROOT / "posthoc" / "source"),
            "--out",
            str(output),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "posthoc" / "scripts" / "objective_selection.py"),
            "--source",
            str(ROOT / "posthoc" / "source"),
            "--out",
            str(output),
        ]
    )
    for name in POSTHOC_TABLES:
        rebuilt = output / name
        released = ROOT / "results" / "tables" / name
        if not rebuilt.is_file() or rebuilt.read_bytes() != released.read_bytes():
            raise RuntimeError(f"Post hoc table differs from the release: {name}")
    print("PASS: rebuilt Tables S7A-C match the released CSV files byte for byte")


def smoke_test(data_root: Path, dataset: str, output: Path, seeds: int) -> None:
    if output.exists():
        raise FileExistsError(f"Smoke-test output already exists: {output}")
    verify_inputs(data_root, [dataset])
    config = json.loads((ROOT / "experiments" / "config" / "datasets.json").read_text())
    config["data_root"] = str(data_root.resolve())
    with tempfile.TemporaryDirectory(prefix="cytometry-smoke-") as temporary:
        config_path = Path(temporary) / "datasets.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        run(
            [
                sys.executable,
                str(ROOT / "experiments" / "src" / "cell_inclusion_kmeans.py"),
                "--config",
                str(config_path),
                "--output",
                str(output.resolve()),
                "--datasets",
                dataset,
                "--seeds",
                str(seeds),
            ]
        )
    run_rows = csv_rows(output / "run_level_metrics.csv")
    if len(run_rows) != seeds * 2:
        raise RuntimeError(f"Expected {seeds * 2} smoke-test runs, found {len(run_rows)}")
    observed = csv_rows(output / "input_manifest.csv")
    expected_hash = selected_contracts([dataset])[0]["sha256"]
    if len(observed) != 1 or observed[0]["sha256"] != expected_hash:
        raise RuntimeError("Smoke-test input manifest does not match the verified raw input")
    if dataset == "Levine_13dim":
        expected_rows = csv_rows(ROOT / "experiments" / "expected" / "levine13_kmeans_seed0.csv")
        expected_by_policy = {row["fit_policy"]: row for row in expected_rows}
        observed_seed0 = {row["fit_policy"]: row for row in run_rows if row["seed"] == "0"}
        for policy, expected in expected_by_policy.items():
            observed_row = observed_seed0.get(policy)
            if observed_row is None:
                raise RuntimeError(f"Missing Levine_13dim seed-0 result: {policy}")
            for metric in ("ari", "macro_precision", "macro_recall", "macro_f1"):
                if abs(float(observed_row[metric]) - float(expected[metric])) > 1e-12:
                    raise RuntimeError(f"Seed-0 {policy} {metric} differs from the accepted run")
            if observed_row["iterations"] != expected["iterations"]:
                raise RuntimeError(f"Seed-0 {policy} iteration count differs from the accepted run")
        print("PASS: Levine_13dim seed-0 metrics match the accepted run")
        if seeds == 30:
            accepted_summary = {
                row["metric"]: row
                for row in csv_rows(ROOT / "analysis" / "inputs" / "kmeans_event_inclusion.csv")
                if row["dataset"] == dataset
            }
            for metric, accepted in accepted_summary.items():
                all_cells = [
                    float(row[metric]) for row in run_rows if row["fit_policy"] == "all_cells_fit"
                ]
                labeled_only = [
                    float(row[metric])
                    for row in run_rows
                    if row["fit_policy"] == "labeled_only_fit"
                ]
                deltas = [left - right for left, right in zip(all_cells, labeled_only, strict=True)]
                observed_statistics = {
                    "mean_all_cells_fit": statistics.fmean(all_cells),
                    "sd_all_cells_fit": statistics.stdev(all_cells),
                    "mean_labeled_only_fit": statistics.fmean(labeled_only),
                    "sd_labeled_only_fit": statistics.stdev(labeled_only),
                    "mean_delta_all_minus_labeled": statistics.fmean(deltas),
                    "median_delta_all_minus_labeled": statistics.median(deltas),
                    "sd_delta": statistics.stdev(deltas),
                    "all_cells_better_fraction": sum(value > 0 for value in deltas) / len(deltas),
                }
                for field, observed_value in observed_statistics.items():
                    if abs(observed_value - float(accepted[field])) > 1e-12:
                        raise RuntimeError(
                            f"30-seed Levine_13dim {metric} {field} differs from the release"
                        )
            print("PASS: Levine_13dim 30-seed summaries match the released analysis input")
    print(f"PASS: raw-data smoke test completed {len(run_rows)} paired-policy runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-release", help="check manifests and released artifacts")
    subparsers.add_parser("paper", help="rebuild manuscript figures and tables")
    posthoc = subparsers.add_parser("posthoc", help="rebuild and compare Table S7 analyses")
    posthoc.add_argument("--output", type=Path)
    subparsers.add_parser("release", help="run release, paper and post hoc checks")
    inputs = subparsers.add_parser(
        "verify-inputs", help="verify original raw data by size and hash"
    )
    inputs.add_argument("--data-root", type=Path, required=True)
    inputs.add_argument("--datasets", nargs="*")
    smoke = subparsers.add_parser("smoke", help="run a small raw-data-to-metrics check")
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument(
        "--dataset", choices=["Levine_13dim", "Levine_32dim", "Samusik_01"], default="Levine_13dim"
    )
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seeds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify-release":
        verify_release()
    elif args.command == "paper":
        reproduce_paper()
    elif args.command == "posthoc":
        reproduce_posthoc(args.output)
    elif args.command == "release":
        verify_release()
        reproduce_paper()
        reproduce_posthoc(None)
    elif args.command == "verify-inputs":
        verify_inputs(args.data_root, args.datasets)
    elif args.command == "smoke":
        if args.seeds < 1:
            raise ValueError("--seeds must be positive")
        smoke_test(args.data_root, args.dataset, args.output, args.seeds)


if __name__ == "__main__":
    main()
