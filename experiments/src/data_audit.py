from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def validate_separator(path: Path, registered: str) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    observed = "comma" if first_line.count(",") > first_line.count("\t") else "tab"
    if observed != registered:
        raise ValueError(
            f"Separator mismatch for {path}: registered={registered}, observed={observed}"
        )
    return "\t" if observed == "tab" else ","


def evaluable_mask(labels: pd.Series, policy: str) -> np.ndarray:
    if policy == "integer_1_to_24_evaluable":
        numeric = pd.to_numeric(labels, errors="coerce")
        return (numeric.between(1, 24) & (numeric % 1 == 0)).to_numpy()
    if policy == "unassigned_string_not_evaluable":
        text = labels.astype("string").str.strip().str.lower()
        return (~(text.isna() | text.eq("unassigned"))).to_numpy()
    if policy == "binary_target_and_other_all_evaluable":
        return np.ones(len(labels), dtype=bool)
    raise ValueError(f"Unknown label policy: {policy}")


def normalized_label(value: object, policy: str) -> str:
    if policy == "integer_1_to_24_evaluable":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "__unassigned__"
        if numeric.is_integer() and 1 <= numeric <= 24:
            return str(int(numeric))
        return "__unassigned__"
    text = str(value).strip()
    if policy == "unassigned_string_not_evaluable" and text.lower() == "unassigned":
        return "__unassigned__"
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=25000)
    parser.add_argument("--experiment-id", default="EXP-001")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    inventory_rows: list[dict] = []
    label_rows: list[dict] = []
    marker_rows: list[dict] = []
    failures: list[str] = []

    for dataset, spec in config["datasets"].items():
        path = data_root / dataset / spec["filename"]
        if not path.is_file():
            failures.append(f"{dataset}: missing {path}")
            continue

        sep = validate_separator(path, spec["separator"])
        header = pd.read_csv(path, sep=sep, nrows=0)
        columns = list(header.columns)
        if "label" not in columns:
            failures.append(f"{dataset}: label column missing")
            continue
        excluded = set(spec["exclude_columns"])
        marker_cols = [column for column in columns if column not in excluded]
        marker_min = {column: np.inf for column in marker_cols}
        marker_max = {column: -np.inf for column in marker_cols}
        marker_sum = {column: 0.0 for column in marker_cols}
        marker_n = {column: 0 for column in marker_cols}
        label_counts: Counter[str] = Counter()
        total_cells = 0
        evaluated_cells = 0

        for chunk in pd.read_csv(path, sep=sep, chunksize=args.chunksize):
            total_cells += len(chunk)
            mask = evaluable_mask(chunk["label"], spec["label_policy"])
            evaluated_cells += int(mask.sum())
            label_counts.update(
                normalized_label(value, spec["label_policy"]) for value in chunk["label"].tolist()
            )
            for marker in marker_cols:
                values = pd.to_numeric(chunk[marker], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    marker_min[marker] = min(marker_min[marker], float(finite.min()))
                    marker_max[marker] = max(marker_max[marker], float(finite.max()))
                    marker_sum[marker] += float(finite.sum())
                    marker_n[marker] += int(finite.size)

        marker_count_ok = len(marker_cols) == int(spec["expected_markers"])
        if not marker_count_ok:
            failures.append(
                f"{dataset}: expected {spec['expected_markers']} markers, observed {len(marker_cols)}"
            )

        inventory_rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "filename": path.name,
                "bytes": path.stat().st_size,
                "total_cells": total_cells,
                "evaluated_cells": evaluated_cells,
                "unassigned_cells": total_cells - evaluated_cells,
                "evaluated_fraction": evaluated_cells / total_cells,
                "columns": len(columns),
                "markers": len(marker_cols),
                "expected_markers": int(spec["expected_markers"]),
                "marker_count_ok": marker_count_ok,
                "input_transform": "already_transformed"
                if spec["cofactor"] is None
                else "untransformed",
                "analysis_transform": "none"
                if spec["cofactor"] is None
                else f"arcsinh(x/{spec['cofactor']})",
                "label_policy": spec["label_policy"],
            }
        )
        for label, count in sorted(label_counts.items()):
            label_rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "count": count,
                    "fraction": count / total_cells,
                    "evaluable": label != "__unassigned__",
                }
            )
        for marker in marker_cols:
            n = marker_n[marker]
            raw_min = marker_min[marker] if n else np.nan
            raw_max = marker_max[marker] if n else np.nan
            raw_mean = marker_sum[marker] / n if n else np.nan
            cofactor = spec["cofactor"]
            marker_rows.append(
                {
                    "dataset": dataset,
                    "marker": marker,
                    "finite_cells": n,
                    "missing_or_nonfinite": total_cells - n,
                    "raw_min": raw_min,
                    "raw_max": raw_max,
                    "raw_mean": raw_mean,
                    "analysis_min": raw_min
                    if cofactor is None
                    else float(np.arcsinh(raw_min / cofactor)),
                    "analysis_max": raw_max
                    if cofactor is None
                    else float(np.arcsinh(raw_max / cofactor)),
                }
            )

    pd.DataFrame(inventory_rows).to_csv(output / "data_inventory.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output / "label_summary.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(output / "marker_summary.csv", index=False)
    manifest = {
        "experiment_id": args.experiment_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "failures": failures,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        f"# {args.experiment_id} Data and label audit",
        "",
        "This report summarizes the input files, marker columns and reference labels used by the experiments.",
        "",
        f"Result: {'all expected structural checks passed' if not failures else 'one or more checks failed'}.",
        "",
        "## Automated checks",
        "",
    ]
    report.extend(
        [f"- {item}" for item in failures]
        or ["- All five files are present and their marker counts match the expected values."]
    )
    report.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "- This experiment validates local files and input structure; it does not by itself prove that the upstream public source can be retrieved again.",
            "- Source URLs, versions and transformation details should be recorded alongside the downloaded files.",
            "- The effect of filtering unlabeled cells before clustering is evaluated in a separate paired experiment.",
        ]
    )
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    exit_code = int(bool(failures))
    summary = {"output": str(output), "failures": failures, "exit_code": exit_code}
    (output / "stdout.log").write_text(
        json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "run_status.json").write_text(
        json.dumps({"exit_code": exit_code, "completed": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
