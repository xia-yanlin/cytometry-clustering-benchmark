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


METRICS = ["ari", "macro_f1", "median_bmu_distance_per_sqrt_marker"]
DIMS = [8, 13, 20, 26, 32]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--accepted-verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent, accepted, protocol, output = (args.parent_run.resolve(), args.accepted_verification.resolve(), args.protocol.resolve(), args.output.resolve())
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    progress = json.loads((parent / "progress_manifest.json").read_text(encoding="utf-8"))
    accepted_manifest = json.loads((accepted / "run_manifest.json").read_text(encoding="utf-8"))
    index = pd.read_csv(parent / "run_index.csv")
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("parent_complete", progress.get("experiment_id") == "EXP-018-R1" and progress.get("all_runs_complete_and_passed") is True, json.dumps({key: progress.get(key) for key in ("experiment_id", "status", "completed_runs", "passed_runs")}))
    check("accepted_verification", accepted_manifest.get("experiment_id") == "EXP-018V-R1" and accepted_manifest.get("all_checks_passed") is True, json.dumps({key: accepted_manifest.get(key) for key in ("experiment_id", "accepted_parent_checks_passed", "all_checks_passed")}))
    check("parent_links", accepted_manifest.get("parent_progress_manifest_sha256") == sha256(parent / "progress_manifest.json") and accepted_manifest.get("parent_run_index_sha256") == sha256(parent / "run_index.csv"), f"progress={sha256(parent / 'progress_manifest.json')}; index={sha256(parent / 'run_index.csv')}")
    balanced = len(index) == 125 and index.groupby(["chain", "n_markers"]).size().eq(5).all() and set(index["n_markers"]) == set(DIMS)
    check("balanced_design", balanced, f"rows={len(index)}; cells={index.groupby(['chain','n_markers']).size().to_dict()}")

    summary_rows = []
    for (chain, dim), group in index.groupby(["chain", "n_markers"], sort=True):
        row = {"chain": int(chain), "n_markers": int(dim), "n_seeds": len(group)}
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            row.update({f"{metric}_mean": float(values.mean()), f"{metric}_sd": float(values.std(ddof=1)), f"{metric}_median": float(np.median(values)), f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max()), f"{metric}_range": float(values.max() - values.min())})
        summary_rows.append(row)
    cell_summary = pd.DataFrame(summary_rows)
    cell_summary.to_csv(output / "chain_dimension_seed_summary.csv", index=False)
    check("cell_summary", len(cell_summary) == 25 and cell_summary["n_seeds"].eq(5).all(), f"rows={len(cell_summary)}")

    full32 = cell_summary[cell_summary.n_markers == 32]
    full32_columns = [column for column in cell_summary.columns if column not in ("chain", "n_markers")]
    full32_exact = all(full32[column].nunique(dropna=False) == 1 for column in full32_columns)
    check("full32_shared_summary_exact", full32_exact, full32.to_json(orient="records"))

    dimension_rows = []
    for dim in DIMS:
        cells = cell_summary[cell_summary.n_markers == dim]
        if dim == 32:
            cells = cells.iloc[[0]]
        row = {"n_markers": dim, "n_marker_chains_contributing": len(cells), "full32_collapsed_to_one_shared_reference": dim == 32}
        for metric in METRICS:
            for level in ("mean", "sd"):
                values = cells[f"{metric}_{level}"].to_numpy(float)
                row.update({f"{metric}_{level}_across_chain_mean": float(values.mean()), f"{metric}_{level}_across_chain_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0, f"{metric}_{level}_across_chain_min": float(values.min()), f"{metric}_{level}_across_chain_max": float(values.max())})
        dimension_rows.append(row)
    pd.DataFrame(dimension_rows).to_csv(output / "dimension_hierarchical_summary.csv", index=False)

    contrast_rows = []
    for dim in DIMS[:-1]:
        for metric in METRICS:
            reference_sd = float(full32.iloc[0][f"{metric}_sd"])
            values = cell_summary.loc[cell_summary.n_markers == dim, f"{metric}_sd"].to_numpy(float) - reference_sd
            contrast_rows.append({"target_dim": dim, "reference_dim": 32, "metric": metric, "quantity": "within_chain_seed_sd", "analysis_unit": "marker_chain", "n_chains": len(values), "reference_sd_shared": reference_sd, "mean_difference_target_minus_reference": float(values.mean()), "sd_difference": float(values.std(ddof=1)), "min_difference": float(values.min()), "max_difference": float(values.max()), "positive_chains": int((values > 0).sum()), "negative_chains": int((values < 0).sum()), "zero_chains": int((values == 0).sum())})
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "seed_sd_chain_contrasts.csv", index=False)
    check("contrast_set", len(contrasts) == 12 and contrasts["n_chains"].eq(5).all(), f"rows={len(contrasts)}")

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "analysis_checks.csv", index=False)
    ari13 = contrasts[(contrasts.target_dim == 13) & (contrasts.metric == "ari")].iloc[0]
    f113 = contrasts[(contrasts.target_dim == 13) & (contrasts.metric == "macro_f1")].iloc[0]
    (output / "scientific_summary.md").write_text(
        "# EXP-018A 维度与运行间稳定性\n\n"
        f"派生分析检查{int(checks_frame['passed'].sum())}/{len(checks_frame)}。13维相对共享32维，五条marker链的ARI seed-SD平均差={ari13['mean_difference_target_minus_reference']:.6f}（正/负={int(ari13['positive_chains'])}/{int(ari13['negative_chains'])}）；Macro F1 seed-SD平均差={f113['mean_difference_target_minus_reference']:.6f}（正/负={int(f113['positive_chains'])}/{int(f113['negative_chains'])}）。\n\n"
        "五seed只能描述运行间离散，不能支持双峰检验或稳定性等级。marker链不是生物重复；这些结果不提供通用高维失败概率。\n",
        encoding="utf-8",
    )
    all_passed = bool(checks_frame["passed"].all())
    manifest = {
        "experiment_id": "EXP-018A", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run": str(parent), "parent_progress_manifest_sha256": sha256(parent / "progress_manifest.json"), "parent_run_index_sha256": sha256(parent / "run_index.csv"),
        "accepted_verification": str(accepted), "accepted_verification_manifest_sha256": sha256(accepted / "run_manifest.json"),
        "protocol": str(protocol), "protocol_sha256": sha256(protocol), "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame), "all_checks_passed": all_passed,
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths = [path for path in output.iterdir() if path.is_file()]
    (output / "artifact_hashes.json").write_text(json.dumps({str(path): sha256(path) for path in paths}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
