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
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

ENDPOINTS = [
    "ari",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_f2",
    "quality_q",
    "n_valid_communities",
    "outlier_events",
    "target_outlier_fraction",
    "target_overlapping_valid_communities",
    "target_effective_valid_communities",
    "target_dominant_valid_community_capture",
    "target_outlier_events",
    "target_events",
]
CORR_ENDPOINTS = ["ari", "target_precision", "target_recall", "target_f1", "target_f2"]
BOOT = 10_000
SEED = 20_260_911


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inventory_failures(inv_path: Path, base: Path):
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    bad = []
    for name, digest in inv.items():
        p = Path(name)
        p = p if p.is_absolute() else base / p
        if not p.is_file() or sha256(p) != digest:
            bad.append(name)
    return bad


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        out[idx] = min(running, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nilsson", type=Path, required=True)
    ap.add_argument("--mosmann", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    parents = {"Nilsson_rare": a.nilsson.resolve(), "Mosmann_rare": a.mosmann.resolve()}
    protocol = a.protocol.resolve()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    frames = []
    pairs = []
    links = []

    def check(scope, name, passed, detail):
        checks.append({"scope": scope, "check": name, "passed": bool(passed), "detail": detail})

    for dataset, parent in parents.items():
        progress = json.loads((parent / "progress_manifest.json").read_text(encoding="utf-8"))
        index = pd.read_csv(parent / "run_index.csv").sort_values("repeat").reset_index(drop=True)
        passed = index.all_checks_passed.astype(str).str.lower().eq("true")
        check(
            dataset,
            "parent_contract",
            progress.get("all_runs_complete_and_passed") is True
            and progress.get("seed_control") == "unavailable"
            and len(index) == 30
            and index.repeat.tolist() == list(range(30))
            and passed.all(),
            f"rows={len(index)}; passed={passed.sum()}",
        )
        bad = inventory_failures(parent / "artifact_hashes.json", parent)
        check(dataset, "top_hashes", not bad, f"bad={len(bad)}")
        labels = {}
        run_hashes = {}
        run_bad = []
        for repeat in range(30):
            d = parent / "runs" / f"repeat{repeat:03d}"
            run_bad.extend(inventory_failures(d / "artifact_hashes.json", d))
            labels[repeat] = np.load(d / "community_labels_all_events.npy", allow_pickle=False)
            run_hashes[str(repeat)] = sha256(d / "run_manifest.json")
        check(dataset, "run_hashes", not run_bad, f"bad={len(run_bad)}")
        for i in range(30):
            for j in range(i + 1, 30):
                pairs.append(
                    {
                        "dataset": dataset,
                        "repeat_a": i,
                        "repeat_b": j,
                        "partition_ari_all_events": float(
                            adjusted_rand_score(labels[i], labels[j])
                        ),
                    }
                )
        index.insert(0, "source_run", parent.name)
        frames.append(index)
        links.append(
            {
                "dataset": dataset,
                "path": str(parent),
                "progress_manifest_sha256": sha256(parent / "progress_manifest.json"),
                "run_index_sha256": sha256(parent / "run_index.csv"),
                "graph_sha256": sha256(parent / "fixed_full_event_graph.npz"),
                "neighbors_sha256": sha256(parent / "neighbor_indices_k30.npy"),
                "run_manifest_sha256_by_repeat": run_hashes,
            }
        )
    points = pd.concat(frames, ignore_index=True)
    points.to_csv(output / "all_run_points.csv", index=False)
    pair = pd.DataFrame(pairs)
    pair.to_csv(output / "pairwise_partition_ari.csv", index=False)
    check(
        "experiment",
        "counts",
        len(points) == 60 and len(pair) == 870 and pair.groupby("dataset").size().eq(435).all(),
        f"points={len(points)}; pairs={len(pair)}",
    )
    rng = np.random.default_rng(SEED)
    desc = []
    corr = []
    for dataset, group in points.groupby("dataset", sort=True):
        group = group.sort_values("repeat")
        for endpoint in ENDPOINTS:
            x = group[endpoint].to_numpy(float)
            means = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
            q1, q3 = np.quantile(x, [0.25, 0.75])
            desc.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "n_runs": len(x),
                    "mean": float(x.mean()),
                    "sd": float(x.std(ddof=1)),
                    "median": float(np.median(x)),
                    "mad_unscaled": float(np.median(np.abs(x - np.median(x)))),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "range": float(np.ptp(x)),
                    "bootstrap_mean_ci95_low": float(np.quantile(means, 0.025)),
                    "bootstrap_mean_ci95_high": float(np.quantile(means, 0.975)),
                    "bootstrap_replicates": BOOT,
                }
            )
        for endpoint in CORR_ENDPOINTS:
            x = group.quality_q.to_numpy(float)
            y = group[endpoint].to_numpy(float)
            defined = bool(np.ptp(x) > 0 and np.ptp(y) > 0)
            if defined:
                r = spearmanr(x, y)
                rho, p_raw = float(r.statistic), float(r.pvalue)
                reason = ""
            else:
                rho, p_raw, reason = float("nan"), float("nan"), "constant_input"
            corr.append(
                {
                    "dataset": dataset,
                    "x": "quality_q",
                    "y": endpoint,
                    "n_runs": len(group),
                    "correlation_defined": defined,
                    "undefined_reason": reason,
                    "spearman_rho": rho,
                    "p_raw": p_raw,
                }
            )
    descriptive = pd.DataFrame(desc)
    descriptive.to_csv(output / "endpoint_descriptive_statistics.csv", index=False)
    correlations = pd.DataFrame(corr)
    correlations["p_for_holm"] = correlations.p_raw.where(correlations.p_raw.notna(), 1.0)
    correlations["p_holm_across_ten"] = holm(correlations.p_for_holm)
    correlations["reject_0_05_holm"] = correlations.correlation_defined & (
        correlations.p_holm_across_ten <= 0.05
    )
    correlations.to_csv(output / "quality_external_correlations.csv", index=False)
    pair_summary = (
        pair.groupby("dataset")
        .partition_ari_all_events.agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    pair_summary["pairs"] = 435
    pair_summary.to_csv(output / "pairwise_partition_ari_summary.csv", index=False)
    check(
        "experiment",
        "analysis_families",
        len(descriptive) == 28
        and len(correlations) == 10
        and descriptive.groupby("dataset").size().eq(14).all(),
        f"descriptive={len(descriptive)}; correlations={len(correlations)}",
    )
    check(
        "experiment",
        "undefined_correlations",
        int((~correlations.correlation_defined).sum()) == 1
        and correlations.loc[~correlations.correlation_defined, "y"].eq("target_recall").all()
        and correlations.loc[~correlations.correlation_defined, "dataset"].eq("Nilsson_rare").all(),
        str(
            correlations.loc[~correlations.correlation_defined, ["dataset", "y"]].to_dict("records")
        ),
    )
    check(
        "experiment",
        "ranges",
        points.ari.between(-1, 1).all()
        and all(
            points[x].between(0, 1).all()
            for x in [
                "target_precision",
                "target_recall",
                "target_f1",
                "target_f2",
                "target_outlier_fraction",
                "target_dominant_valid_community_capture",
            ]
        ),
        "legal",
    )
    cf = pd.DataFrame(checks)
    cf.to_csv(output / "analysis_checks.csv", index=False)
    initial = Path(__file__).with_name("analyze_phenograph_rare_full_event_louvain30_initial.py")
    r1 = Path(__file__).with_name("analyze_phenograph_rare_full_event_louvain30_r1.py")
    initial_hash = sha256(initial) if initial.is_file() else None
    r1_hash = sha256(r1) if r1.is_file() else None
    manifest = {
        "experiment_id": "EXP-035C-R2",
        "supersedes": "EXP-035C-R1 after the Mosmann parent inventory was refreshed to include the execution-boundary note",
        "scientific_values_expected_identical_to": "EXP-035C-R1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "initial_script_snapshot": str(initial.resolve()) if initial.is_file() else None,
        "initial_script_snapshot_sha256": initial_hash,
        "r1_script_snapshot": str(r1.resolve()) if r1.is_file() else None,
        "r1_script_snapshot_sha256": r1_hash,
        "source_links": links,
        "endpoints": ENDPOINTS,
        "correlation_endpoints": CORR_ENDPOINTS,
        "repeat_not_seed": True,
        "evaluation_scope": "all_events",
        "undefined_correlation_policy": "record NaN rho/p; use p=1 placeholder in the preregistered Holm family of 10",
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOT,
        "checks_passed": int(cf.passed.sum()),
        "checks_total": len(cf),
        "all_checks_passed": bool(cf.passed.all()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "scientific_summary.md").write_text(
        f"# EXP-035C-R2 PhenoGraph full-event analysis of rare-population datasets\n\nThe analysis was rebound to the Mosmann parent manifest containing the execution-boundary note. It includes 60 parent runs, 870 all-event partition pairs, 28 endpoint summaries, and 10 preregistered correlations. Checks passed: {manifest['checks_passed']}/{manifest['checks_total']}. Scientific values must match C-R1. Nilsson target recall is constant, so its correlation with Q is explicitly undefined and occupies its place in the Holm family with p=1. Repeats are not controlled seeds, and no best repeat was selected.\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
