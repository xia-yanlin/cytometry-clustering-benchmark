from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

TARGET = {"Nilsson_rare": "HSCs", "Mosmann_rare": "activated"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_separator(path: Path, registered: str) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    observed = "comma" if first.count(",") > first.count("\t") else "tab"
    if observed != registered:
        raise ValueError(
            f"Separator mismatch: registered={registered}, observed={observed}, file={path}"
        )
    return "," if observed == "comma" else "\t"


def load_dataset(root: Path, name: str, spec: dict):
    path = root / name / spec["filename"]
    frame = pd.read_csv(path, sep=validate_separator(path, spec["separator"]))
    markers = [column for column in frame.columns if column not in set(spec["exclude_columns"])]
    X = frame[markers].to_numpy(dtype=np.float64)
    X = np.arcsinh(X / float(spec["cofactor"]))
    y = frame["label"].astype(str).to_numpy()
    return path, X, y, markers


def score_target(y: np.ndarray, clusters: np.ndarray, target: str):
    actual = y == target
    candidates = []
    for cluster in np.unique(clusters):
        predicted = clusters == cluster
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
        candidates.append(
            {
                "selected_cluster": int(cluster),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "selected_cluster_cells": int(predicted.sum()),
            }
        )
    candidates.sort(key=lambda row: (-row["f1"], -row["recall"], row["selected_cluster"]))
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-004")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions").mkdir()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    rows = []
    inputs = []
    for dataset, target in TARGET.items():
        spec = config["datasets"][dataset]
        path, X, y, markers = load_dataset(root, dataset, spec)
        inputs.append(
            {
                "dataset": dataset,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "cells": len(y),
                "markers": len(markers),
                "target": target,
                "target_cells": int(np.sum(y == target)),
            }
        )
        for k in (2, 10, 40):
            for seed in range(args.seeds):
                model = KMeans(
                    n_clusters=k,
                    n_init=1,
                    max_iter=args.max_iter,
                    algorithm="lloyd",
                    random_state=seed,
                )
                started = time.perf_counter()
                clusters = model.fit_predict(X)
                runtime = time.perf_counter() - started
                target_metrics = score_target(y, clusters, target)
                prediction_path = args.output / "predictions" / f"{dataset}_K{k}_seed{seed:02d}.npz"
                np.savez_compressed(
                    prediction_path,
                    prediction_all=clusters.astype(np.int32),
                    cluster_centers=model.cluster_centers_,
                    inertia=np.array([model.inertia_]),
                    n_iter=np.array([model.n_iter_]),
                )
                row = {
                    "experiment_id": args.experiment_id,
                    "dataset": dataset,
                    "algorithm": "sklearn_KMeans",
                    "parameter_regime": f"fixed_K_{k}_sensitivity",
                    "k": k,
                    "seed": seed,
                    "n_init": 1,
                    "max_iter": args.max_iter,
                    "total_cells": len(y),
                    "target": target,
                    "target_cells": int(np.sum(y == target)),
                    "target_prevalence": float(np.mean(y == target)),
                    "ari_raw_partition": float(adjusted_rand_score(y, clusters)),
                    "runtime_seconds": runtime,
                    "inertia": float(model.inertia_),
                    "iterations": int(model.n_iter_),
                    "prediction_file": str(prediction_path),
                    "prediction_sha256": sha256(prediction_path),
                    **target_metrics,
                }
                rows.append(row)
                print(
                    f"{dataset} K={k:02d} seed={seed:02d} "
                    f"ARI={row['ari_raw_partition']:.6f} P={row['precision']:.6f} "
                    f"R={row['recall']:.6f} F1={row['f1']:.6f} runtime={runtime:.2f}s",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results.to_csv(args.output / "run_level_metrics.csv", index=False)
    pd.DataFrame(inputs).to_csv(args.output / "input_manifest.csv", index=False)
    summary = (
        results.groupby(["dataset", "k"])
        .agg(
            seeds=("seed", "count"),
            ari_mean=("ari_raw_partition", "mean"),
            ari_sd=("ari_raw_partition", "std"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            f2_mean=("f2", "mean"),
            f2_sd=("f2", "std"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_points": [12, 16, 17, 21, 31],
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "seeds": list(range(args.seeds)),
        "k_values": [2, 10, 40],
        "completed": True,
        "exit_code": 0,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.experiment_id} Rare-population K-sensitivity experiment",
        "",
        "This experiment addresses reviewer questions 12, 16, 17, 21, and 31. The report below presents numerical results without manuscript framing.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, K={row.k}: ARI {row.ari_mean:.6f}±{row.ari_sd:.6f}; "
            f"P {row.precision_mean:.6f}±{row.precision_sd:.6f}; "
            f"R {row.recall_mean:.6f}±{row.recall_sd:.6f}; "
            f"F1 {row.f1_mean:.6f}±{row.f1_sd:.6f}; F2 {row.f2_mean:.6f}±{row.f2_sd:.6f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is a K-sensitivity experiment for K-means and does not represent other algorithms.",
            "- The best target cluster is matched using reference labels for external evaluation only; it is not a deployable detector.",
            "- No single winner is selected across K values; the analysis asks only whether ARI and target-population metrics convey different information.",
        ]
    )
    (args.output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "stdout.log").write_text(
        f"Completed {len(results)} runs across 2 datasets, 3 K values, {args.seeds} seeds.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output), "runs": len(results), "exit_code": 0}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
