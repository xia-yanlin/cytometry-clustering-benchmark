from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Newer Windows installations may not provide WMIC.  Pinning the logical-core
# count prevents joblib from invoking WMIC merely to detect physical cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore", message="Could not find the number of physical cores.*", category=UserWarning
)

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

TRUE_K = {"Levine_13dim": 24, "Levine_32dim": 14, "Samusik_01": 24}


def validate_separator(path: Path, registered: str) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    observed = "comma" if first_line.count(",") > first_line.count("\t") else "tab"
    if observed != registered:
        raise ValueError(
            f"Separator mismatch for {path}: registered={registered}, observed={observed}"
        )
    return "\t" if observed == "tab" else ","


def load_dataset(root: Path, name: str, spec: dict):
    path = root / name / spec["filename"]
    sep = validate_separator(path, spec["separator"])
    frame = pd.read_csv(path, sep=sep)
    marker_cols = [column for column in frame.columns if column not in set(spec["exclude_columns"])]
    X = frame[marker_cols].to_numpy(dtype=np.float64)
    if spec["cofactor"] is not None:
        X = np.arcsinh(X / float(spec["cofactor"]))
    labels = frame["label"]
    if spec["label_policy"] == "integer_1_to_24_evaluable":
        numeric = pd.to_numeric(labels, errors="coerce")
        mask = (numeric.between(1, 24) & (numeric % 1 == 0)).to_numpy()
        y = np.array(
            [
                str(int(value)) if keep else "__unassigned__"
                for value, keep in zip(numeric, mask, strict=False)
            ]
        )
    else:
        text = labels.astype("string").str.strip()
        mask = (~(text.isna() | text.str.lower().eq("unassigned"))).to_numpy()
        y = text.mask(~mask, "__unassigned__").astype(str).to_numpy()
    return path, X, y, mask, marker_cols


def align_and_score(y_true: np.ndarray, y_pred: np.ndarray):
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for i, label in enumerate(true_labels):
        for j, cluster in enumerate(pred_labels):
            matrix[i, j] = int(np.sum((y_true == label) & (y_pred == cluster)))
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {pred_labels[col]: true_labels[row] for row, col in zip(rows, cols, strict=False)}
    aligned = np.array([mapping.get(value, "__extra_cluster__") for value in y_pred])
    per_population = []
    for label in true_labels:
        actual = y_true == label
        predicted = aligned == label
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_population.append(
            {
                "population": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    macro_f1 = float(np.mean([row["f1"] for row in per_population]))
    macro_precision = float(np.mean([row["precision"] for row in per_population]))
    macro_recall = float(np.mean([row["recall"] for row in per_population]))
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "n_clusters": int(len(pred_labels)),
        "unmatched_cells": int(np.sum(aligned == "__extra_cluster__")),
    }, per_population


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-003")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--datasets", nargs="*", default=list(TRUE_K))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions").mkdir()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    run_rows = []
    population_rows = []
    input_rows = []

    for dataset in args.datasets:
        if dataset not in TRUE_K:
            raise ValueError(f"Unsupported dataset for this experiment: {dataset}")
        spec = config["datasets"][dataset]
        path, X, y, evaluable, markers = load_dataset(root, dataset, spec)
        eval_indices = np.flatnonzero(evaluable)
        np.save(args.output / "predictions" / f"{dataset}_evaluation_indices.npy", eval_indices)
        input_rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "bytes": path.stat().st_size,
                "total_cells": len(y),
                "evaluated_cells": int(evaluable.sum()),
                "markers": len(markers),
                "true_k": TRUE_K[dataset],
            }
        )
        y_eval = y[evaluable]
        for seed in range(args.seeds):
            for policy in ("all_cells_fit", "labeled_only_fit"):
                fit_mask = np.ones(len(y), dtype=bool) if policy == "all_cells_fit" else evaluable
                model = KMeans(
                    n_clusters=TRUE_K[dataset],
                    n_init=1,
                    max_iter=args.max_iter,
                    algorithm="lloyd",
                    random_state=seed,
                )
                started = time.perf_counter()
                model.fit(X[fit_mask])
                prediction_all = model.predict(X)
                runtime = time.perf_counter() - started
                prediction_eval = prediction_all[evaluable]
                metrics, per_population = align_and_score(y_eval, prediction_eval)
                prediction_path = (
                    args.output / "predictions" / f"{dataset}_{policy}_seed{seed:02d}.npz"
                )
                np.savez_compressed(
                    prediction_path,
                    prediction_all=prediction_all.astype(np.int32),
                    cluster_centers=model.cluster_centers_,
                    inertia=np.array([model.inertia_]),
                    n_iter=np.array([model.n_iter_]),
                )
                base = {
                    "experiment_id": args.experiment_id,
                    "dataset": dataset,
                    "algorithm": "sklearn_KMeans",
                    "parameter_regime": "K=true_policy_sensitivity",
                    "fit_policy": policy,
                    "seed": seed,
                    "n_init": 1,
                    "max_iter": args.max_iter,
                    "requested_k": TRUE_K[dataset],
                    "fit_cells": int(fit_mask.sum()),
                    "total_cells": len(y),
                    "evaluated_cells": int(evaluable.sum()),
                    "runtime_seconds": runtime,
                    "inertia": float(model.inertia_),
                    "iterations": int(model.n_iter_),
                    "prediction_file": str(prediction_path),
                }
                base.update(metrics)
                run_rows.append(base)
                for row in per_population:
                    population_rows.append(
                        {
                            "experiment_id": args.experiment_id,
                            "dataset": dataset,
                            "fit_policy": policy,
                            "seed": seed,
                            **row,
                        }
                    )
                print(
                    f"{dataset} seed={seed:02d} {policy} "
                    f"ARI={metrics['ari']:.6f} MacroF1={metrics['macro_f1']:.6f} "
                    f"runtime={runtime:.2f}s",
                    flush=True,
                )

    runs = pd.DataFrame(run_rows)
    populations = pd.DataFrame(population_rows)
    inputs = pd.DataFrame(input_rows)
    runs.to_csv(args.output / "run_level_metrics.csv", index=False)
    populations.to_csv(args.output / "population_level_metrics.csv", index=False)
    inputs.to_csv(args.output / "input_summary.csv", index=False)

    paired = runs.pivot_table(
        index=["dataset", "seed"],
        columns="fit_policy",
        values=["ari", "macro_precision", "macro_recall", "macro_f1"],
    )
    paired.columns = [f"{metric}_{policy}" for metric, policy in paired.columns]
    paired = paired.reset_index()
    for metric in ("ari", "macro_precision", "macro_recall", "macro_f1"):
        paired[f"delta_{metric}_all_minus_labeled"] = (
            paired[f"{metric}_all_cells_fit"] - paired[f"{metric}_labeled_only_fit"]
        )
    paired.to_csv(args.output / "paired_policy_differences.csv", index=False)

    summary = (
        paired.groupby("dataset")
        .agg(
            seeds=("seed", "count"),
            mean_delta_ari=("delta_ari_all_minus_labeled", "mean"),
            sd_delta_ari=("delta_ari_all_minus_labeled", "std"),
            mean_delta_macro_f1=("delta_macro_f1_all_minus_labeled", "mean"),
            sd_delta_macro_f1=("delta_macro_f1_all_minus_labeled", "std"),
            all_better_ari_fraction=(
                "delta_ari_all_minus_labeled",
                lambda values: float(np.mean(values > 0)),
            ),
            all_better_f1_fraction=(
                "delta_macro_f1_all_minus_labeled",
                lambda values: float(np.mean(values > 0)),
            ),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "policy_summary.csv", index=False)

    manifest = {
        "experiment_id": args.experiment_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Effect of pre-clustering removal of unassigned cells",
        "parameter_regime": "K=true sensitivity; not a deployable primary comparison",
        "preprocessing": "arcsinh only according to datasets.json; no feature standardization",
        "seeds": list(range(args.seeds)),
        "script": str(Path(__file__).resolve()),
        "config": str(args.config.resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "completed": True,
        "exit_code": 0,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "stdout.log").write_text(
        f"Completed {len(runs)} runs across {len(args.datasets)} datasets.\n", encoding="utf-8"
    )
    lines = [
        f"# {args.experiment_id} Paired experiment on the inclusion of unlabeled cells",
        "",
        "The analysis compares all-event fitting with labeled-event fitting while keeping the algorithm, cluster count and random seed fixed.",
        "",
        "This experiment uses K=true and is therefore an input-policy sensitivity analysis, not an unlabeled primary comparison.",
        "",
        "## Results",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.dataset}: {row.seeds} paired seeds; all-cell fitting minus labeled-only fitting produced a mean ARI difference of {row.mean_delta_ari:+.6f} and a mean macro-F1 difference of {row.mean_delta_macro_f1:+.6f}."
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All runs use the same algorithm, K, parameters, seed, and evaluation cells; only the fitting set changes.",
            "- Each run saves all-event cluster labels, cluster centers, inertia, and iteration count.",
            "- This experiment quantifies the effect of label-based filtering; it does not determine the best algorithm or the manuscript narrative.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "runs": len(runs), "exit_code": 0}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
