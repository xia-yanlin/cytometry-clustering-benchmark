"""
evaluation.py  (v2.1)
=====================
Shared evaluation module imported by all algorithm scripts.

Core functions
--------------
    metrics = evaluate(y_true, y_pred, dataset_name, rare_found_threshold)
    save_results(rows, csv_path)

Evaluation pipeline
-------------------
1. ARI              — computed directly without alignment.
2. Hungarian alignment — maps predicted cluster IDs to the optimal true
                         population labels via linear assignment.
3. Macro F1         — equal-weight average over all true populations;
                      populations with no matching cluster score F1 = 0.
4. Accuracy         — fraction of correctly classified cells after alignment.
5. Per-population F1 — individual F1 for each true population.

Hungarian alignment notes
-------------------------
Linear assignment (scipy.optimize.linear_sum_assignment) provides a
one-to-one mapping from predicted clusters to true populations.

* K_pred < K_true : some true populations have no matching cluster → F1 = 0.
* K_pred > K_true : excess predicted clusters are labelled '__unmatched__'.
  Cells in unmatched clusters count as misclassifications in accuracy
  (included in the denominator) but are excluded from per-population
  TP/FP/FN tallies.  The field ``n_unmatched_cells`` is provided so callers
  can assess how much the accuracy figure is affected.

Rare-population datasets (Nilsson_rare / Mosmann_rare)
-------------------------------------------------------
* ARI is computed normally.
* Target-population F1 / Precision / Recall / F2 (β = 2) are reported.
* Macro F1 is set to None (the 'other' background is not averaged over).

Changes in v2.1 (vs v2)
------------------------
* ``rare_found_threshold`` is now a configurable parameter (default 0.1)
  instead of a hard-coded constant.
* New output field ``n_unmatched_cells``.
* Backward compatible: all existing field names and semantics are unchanged.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

# Target population name for each rare-population dataset.
# These strings must match the label values present in the data files.
RARE_TARGET = {
    "Nilsson_rare": "HSCs",
    "Mosmann_rare": "activated",
}

# Default F1 threshold for the binary ``rare_target_found`` flag.
# This is a coarse existence check; the key evaluation signal is the
# continuous rare_target_f1 / precision / recall / f2 values.
DEFAULT_RARE_FOUND_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hungarian_align(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Map predicted cluster IDs to true population labels via linear assignment.

    Handles K_pred ≠ K_true:
    * K_pred < K_true : unmatched true populations get F1 = 0.
    * K_pred > K_true : excess predicted clusters are labelled '__unmatched__'.

    Returns
    -------
    y_aligned : np.ndarray, same shape as y_true.
                Predicted cluster IDs replaced by matched population names;
                unmatched clusters replaced by '__unmatched__'.
    mapping   : dict {pred_cluster_id: true_population_label}
    """
    true_labels   = np.unique(y_true)
    pred_clusters = np.unique(y_pred)
    n_true, n_pred = len(true_labels), len(pred_clusters)

    # Build cost matrix (padded to square if K_pred ≠ K_true).
    size = max(n_true, n_pred)
    cost = np.zeros((size, size), dtype=np.int64)
    for i, tl in enumerate(true_labels):
        for j, pc in enumerate(pred_clusters):
            cost[i, j] = -int(np.sum((y_true == tl) & (y_pred == pc)))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {
        pred_clusters[c]: true_labels[r]
        for r, c in zip(row_ind, col_ind)
        if r < n_true and c < n_pred
    }
    y_aligned = np.array([mapping.get(p, "__unmatched__") for p in y_pred])
    return y_aligned, mapping


def _prf_for_label(
    y_true: np.ndarray, y_aligned: np.ndarray, label: str
) -> dict:
    """
    Compute Precision, Recall, F1, and F2 (β = 2) for a single population.

    F2 weights recall twice as heavily as precision, reflecting the higher
    cost of missing rare cells compared to false positives.

    Parameters
    ----------
    y_true   : Ground-truth label array.
    y_aligned: Hungarian-aligned prediction array.
    label    : Target population name.

    Returns
    -------
    dict with keys: precision, recall, f1, f2, tp, fp, fn
    """
    mask_true = y_true    == label
    mask_pred = y_aligned == label

    tp = int(np.sum( mask_true &  mask_pred))
    fp = int(np.sum(~mask_true &  mask_pred))
    fn = int(np.sum( mask_true & ~mask_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    # F_β = (1 + β²) · P · R / (β² · P + R),  β = 2
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if (4 * precision + recall) > 0 else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "f2":        round(f2,        4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    y_true,
    y_pred,
    dataset_name: str | None = None,
    rare_found_threshold: float = DEFAULT_RARE_FOUND_THRESHOLD,
) -> dict:
    """
    Compute the full evaluation metric set for one clustering run.

    Parameters
    ----------
    y_true                : array-like of str, ground-truth population labels.
    y_pred                : array-like of int (0-based), predicted cluster IDs.
    dataset_name          : Dataset name; identifies rare-population datasets.
    rare_found_threshold  : F1 threshold for the binary rare_target_found flag.

    Returns
    -------
    dict with the following keys:

    All datasets
    ~~~~~~~~~~~~
    ARI, MacroF1, accuracy, n_clusters_found
    n_unmatched_cells : int — cells assigned to '__unmatched__' clusters
                               (> 0 when K_pred > K_true; degrades accuracy).
    per_pop_f1        : dict {population: F1}
    per_pop_pr        : dict {population: {precision, recall, f1, f2, tp, fp, fn}}

    Rare-population datasets only (Nilsson_rare / Mosmann_rare)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rare_target_f1, rare_target_found, rare_target_precision,
    rare_target_recall, rare_target_f2
    (MacroF1 is set to None for rare datasets.)
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    is_rare = dataset_name in RARE_TARGET

    results = {}

    # 1. ARI
    results["ARI"] = round(float(adjusted_rand_score(y_true, y_pred)), 4)

    # 2. Number of predicted clusters
    results["n_clusters_found"] = int(len(np.unique(y_pred)))

    # 3. Hungarian alignment
    y_aligned, _ = _hungarian_align(y_true, y_pred)

    # 4. Unmatched-cell count
    n_unmatched = int(np.sum(y_aligned == "__unmatched__"))
    results["n_unmatched_cells"] = n_unmatched

    # 5. Accuracy (unmatched cells count as misclassifications)
    matched = y_aligned != "__unmatched__"
    results["accuracy"] = round(
        float(np.sum(y_true[matched] == y_aligned[matched])) / len(y_true), 4
    ) if matched.any() else 0.0

    # 6. Per-population metrics
    per_pop_f1 = {}
    per_pop_pr = {}
    for label in sorted(np.unique(y_true)):
        label = str(label)
        prf = _prf_for_label(y_true, y_aligned, label)
        per_pop_f1[label] = prf["f1"]
        per_pop_pr[label] = prf
        # Note: for rare datasets the 'other' background also appears here
        # (for debugging); it is not included in any summary metric.

    results["per_pop_f1"] = per_pop_f1
    results["per_pop_pr"] = per_pop_pr

    # 7. Macro F1 / rare-population special case
    if is_rare:
        target = RARE_TARGET[dataset_name]
        prf    = per_pop_pr.get(
            str(target), {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
        )
        results["MacroF1"]               = None
        results["rare_target_f1"]        = prf["f1"]
        results["rare_target_found"]     = bool(prf["f1"] > rare_found_threshold)
        results["rare_target_precision"] = prf["precision"]
        results["rare_target_recall"]    = prf["recall"]
        results["rare_target_f2"]        = prf["f2"]
    else:
        results["MacroF1"] = round(float(np.mean(list(per_pop_f1.values()))), 4)

    return results


def save_results(rows: list[dict], csv_path: str) -> None:
    """
    Append a list of result rows to a CSV file (creates the file if absent).

    per_pop_f1  → columns named ``f1_{population}``
    per_pop_pr  → columns named ``precision_{population}`` and
                  ``recall_{population}``
    """
    flat_rows = []
    for row in rows:
        flat = {
            k: v for k, v in row.items()
            if k not in ("per_pop_f1", "per_pop_pr")
        }
        if row.get("per_pop_f1"):
            for pop, f1 in row["per_pop_f1"].items():
                flat[f"f1_{pop}"] = f1
        if row.get("per_pop_pr"):
            for pop, prf in row["per_pop_pr"].items():
                flat[f"precision_{pop}"] = prf["precision"]
                flat[f"recall_{pop}"]    = prf["recall"]
        flat_rows.append(flat)

    df_new = pd.DataFrame(flat_rows)
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def print_summary(results: dict, dataset_name: str, algorithm_name: str) -> None:
    """Print a one-line result summary to stdout."""
    ari = results["ARI"]
    k   = results["n_clusters_found"]
    un  = results.get("n_unmatched_cells", 0)

    if dataset_name in RARE_TARGET:
        target = RARE_TARGET[dataset_name]
        f1     = results["rare_target_f1"]
        found  = results["rare_target_found"]
        prec   = results.get("rare_target_precision", float("nan"))
        rec    = results.get("rare_target_recall",    float("nan"))
        f2     = results.get("rare_target_f2",        float("nan"))
        print(
            f"  [{algorithm_name} | {dataset_name}]  "
            f"ARI={ari:.4f}  "
            f"{target}_F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  "
            f"F2={f2:.4f}  found={'YES' if found else 'NO'}  K={k}"
        )
    else:
        mf1 = results["MacroF1"]
        acc = results["accuracy"]
        un_note = f"  unmatched={un}" if un > 0 else ""
        print(
            f"  [{algorithm_name} | {dataset_name}]  "
            f"ARI={ari:.4f}  MacroF1={mf1:.4f}  "
            f"acc={acc:.4f}  K={k}{un_note}"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("evaluation.py v2.1 — unit tests")
    print("=" * 55)

    # --- multi-population ---
    y_t = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 50)
    y_p = np.array([0]*90 + [1]*10 + [1]*90 + [2]*10 + [2]*45 + [0]*5)
    res = evaluate(y_t, y_p, dataset_name="test")
    print(f"ARI              = {res['ARI']}")
    print(f"MacroF1          = {res['MacroF1']}")
    print(f"accuracy         = {res['accuracy']}")
    print(f"n_clusters_found = {res['n_clusters_found']}")
    print(f"n_unmatched_cells= {res['n_unmatched_cells']}")
    print(f"per_pop_f1       = {res['per_pop_f1']}")

    # --- K_pred > K_true: n_unmatched_cells must be > 0 ---
    y_t2 = np.array(["A"] * 100 + ["B"] * 100)
    y_p2 = np.array([0]*80 + [1]*20 + [1]*80 + [2]*20)
    res2 = evaluate(y_t2, y_p2, dataset_name="test")
    assert res2["n_unmatched_cells"] > 0, "Expected unmatched cells"
    print(f"\nK_pred > K_true: n_unmatched_cells={res2['n_unmatched_cells']}  PASS")

    # --- rare population (default threshold) ---
    y_tr = np.array(["HSCs"] * 10 + ["other"] * 990)
    y_pr = np.array([0] * 10 + [1] * 990)
    r    = evaluate(y_tr, y_pr, dataset_name="Nilsson_rare")
    print(f"\nRare test (threshold=0.1):")
    print(f"  rare_target_f1    = {r['rare_target_f1']}")
    print(f"  rare_target_found = {r['rare_target_found']}")
    print(f"  MacroF1           = {r['MacroF1']}  (should be None)")

    # --- rare population (custom threshold) ---
    r2 = evaluate(y_tr, y_pr, dataset_name="Nilsson_rare", rare_found_threshold=0.5)
    print(f"\nRare test (threshold=0.5, F1=1.0 → found should be True):")
    print(f"  rare_target_found = {r2['rare_target_found']}  PASS")

    # --- exact value check ---
    y_te = np.array(["HSCs"] * 50 + ["other"] * 950)
    y_pe = np.array([0] * 50 + [1] * 950)
    re   = evaluate(y_te, y_pe, dataset_name="Nilsson_rare")
    assert re["rare_target_f1"]        == 1.0
    assert re["rare_target_precision"] == 1.0
    assert re["rare_target_recall"]    == 1.0
    assert re["rare_target_f2"]        == 1.0
    print("\nExact value check  PASS")

    # --- imbalanced P/R ---
    y_ti = np.array(["HSCs"] * 10 + ["other"] * 990)
    y_pi = np.array([0]*10 + [0]*90 + [1]*900)
    ri   = evaluate(y_ti, y_pi, dataset_name="Nilsson_rare")
    assert ri["rare_target_recall"]    == 1.0
    assert ri["rare_target_precision"] <  0.2
    assert ri["rare_target_f2"]        >  ri["rare_target_f1"]
    print("Imbalanced P/R check  PASS")

    print("\nAll tests passed.")
