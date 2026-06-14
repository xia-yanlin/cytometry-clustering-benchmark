"""
data_loader.py
==============
Shared data-loading module imported by all algorithm scripts.

Usage
-----
    from data_loader import load_dataset, load_all, DATASETS, TRUE_K

    for name in DATASETS:
        X, y, marker_cols, pop_names = load_dataset(name)

Data root configuration
-----------------------
Set the environment variable CYTOMETRY_DATA_ROOT to the directory that
contains one sub-folder per dataset (e.g. Levine_13dim/, Samusik_01/, …).
If the variable is not set, the module falls back to a ``data/`` sub-folder
located next to this file.

    Linux / macOS:  export CYTOMETRY_DATA_ROOT=/path/to/datasets
    Windows CMD:    set CYTOMETRY_DATA_ROOT=D:\\path\\to\\datasets
    Windows PS:     $env:CYTOMETRY_DATA_ROOT = "D:\\path\\to\\datasets"

Alternatively, edit the DATA_ROOT line below directly.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "CYTOMETRY_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = [
    "Levine_13dim",
    "Levine_32dim",
    "Samusik_01",
    "Nilsson_rare",
    "Mosmann_rare",
]

# True population count for each dataset.
# For rare-population datasets (Nilsson_rare, Mosmann_rare) K=2 represents
# "target population + background", used only by K=true algorithms
# (k-means, GMM, Ward, FlowSOM).  The semantic differs from multi-population
# datasets where TRUE_K equals the number of true immune-cell populations.
TRUE_K = {
    "Levine_13dim": 24,
    "Levine_32dim": 14,
    "Samusik_01":   24,
    "Nilsson_rare":  2,   # HSC + background
    "Mosmann_rare":  2,   # activated CD4 T + background
}

# arcsinh cofactor per dataset (None = already transformed, skip).
# Levine_13dim: the HDCytoData version is pre-transformed; use raw values.
COFACTOR = {
    "Levine_13dim": None,
    "Levine_32dim": 5,
    "Samusik_01":   5,
    "Nilsson_rare": 150,
    "Mosmann_rare": 150,
}

# Columns to exclude (non-marker metadata + label).
EXCLUDE_COLS = {
    "Levine_13dim": ["label"],
    "Levine_32dim": [
        "Time", "Cell_length", "DNA1", "DNA2",
        "Viability", "file_number", "event_number", "label",
    ],
    "Samusik_01": [
        "Time", "Cell_length",
        "BC1", "BC2", "BC3", "BC4", "BC5", "BC6",
        "DNA1", "DNA2", "Cisplatin", "beadDist", "label",
    ],
    "Nilsson_rare": ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "PI", "Time", "label"],
    "Mosmann_rare": [
        "FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W",
        "Live_Dead", "Time", "label",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_sep(filepath: str) -> str:
    """Infer delimiter (comma or tab) from the first line of the file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    return "," if first.count(",") > first.count("\t") else "\t"


def _find_file(name: str) -> str:
    """
    Locate the primary data file inside the dataset folder.
    Files whose name contains 'notransform' are preferred (raw counts).
    """
    folder = os.path.join(DATA_ROOT, name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Set CYTOMETRY_DATA_ROOT or edit DATA_ROOT in data_loader.py."
        )
    files = [f for f in os.listdir(folder) if f.endswith((".txt", ".csv"))]
    if not files:
        raise FileNotFoundError(f"No .txt/.csv file found in {folder}")
    notrans = [f for f in files if "notransform" in f.lower()]
    chosen = notrans[0] if notrans else files[0]
    return os.path.join(folder, chosen)


def _filter_labeled(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Remove unlabelled cells; return only annotated rows.

    Strategy per dataset
    --------------------
    Levine_13dim  : keep rows with label ∈ {1…24}.  The label column may be
                    stored as float (with NaN for unlabelled cells), so we
                    coerce to numeric before filtering.
    Levine_32dim  : drop rows where label == 'unassigned'.
    Samusik_01    : drop rows where label == 'unassigned'.
    Nilsson_rare  : keep all rows ('other' is the background class).
    Mosmann_rare  : keep all rows ('other' is the background class).
    """
    if name == "Levine_13dim":
        numeric = pd.to_numeric(df["label"], errors="coerce")
        return df[(numeric >= 1) & (numeric <= 24)].copy()
    if name in ("Levine_32dim", "Samusik_01"):
        return df[df["label"] != "unassigned"].copy()
    return df.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    name: str, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load and pre-process one dataset.

    Parameters
    ----------
    name    : Dataset name; must be a key of DATASETS.
    verbose : Print a one-line summary when True.

    Returns
    -------
    X           : np.ndarray, shape (n_cells, n_markers).
                  arcsinh-transformed where COFACTOR is not None.
    y           : np.ndarray of str, shape (n_cells,).
                  Cell-population labels (all cast to string).
    marker_cols : list[str], names of the marker columns used.
    pop_names   : list[str], sorted unique population names.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}.  Choose from {DATASETS}.")

    fpath = _find_file(name)
    sep   = _detect_sep(fpath)
    df    = pd.read_csv(fpath, sep=sep)
    df    = _filter_labeled(df, name)

    exclude     = set(EXCLUDE_COLS[name])
    marker_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[marker_cols].values.astype(np.float64)
    if COFACTOR[name] is not None:
        X = np.arcsinh(X / COFACTOR[name])

    # Normalise labels to plain strings
    if name == "Levine_13dim":
        numeric = pd.to_numeric(df["label"], errors="coerce")
        y = np.array([str(int(v)) for v in numeric.values])
    else:
        y = df["label"].astype(str).values

    pop_names = sorted(set(y))

    if verbose:
        print(
            f"[{name}]  cells={X.shape[0]:,}  markers={X.shape[1]}"
            f"  populations={len(pop_names)}"
            f"  X range=[{X.min():.2f}, {X.max():.2f}]"
        )

    return X, y, marker_cols, pop_names


def load_all(verbose: bool = True) -> dict:
    """Load all five datasets and return them as a dict keyed by name."""
    return {name: load_dataset(name, verbose=verbose) for name in DATASETS}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("data_loader — self-test")
    print("=" * 60)
    data = load_all(verbose=True)

    print("\nPopulation-count check:")
    for name, (X, y, markers, pops) in data.items():
        expected = TRUE_K[name]
        actual   = len(pops)
        status   = "OK" if actual == expected else f"WARN expected {expected}"
        print(f"  {name:<20}  found {actual} populations  [{status}]")

    print("\nPopulation labels:")
    for name, (X, y, markers, pops) in data.items():
        print(f"  {name:<20}  {pops}")
