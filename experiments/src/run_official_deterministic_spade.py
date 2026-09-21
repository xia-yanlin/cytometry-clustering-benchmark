from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_official_xshift import read_fcs
from scipy import sparse
from scipy.io import loadmat, savemat
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN_ROOT = ROOT / "runs" / "EXP-044_deterministic_spade_official_endpoint_20260915"
PROTOCOL = ROOT / "protocols" / "EXP-044_deterministic_spade_official_endpoint_preregistered.md"
PROTOCOL_R1 = (
    ROOT / "protocols" / "EXP-044-R1_deterministic_spade_fcs_compatibility_preregistered.md"
)
PROTOCOL_R2 = (
    ROOT / "protocols" / "EXP-044-R2_deterministic_spade_mkl_compatibility_preregistered.md"
)
PROTOCOL_R3 = (
    ROOT / "protocols" / "EXP-044-R3_deterministic_spade_legacy_mkl_cpu_detection_preregistered.md"
)
INPUT_FCS = (
    ROOT
    / "runs"
    / "EXP-010B_xshift_nilsson_fixedK20_20260910"
    / "Nilsson_rare_all_events_markers.fcs"
)
SPADE_REPO = WORKSPACE / "reference_implementations" / "deterministic-spade"
SPADE_EXE = (
    SPADE_REPO
    / "standalone_win"
    / "SPADE3_2016_10_10_win"
    / "for_testing"
    / "SPADE3_2016_10_10_win.exe"
)
INSTALLER = (
    ROOT
    / "environments"
    / "matlab_runtime_R2015a_v85"
    / "downloads"
    / "MCR_R2015a_win64_installer.exe"
)
RUNTIME_ROOT = (
    Path(os.environ["MATLAB_RUNTIME_ROOT"]) if "MATLAB_RUNTIME_ROOT" in os.environ else None
)

EXPECTED_INPUT_SHA256 = "f83c6da079149e82a42ecaab6cc3ee2be63a9603cee3ad80c28f278867e43b6b"
EXPECTED_EXE_SHA256 = "abbe3dacd1adf5a9eb6a2d6467b073c4a678b93c51c7c1b1c605f123fcdeda8a"
EXPECTED_INSTALLER_SHA256 = "6e265e0619851a86baa5f5edcae836cc68046977ecb00eeb6085e5a4f98ff812"
EXPECTED_EVENTS = 44_140
EXPECTED_MARKERS = 13
INPUT_NAME = "Nilsson_rare_all_events_markers.fcs"
RESULT_NAME = "SPADE_cluster_mst_upsample_result.mat"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cellstr(values: list[str]) -> np.ndarray:
    result = np.empty((len(values), 1), dtype=object)
    for index, value in enumerate(values):
        result[index, 0] = value
    return result


def attempt_path(attempt: int) -> Path:
    return RUN_ROOT / f"attempt{attempt:03d}"


def fcs_offsets(raw: bytes) -> dict[str, int]:
    if raw[:6] not in {b"FCS2.0", b"FCS3.0", b"FCS3.1"}:
        raise ValueError(f"unsupported FCS signature: {raw[:6]!r}")

    def number(start: int, stop: int) -> int:
        field = raw[start:stop].decode("ascii").strip()
        return int(field) if field else 0

    return {
        "text_start": number(10, 18),
        "text_end": number(18, 26),
        "data_start": number(26, 34),
        "data_end": number(34, 42),
    }


def make_spade_delimiter_derivative(source: Path, destination: Path) -> dict[str, object]:
    parent = source.read_bytes()
    offsets = fcs_offsets(parent)
    text_start = offsets["text_start"]
    text_end = offsets["text_end"]
    data_start = offsets["data_start"]
    data_end = offsets["data_end"]
    text = parent[text_start : text_end + 1]
    if not text or text[0] != ord("#") or text[-1] != ord("#"):
        raise ValueError("parent TEXT segment does not have the registered '#' boundary delimiter")
    if b"##" in text:
        raise ValueError(
            "parent TEXT contains escaped/doubled delimiters; generic replacement is not registered"
        )
    if text.count(b"#") % 2 != 1:
        raise ValueError("parent TEXT delimiter count is inconsistent with key/value pairs")

    derivative = bytearray(parent)
    changed_positions: list[int] = []
    for position in range(text_start, text_end + 1):
        if derivative[position] == ord("#"):
            derivative[position] = ord("|")
            changed_positions.append(position)
    destination.write_bytes(derivative)
    rendered = bytes(derivative)

    if len(parent) != len(rendered):
        raise ValueError("compatibility conversion changed file length")
    if fcs_offsets(rendered) != offsets:
        raise ValueError("compatibility conversion changed FCS HEADER offsets")
    if parent[data_start : data_end + 1] != rendered[data_start : data_end + 1]:
        raise ValueError("compatibility conversion changed the FCS DATA segment")
    if any(position < text_start or position > text_end for position in changed_positions):
        raise ValueError("compatibility conversion changed a byte outside the TEXT segment")

    parent_matrix, parent_markers = read_fcs(source)
    derived_matrix, derived_markers = read_fcs(destination)
    if parent_markers != derived_markers:
        raise ValueError("compatibility conversion changed marker names or order")
    if not np.array_equal(parent_matrix, derived_matrix):
        raise ValueError("compatibility conversion changed expression values or event order")
    return {
        "operation": "replace FCS TEXT delimiter '#' with '|'",
        "text_start": text_start,
        "text_end": text_end,
        "data_start": data_start,
        "data_end": data_end,
        "changed_byte_count": len(changed_positions),
        "changed_positions": changed_positions,
        "file_length_unchanged": True,
        "header_offsets_unchanged": True,
        "data_segment_bytes_exact": True,
        "matrix_values_exact": True,
        "marker_names_and_order_exact": True,
    }


def prepare(attempt: int, compat_delimiter: bool, protocol_revision: str) -> None:
    target = attempt_path(attempt)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")

    for path, expected in (
        (INPUT_FCS, EXPECTED_INPUT_SHA256),
        (SPADE_EXE, EXPECTED_EXE_SHA256),
        (INSTALLER, EXPECTED_INSTALLER_SHA256),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")

    matrix, marker_names = read_fcs(INPUT_FCS)
    if matrix.shape != (EXPECTED_EVENTS, EXPECTED_MARKERS):
        raise ValueError(f"unexpected input shape: {matrix.shape}")
    if len(set(marker_names)) != EXPECTED_MARKERS:
        raise ValueError("marker names are not unique")

    target.mkdir(parents=True, exist_ok=False)
    copied_fcs = target / INPUT_NAME
    compatibility = None
    if compat_delimiter:
        compatibility = make_spade_delimiter_derivative(INPUT_FCS, copied_fcs)
    else:
        shutil.copy2(INPUT_FCS, copied_fcs)
        if sha256(copied_fcs) != EXPECTED_INPUT_SHA256:
            raise ValueError("copied FCS failed byte-identity check")
    copied_matrix, copied_markers = read_fcs(copied_fcs)
    if (
        copied_matrix.shape != matrix.shape
        or copied_markers != marker_names
        or not np.array_equal(copied_matrix, matrix)
    ):
        raise ValueError("prepared FCS is not numerically identical to the registered parent")

    parameters = {
        "all_fcs_filenames": cellstr([INPUT_NAME]),
        "file_annot": cellstr(["Nilsson_rare"]),
        "all_markers": cellstr(marker_names),
        "all_overlapping_markers": cellstr(marker_names),
        "used_markers": cellstr(marker_names),
        "apply_compensation": np.array([[0.0]]),
        "transformation_option": np.array([[1.0]]),
        "arcsinh_cofactor": np.array([[150.0]]),
        "kernel_width_factor": np.array([[5.0]]),
        "density_estimation_optimization_factor": np.array([[1.5]]),
        "outlier_density": np.array([[1.0]]),
        "target_density_mode": np.array([[2.0]]),
        "target_density": np.array([[3.0]]),
        "target_cell_number": np.array([[20_000.0]]),
        "max_allowable_events": np.array([[50_000.0]]),
        "number_of_desired_clusters": np.array([[40.0]]),
        "file_used_to_build_SPADE_tree": cellstr(["Nilsson_rare"]),
        "clustering_algorithm": "kmeans",
    }
    savemat(target / "SPADE_parameters.mat", parameters, do_compression=False, oned_as="row")

    protocol_by_revision = {
        "base": PROTOCOL,
        "r1": PROTOCOL_R1,
        "r2": PROTOCOL_R2,
        "r3": PROTOCOL_R3,
    }
    protocol_path = protocol_by_revision[protocol_revision]
    if protocol_revision in {"r1", "r2", "r3"} and not compat_delimiter:
        raise ValueError(f"protocol {protocol_revision} requires --compat-delimiter")
    runtime_environment_by_revision = {
        "base": {},
        "r1": {},
        "r2": {"MATLAB_DISABLE_CBWR": "1", "MKL_CBWR": "SSE4_2"},
        "r3": {"MKL_DEBUG_CPU_TYPE": "4"},
    }
    runtime_environment = runtime_environment_by_revision[protocol_revision]

    manifest = {
        "experiment_id": "EXP-044",
        "attempt": attempt,
        "status": "prepared",
        "prepared_utc": utcnow(),
        "dataset": "Nilsson_rare",
        "input_fcs": str(INPUT_FCS),
        "input_parent_sha256": sha256(INPUT_FCS),
        "input_copy": str(copied_fcs),
        "input_sha256": sha256(copied_fcs),
        "fcs_compatibility_conversion": compatibility,
        "events": int(matrix.shape[0]),
        "markers": marker_names,
        "n_markers": int(matrix.shape[1]),
        "parameters": {
            "compensation": False,
            "transformation": "SPADE flow_arcsinh",
            "arcsinh_cofactor": 150,
            "kernel_width_factor": 5,
            "density_estimation_optimization_factor": 1.5,
            "outlier_density_percentile": 1,
            "target_cell_number": 20_000,
            "max_allowable_events": 50_000,
            "clustering_algorithm": "deterministic kmeans",
            "requested_clusters": 40,
        },
        "label_used_for_fit_or_selection": False,
        "official_executable": str(SPADE_EXE),
        "official_executable_sha256": sha256(SPADE_EXE),
        "matlab_runtime_root": str(RUNTIME_ROOT),
        "matlab_runtime_core_present": (
            RUNTIME_ROOT / "runtime" / "win64" / "mclmcrrt8_5.dll"
        ).is_file(),
        "matlab_runtime_installer": str(INSTALLER),
        "matlab_runtime_installer_sha256": sha256(INSTALLER),
        "protocol_revision": protocol_revision,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "runtime_environment": runtime_environment,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(target / "prepared_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _unwrap_assignment(raw: np.ndarray) -> np.ndarray:
    value = raw
    while isinstance(value, np.ndarray) and value.dtype == object:
        if value.size != 1:
            raise ValueError(f"expected one FCS assignment cell, found {value.size}")
        value = value.reshape(-1)[0]
    return np.asarray(value).reshape(-1)


def verify(attempt: int) -> None:
    target = attempt_path(attempt)
    prepared = json.loads((target / "prepared_manifest.json").read_text(encoding="utf-8"))
    result_path = target / RESULT_NAME
    required_files = [
        target / "SPADE_parameters.mat",
        target / INPUT_NAME.replace(".fcs", ".mat"),
        target / "SPADE_pooled_downsampled_data.mat",
        result_path,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required official artifacts: {missing}")

    result = loadmat(result_path, squeeze_me=False, struct_as_record=False)
    required_variables = {
        "idx",
        "mst_tree",
        "all_assign",
        "all_fcs_filenames",
        "marker_node_average",
    }
    missing_variables = sorted(required_variables.difference(result))
    if missing_variables:
        raise ValueError(f"missing required MAT variables: {missing_variables}")

    labels_float = _unwrap_assignment(result["all_assign"])
    finite = bool(np.all(np.isfinite(labels_float)))
    integral = bool(np.allclose(labels_float, np.rint(labels_float)))
    labels = np.rint(labels_float).astype(np.int32)
    positive = bool(np.all(labels > 0))
    unique, counts = np.unique(labels, return_counts=True)

    idx = np.asarray(result["idx"]).reshape(-1)
    mst_raw = result["mst_tree"]
    mst = np.asarray(mst_raw.toarray() if sparse.issparse(mst_raw) else mst_raw, dtype=float)
    checks = {
        "input_hash_exact": sha256(target / INPUT_NAME) == prepared["input_sha256"],
        "registered_parent_hash_exact": prepared["input_parent_sha256"] == EXPECTED_INPUT_SHA256,
        "input_shape_exact": prepared["events"] == EXPECTED_EVENTS
        and prepared["n_markers"] == EXPECTED_MARKERS,
        "runtime_core_present": (RUNTIME_ROOT / "runtime" / "win64" / "mclmcrrt8_5.dll").is_file(),
        "required_files_present": not missing,
        "required_variables_present": not missing_variables,
        "assignment_length_exact": len(labels) == EXPECTED_EVENTS,
        "assignments_finite": finite,
        "assignments_integral": integral,
        "assignments_positive": positive,
        "occupied_cluster_bound": 0 < len(unique) <= 40,
        "pooled_idx_nonempty": idx.size > 0,
        "pooled_idx_integral": bool(np.all(np.isfinite(idx)) and np.allclose(idx, np.rint(idx))),
        "mst_square": mst.ndim == 2 and mst.shape[0] == mst.shape[1],
        "mst_finite": bool(np.all(np.isfinite(mst))),
        "mst_symmetric": bool(
            mst.ndim == 2 and mst.shape[0] == mst.shape[1] and np.allclose(mst, mst.T)
        ),
        "mst_node_compatible": bool(mst.ndim == 2 and mst.shape[0] >= len(unique)),
    }
    passed = all(checks.values())
    np.save(target / "cluster_ids_all_events.npy", labels, allow_pickle=False)
    pd.DataFrame({"cluster_id": unique, "events": counts}).to_csv(
        target / "cluster_sizes.csv", index=False
    )
    report = {
        "experiment_id": "EXP-044",
        "attempt": attempt,
        "verified_utc": utcnow(),
        "events": int(len(labels)),
        "requested_clusters": 40,
        "occupied_clusters": int(len(unique)),
        "pooled_clustered_events": int(idx.size),
        "mst_shape": list(mst.shape),
        "checks": checks,
        "all_checks_passed": passed,
        "artifact_sha256": {
            path.name: sha256(path)
            for path in sorted(target.iterdir())
            if path.is_file()
            and path.name not in {"validation_report.json", "artifact_hashes.json"}
        },
    }
    write_json(target / "validation_report.json", report)
    write_json(target / "artifact_hashes.json", report["artifact_sha256"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


def compare(first: int, second: int) -> None:
    path_a = attempt_path(first) / "cluster_ids_all_events.npy"
    path_b = attempt_path(second) / "cluster_ids_all_events.npy"
    labels_a = np.load(path_a, allow_pickle=False)
    labels_b = np.load(path_b, allow_pickle=False)
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"repeat shape mismatch: {labels_a.shape} != {labels_b.shape}")
    ari = float(adjusted_rand_score(labels_a, labels_b))
    report = {
        "experiment_id": "EXP-044",
        "compared_utc": utcnow(),
        "attempts": [first, second],
        "events": int(labels_a.size),
        "numeric_labels_exact": bool(np.array_equal(labels_a, labels_b)),
        "partition_ari": ari,
        "partition_exact_up_to_permutation": bool(np.isclose(ari, 1.0, atol=1e-12, rtol=0)),
        "label_file_sha256": {str(first): sha256(path_a), str(second): sha256(path_b)},
    }
    write_json(RUN_ROOT / "repeat_comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["partition_exact_up_to_permutation"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--attempt", type=int, required=True)
    prep.add_argument("--compat-delimiter", action="store_true")
    prep.add_argument("--protocol-revision", choices=["base", "r1", "r2", "r3"], default="base")
    check = sub.add_parser("verify")
    check.add_argument("--attempt", type=int, required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--first", type=int, default=0)
    comp.add_argument("--second", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if RUNTIME_ROOT is None:
        raise SystemExit("Set MATLAB_RUNTIME_ROOT to the MATLAB Runtime 8.5 installation directory")
    if args.command == "prepare":
        prepare(args.attempt, args.compat_delimiter, args.protocol_revision)
    elif args.command == "verify":
        verify(args.attempt)
    else:
        compare(args.first, args.second)


if __name__ == "__main__":
    main()
