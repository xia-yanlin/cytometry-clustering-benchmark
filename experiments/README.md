# Result-generation code

`src/` contains the 55 source files retained for the final revision: 46 Python scripts and 9 R scripts. Together they cover data retrieval and identity checks, final method execution, external evaluation, sensitivity analyses, result aggregation and verification. The four display scripts under `analysis/scripts` and the two post hoc scripts under `posthoc/scripts` bring the repository-wide total to 61 source files.

This is a dependency-closed selection. Two helper modules omitted from the earlier release candidate (`analyze_flowsom_stability30.py` and `run_flowsom_nested_marker_dimension.py`) and two direct verification dependencies (`verify_hdcytodata_levine13_numeric_identity.R` and `verify_remaining_xshift_pipeline.py`) are included. The superseded `analyze_cell_inclusion.py` and the earlier PhenoGraph finalization snapshots are not included; the current finalizers no longer require those snapshots to be present.

The files are organized by function:

- `retrieve_*`, `compare_*`, `data_audit.py` and the dataset-identity `verify_*` scripts retrieve or verify the public inputs.
- `run_*`, `cell_inclusion_kmeans.py` and `rare_kmeans_experiment.py` execute the selected clustering and sensitivity regimes.
- `evaluate_*` and `metric_validation.py` implement external evaluation and metric checks.
- `analyze_*`, `audit_*`, `finalize_*` and `replicate_multimodality_power.py` generate the accepted summaries used by the manuscript.
- `build_method_identity_matrix.py` and `increment_parameter_regime_registry_xshift_levine32.py` generate the method-identity and parameter-regime provenance tables.

No run directories, event-level partitions, temporary recovery folders or superseded scripts are tracked. Several analysis scripts read accepted parent outputs through explicit command-line arguments; the large parent artifacts are not duplicated in GitHub.

`config/datasets.json` gives marker exclusions, transformation cofactors and label policies. Set `data_root` to the directory holding the five analysis files, or use the repository's `data/raw` layout described in [data/README.md](../data/README.md). For the scripts that use it, `CYTOMETRY_DATA_ROOT` may be set instead. The SPADE runner reads `MATLAB_RUNTIME_ROOT` for MATLAB Runtime 8.5.

The analyses used separate implementation environments:

| Method or task | Recorded implementation |
| --- | --- |
| Python FlowSOM | Saeys Lab FlowSOM Python v0.2.2 |
| R FlowSOM | Bioconductor FlowSOM 2.18.0 under R 4.5.2 |
| PhenoGraph | PhenoGraph 1.5.7 with the default Louvain interface |
| X-shift | Nolan Lab Vortex standalone, 29 June 2017 rev2, under Java 8 |
| Deterministic-SPADE | Qiu Lab Windows implementation with MATLAB Runtime 8.5 |

The observed Python package lists and R session are in `environments/`. These records describe the original method environments; the root `requirements.txt` covers figure, table and post hoc analysis.

X-shift and default PhenoGraph Louvain repeat runs do not expose controlled seeds. The selected Sony proof-of-concept analysis is not a claim of event-level equivalence to Sony's cloud service. Results for named methods refer to the implementation and conditions specified in the manuscript tables, not a common-budget method ranking.

The run-specific scripts retain checks tied to the analysis inputs and parent outputs. Two early X-shift evaluations used revisions of `evaluate_xshift_cross_dataset.py` predating the current copy; their accepted numerical summaries are frozen under `analysis/inputs`, while the included script covers the later evaluated path. For a fresh raw-data rerun, create the parent outputs in experimental order and use the recorded method environments before executing dependent finalizers.
