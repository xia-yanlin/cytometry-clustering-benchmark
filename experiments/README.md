# Experimental scripts

`src/` contains the scripts used for data preparation, method-specific runs, evaluation and sensitivity analyses. They are kept as separate scripts because the methods use different software environments and several analyses build on outputs from an earlier stage.

The filenames indicate their role:

- `retrieve_*` and `compare_*` prepare or compare the public datasets;
- `run_*` scripts execute clustering and sensitivity analyses;
- `evaluate_*` scripts calculate the external clustering metrics; and
- `analyze_*` and `finalize_*` scripts assemble run-level results for the manuscript tables and figures.

Large event-level partitions and intermediate run directories are not stored in the repository. Scripts that depend on an earlier run accept the relevant input or parent directory as a command-line argument.

`config/datasets.json` gives marker exclusions, transformation cofactors and label policies. Set `data_root` to the directory holding the five analysis files, or use the repository's `data/raw` layout described in [data/README.md](../data/README.md). For the scripts that use it, `CYTOMETRY_DATA_ROOT` may be set instead. The SPADE runner reads `MATLAB_RUNTIME_ROOT` for MATLAB Runtime 8.5.

The analyses used separate implementation environments:

| Method or task | Recorded implementation |
| --- | --- |
| Python FlowSOM | Saeys Lab FlowSOM Python v0.2.2 |
| R FlowSOM | Bioconductor FlowSOM 2.18.0 under R 4.5.2 |
| PhenoGraph | PhenoGraph 1.5.7 with the default Louvain interface |
| X-shift | Nolan Lab Vortex standalone, 29 June 2017 rev2, under Java 8 |
| Deterministic-SPADE | Qiu Lab Windows implementation with MATLAB Runtime 8.5 |

The Python package lists and R session information are in `environments/`. The root `requirements.txt` covers only the figure, table and post hoc analyses.

X-shift and default PhenoGraph Louvain repeat runs do not expose controlled seeds. The selected Sony proof-of-concept analysis is not a claim of event-level equivalence to Sony's cloud service. Results for named methods refer to the implementation and conditions specified in the manuscript tables, not a common-budget method ranking.

For a full rerun, prepare the datasets first, use the recorded method environment and follow the stages implied by the script arguments. The processed inputs in `analysis/inputs` can be used when only the manuscript figures and tables are needed.
