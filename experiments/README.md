# Experimental scripts

`src/` contains the selected fitting, evaluation and data-identity scripts used during the revision. They are organized by method and experiment rather than as a single command-line pipeline. Several read outputs of earlier runs from `experiments/runs`; those large event-level partitions are not tracked in GitHub. The small `runs/` and `source_data/` files included here support the table builder and dataset identity summaries.

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

The original run-specific scripts retain checks tied to the analysis inputs and parent outputs. Two early X-shift evaluations used revisions of `evaluate_xshift_cross_dataset.py` predating the copy in this repository; their accepted numerical summaries are available under `analysis/inputs`. The included current script covers the later evaluated path. For a fresh raw-data rerun, recreate parent outputs and method environments before executing dependent scripts.
