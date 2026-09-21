# Cytometry clustering evaluation

Code and processed results for the manuscript *Reference annotations and analysis choices define what cytometry clustering benchmarks measure*. The study uses five public cytometry datasets to examine how reference labels, event inclusion, parameter settings, matching rules and repeated runs affect clustering scores.

## Repository structure

| Path | Contents |
| --- | --- |
| `analysis/inputs` | Processed inputs used to make the manuscript figures and tables |
| `analysis/scripts` | Figure and table generation scripts |
| `posthoc` | Matching-sensitivity and objective-selection analyses for Table S7 |
| `experiments` | Data preparation, method-specific runs and evaluation scripts |
| `results/figures` | Figures 1–3 and S1–S6 in PDF format |
| `results/tables` | Main and supplementary tables in machine-readable formats |
| `data` | Dataset descriptions, accessions and preprocessing details |

## Installation

Python 3.12 was used for the figure, table and post hoc analyses.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Method-specific experiments used separate Python, R, Java or MATLAB environments. The versions used in the study are listed in `experiments/environments`.

## Recreate the figures and tables

Run the following commands from the repository root:

```bash
python analysis/scripts/build_main_figure_data.py
python analysis/scripts/render_main_figures.py
python analysis/scripts/render_supplement_figures.py
python analysis/scripts/build_tables.py
```

The scripts write editable figure data and rendered outputs to `analysis/figure_data`, `analysis/figures` and `analysis/tables`. The submitted PDF figures are also available in `results/figures`.

## Recreate the Table S7 analyses

```bash
python posthoc/scripts/matching_sensitivity.py --source posthoc/source --out posthoc/output
python posthoc/scripts/objective_selection.py --source posthoc/source --out posthoc/output
```

These scripts reproduce the matching-sensitivity, matching-ceiling and objective-selection summaries reported in Table S7. They use the included contingency tables and parameter-grid results; clustering is not rerun in this step.

## Data and experimental scripts

The datasets are available through HDCytoData/ExperimentHub under accessions EH2242, EH2240, EH2244, EH2248 and EH2250. The event-level matrices are not included in this repository. See [`data/README.md`](data/README.md) for filenames, transformations and label handling.

The scripts in [`experiments/src`](experiments/src) cover data preparation, clustering runs, sensitivity analyses and evaluation. Some scripts require outputs from an earlier stage of the analysis, as described in [`experiments/README.md`](experiments/README.md). FlowSOM, PhenoGraph, Vortex X-shift and Deterministic-SPADE are external implementations and must be installed separately when rerunning those analyses.

## Citation

If you use this code, please cite the accompanying manuscript. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The code in this repository is available under the [MIT License](LICENSE). Datasets and third-party software remain subject to their original licenses.
