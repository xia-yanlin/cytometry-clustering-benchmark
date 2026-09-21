# Cytometry clustering evaluation

Code and analysis tables for the revised cytometry clustering study. The analyses examine how reference labels, event inclusion, parameter settings and repeated runs affect external clustering scores across five public datasets.

## Contents

| Directory | Contents |
| --- | --- |
| `analysis/inputs` | Small result tables used to assemble the manuscript figures and tables |
| `analysis/scripts` | Figure and table generation code |
| `posthoc` | Alternative matching and objective-selection analyses for Table S7 |
| `experiments/src` | Method-specific fitting and evaluation scripts |
| `results/figures` | Figures 1–3 and S1–S6 as submitted PDFs |
| `results/tables` | Machine-readable versions of the manuscript and supplementary tables |
| `data/datasets.csv` | Dataset sizes, transformations and label rules |

## Recreate figures and tables

Use Python 3.12 and install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
python analysis/scripts/build_main_figure_data.py
python analysis/scripts/render_main_figures.py
python analysis/scripts/render_supplement_figures.py
python analysis/scripts/build_tables.py
```

The scripts write editable data, SVG, PDF and PNG files inside `analysis/figure_data`, `analysis/figures` and `analysis/tables`. The nine PDFs in `results/figures` are the selected manuscript layouts. Tables S7A–C are generated separately by the analyses below.

## Recreate Table S7 analyses

```bash
python posthoc/scripts/matching_sensitivity.py --source posthoc/source --out posthoc/output
python posthoc/scripts/objective_selection.py --source posthoc/source --out posthoc/output
```

The matching analysis reads 153 saved contingency tables; the objective-selection analysis reads the same nine-setting FlowSOM grids shown in Table S6. Both are descriptive post hoc analyses. The primary scores and figure-generation inputs are not changed by these scripts.

## Source data and experimental code

The five datasets are available through HDCytoData/ExperimentHub as EH2242, EH2240, EH2244, EH2248 and EH2250. See [data/README.md](data/README.md) for the input format and transformation rules. The large event-level matrices are not stored in GitHub.

The scripts in [experiments/README.md](experiments/README.md) record the method-specific runs. Named implementations such as Vortex X-shift, FlowSOM and PhenoGraph are external projects; this repository does not relabel substitute code as those methods. Experimental scripts may need the documented parent outputs and their own Python, R, Java or MATLAB environments. The commands above reproduce the figures and tables from the included intermediate results without repeating the long clustering runs.
