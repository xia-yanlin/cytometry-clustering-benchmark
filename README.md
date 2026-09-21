# Cytometry clustering evaluation

This repository is the reproducibility package for the revised manuscript *Reference annotations and analysis choices define what cytometry clustering benchmarks measure*. Across five public datasets, the study examines how reference labels, event inclusion, parameter settings, matching rules and repeated runs affect external clustering scores.

The package is limited to the final revision. It contains 62 source files: 55 scripts for data provenance, final experiment execution, evaluation and result generation; four scripts for manuscript figures and tables; two post hoc analysis scripts; and one unified validation and reproduction entry point. Superseded implementations, interrupted run directories and temporary debugging code are not part of this release.

## Repository guide

| Path | Contents |
| --- | --- |
| `analysis/inputs` | Frozen, machine-readable inputs used to assemble the figures and tables |
| `analysis/scripts` | Four scripts that build figure data, figures and tables |
| `posthoc` | Matching-sensitivity and objective-selection analyses for Table S7 |
| `experiments/src` | Fifty-five scripts used to retrieve or verify data and generate the final analysis results |
| `experiments/config` | Dataset filenames, marker exclusions, transformations and label policies |
| `experiments/environments` | Package and session records for the method-specific environments |
| `tools/reproduce.py` | Unified release, paper, post hoc, input and raw-data smoke checks |
| `CODE_MANIFEST.csv` | Exact inventory of the 62 retained source files and their roles |
| `RESULT_PROVENANCE.csv` | Mapping from every frozen manuscript input to its generating script |
| `results/figures` | The selected PDFs for Figures 1–3 and S1–S6 |
| `results/tables` | Machine-readable versions of the main and supplementary tables |
| `data/datasets.csv` | Dataset sizes, transformations and reference-label rules |
| `data/input_manifest.csv` | Exact filenames, byte sizes and SHA-256 hashes of the five raw inputs |

The six display and post hoc scripts have distinct roles:

| Script | Purpose |
| --- | --- |
| `analysis/scripts/build_main_figure_data.py` | Assemble the plotting datasets used by the main figures |
| `analysis/scripts/render_main_figures.py` | Render Figures 1–3 |
| `analysis/scripts/render_supplement_figures.py` | Render Figures S1–S6 |
| `analysis/scripts/build_tables.py` | Build the main and supplementary tables and the XLSX workbook |
| `posthoc/scripts/matching_sensitivity.py` | Reproduce the matching-sensitivity and ceiling analyses in Table S7 |
| `posthoc/scripts/objective_selection.py` | Reproduce the objective-selection analysis in Table S7 |

## Quick reproduction

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python tools/reproduce.py release
```

This command validates the repository, rebuilds the editable CSV, SVG, PDF, PNG and XLSX outputs, reruns both post hoc analyses, and checks the deterministic outputs against the release. The nine PDFs in `results/figures` are the selected manuscript layouts.

## Raw-data verification and smoke test

```bash
python tools/reproduce.py verify-inputs --data-root DATA_ROOT
python -m pip install -r requirements-smoke.txt
python tools/reproduce.py smoke --data-root DATA_ROOT --dataset Levine_13dim --seeds 1 --output reproduction_runs/levine13_smoke
```

The first command checks the original matrices against their released byte sizes and SHA-256 hashes. The second runs a real raw-matrix-to-metrics path for the paired event-inclusion analysis and compares Levine_13dim seed-0 metrics with the accepted run. With `--seeds 30`, it also checks the aggregate values against the released scientific summary.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for separate commands, expected outputs and the method-level boundary.

## Reproduction scope and data

The five datasets are available through HDCytoData/ExperimentHub as EH2242, EH2240, EH2244, EH2248 and EH2250. The large event-level matrices are intentionally not duplicated in this repository. See [data/README.md](data/README.md) for filenames, transformations and label handling.

There are three reproduction levels:

1. `python tools/reproduce.py release` rebuilds and verifies the manuscript evidence from the frozen analysis inputs.
2. `verify-inputs` and `smoke` check exact raw-input identity and exercise a representative raw-data-to-metrics path.
3. The selected scripts under [`experiments/src`](experiments/README.md) document and rerun method-specific analyses from raw data or accepted parent outputs. These runs require their recorded Python, R, Java or MATLAB environments and, for some finalization scripts, the parent artifacts named in their command-line arguments.

Named implementations such as Vortex X-shift, FlowSOM, PhenoGraph and Deterministic-SPADE remain external projects. This repository provides the orchestration, evaluation and analysis code used in the revision; it does not relicense or replace those implementations.

## Quality checks

The automated checks perform the following steps on every push and pull request:

1. verify the code inventory, dataset contract, result provenance, formatting, imports, common Python bug patterns, Python syntax and R syntax across all 62 source files;
2. rebuild all figure data, figures and tables;
3. confirm that the rebuilt figure-data CSVs have not drifted from the tracked versions;
4. rerun both post hoc analyses; and
5. compare the rebuilt Tables S7A–C with the released tables byte for byte.

The root `requirements.txt` pins the direct dependencies for figure, table and post hoc reproduction. `requirements-dev.txt` adds only the formatter and linter used by the automated checks. Method-specific dependencies are recorded under `experiments/environments` because the original runs used separate environments.

## Citation

If you use this repository, please cite the accompanying manuscript, *Reference annotations and analysis choices define what cytometry clustering benchmarks measure*, and this software repository. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff). The archive DOI can be added after a release has been deposited.

## License

The original code in this repository is available under the [MIT License](LICENSE). The public datasets, named method implementations and other third-party materials retain their own licenses and terms; the MIT License in this repository does not replace them.
