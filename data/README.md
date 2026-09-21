# Datasets

The five public datasets come from HDCytoData on ExperimentHub. The accession IDs are listed below; exact event counts and label-handling rules are recorded in [datasets.csv](datasets.csv).

| Dataset | ExperimentHub | Analysis input |
| --- | --- | --- |
| Levine_13dim | EH2242 | 13 markers; local TXT already contains `asinh(raw/5)` values |
| Levine_32dim | EH2240 | 32 markers; apply `asinh(x/5)` once |
| Samusik_01 | EH2244 | 39 markers; apply `asinh(x/5)` once |
| Nilsson_rare | EH2248 | 13 markers; apply `asinh(x/150)` once |
| Mosmann_rare | EH2250 | 14 markers; apply `asinh(x/150)` once |

The HDCytoData EH2242 expression object is untransformed. The Levine_13dim TXT used in the analyses was its transformed counterpart, so that file was not transformed a second time. For the multiclass datasets, unassigned events could enter a full-event fit but did not enter external-label scoring. The rare datasets used an evaluable binary target-versus-other reference.

Raw event-level matrices are not needed to rebuild the manuscript displays. The frozen tables under `analysis/inputs` and the contingency tables under `posthoc/source` are included for that purpose. Exact identities for the five original analysis files are recorded in [input_manifest.csv](input_manifest.csv).

To rerun the result-generating experiments, place the analysis files under `data/raw/<dataset>/<filename>` using the filenames in `datasets.csv`; `experiments/config/datasets.json` uses that layout from the repository root. The R retrieval and verification scripts under `experiments/src` can retrieve or check the official ExperimentHub objects. Some method-specific scripts accept an explicit data path or the `CYTOMETRY_DATA_ROOT` environment variable instead; consult each script's `--help` output.

Verify a prepared data directory before running an experiment:

```bash
python tools/reproduce.py verify-inputs --data-root data/raw
```
