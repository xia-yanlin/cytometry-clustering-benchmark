# Datasets

The five datasets come from HDCytoData on ExperimentHub. The accession IDs are listed below; event counts and label handling are in [datasets.csv](datasets.csv).

| Dataset | ExperimentHub | Analysis input |
| --- | --- | --- |
| Levine_13dim | EH2242 | 13 markers; local TXT already contains `asinh(raw/5)` values |
| Levine_32dim | EH2240 | 32 markers; apply `asinh(x/5)` once |
| Samusik_01 | EH2244 | 39 markers; apply `asinh(x/5)` once |
| Nilsson_rare | EH2248 | 13 markers; apply `asinh(x/150)` once |
| Mosmann_rare | EH2250 | 14 markers; apply `asinh(x/150)` once |

The HDCytoData EH2242 expression object is untransformed. The Levine_13dim TXT used in the analyses is its transformed version; do not transform that TXT a second time. For the multiclass datasets, unassigned events may enter a full-event fit but do not enter external-label scoring. The rare datasets use an evaluable binary target-versus-other reference.

Place analysis input files under `data/raw/<dataset>/<filename>` using the filenames in `datasets.csv`. `experiments/config/datasets.json` uses this path when commands are launched from the repository root. You may instead set an absolute `data_root` in that config for the scripts that read it. A few scripts read `CYTOMETRY_DATA_ROOT` from the environment.

R retrieval scripts for the official objects are in `experiments/src/retrieve_hdcytodata_levine13_authoritative.R` and `experiments/src/retrieve_hdcytodata_remaining_four.R`. The `analysis/inputs` tables and `posthoc/source` contingency tables are included so the displayed results can be rebuilt without downloading the large matrices.
