# Datasets

The five public datasets come from HDCytoData on ExperimentHub. The accession IDs are listed below; event counts and label-handling rules are recorded in [datasets.csv](datasets.csv).

| Dataset | ExperimentHub | Analysis input |
| --- | --- | --- |
| Levine_13dim | EH2242 | 13 markers; local TXT already contains `asinh(raw/5)` values |
| Levine_32dim | EH2240 | 32 markers; apply `asinh(x/5)` once |
| Samusik_01 | EH2244 | 39 markers; apply `asinh(x/5)` once |
| Nilsson_rare | EH2248 | 13 markers; apply `asinh(x/150)` once |
| Mosmann_rare | EH2250 | 14 markers; apply `asinh(x/150)` once |

The HDCytoData EH2242 expression object is untransformed. The Levine_13dim TXT used in the analyses is its transformed counterpart, so that file should not be transformed a second time. For the multiclass datasets, unassigned events may enter a full-event fit but are excluded from external-label scoring. The rare datasets use an evaluable binary target-versus-other reference.

The processed tables under `analysis/inputs` and the contingency tables under `posthoc/source` are included so the figures and tables can be recreated without downloading the event-level matrices.

To rerun the clustering experiments, place the input files under `data/raw/<dataset>/<filename>` using the filenames in `datasets.csv`. The default paths are defined in `experiments/config/datasets.json`. Some scripts also accept a data path on the command line or through the `CYTOMETRY_DATA_ROOT` environment variable; run a script with `--help` to see its options.
