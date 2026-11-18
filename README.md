# cacao-coffee-marker-panels

Genomic prediction, validation-design comparison, and low-density SNP panel
evaluation for public cacao and coffee genotype–phenotype data.

Version 2.0.0 provides source-specific readers, nested prediction and marker-panel
procedures, recorded-partition replay, and reporting from saved predictions.
See [changes from version 1](CHANGELOG.md), [data access](docs/DATA_ACCESS.md),
and [validation scope](docs/VALIDATION.md).

## Requirements

Python 3.10 or later, NumPy, SciPy, pandas, scikit-learn, openpyxl and xlrd are
required. `requirements.txt` gives supported ranges. The original Windows analysis
versions are recorded in `requirements-reproduction.txt`.

Run commands from the repository directory; editable installation is optional.
To check the numerical and interface tests:

```text
python -m unittest discover -s tests -v
```

## Recalculate reported summaries without fitting

Dataset S1 contains the saved primary and size-matched full-marker ridge
predictions, fixed genetic-cluster labels, and reference summary tables.
Extract `Ahn_CacaoCoffee_Dataset_S1_Tables_R1_v1.zip` beside this repository, then run:

```text
python scripts/reaggregate_frozen_predictions.py --input-dir ../Ahn_CacaoCoffee_Dataset_S1_Tables_R1_v1/reaggregation --output-dir reaggregated_results --check-against ../Ahn_CacaoCoffee_Dataset_S1_Tables_R1_v1/reaggregation/grouped_metrics_by_repeat.tsv
```

This script is identical to the copy in Dataset S1. It requires only NumPy and
pandas, reads stored predictions and each prediction's original training mean,
and produces group-level, repeat-level and summary TSVs. It reports both
outer-fold and fixed-cluster correlations, eligible group/sample coverage,
training-mean-relative MSE skill, and pooled covariance diagnostics.
[Reporting definitions](docs/PREDICTION_REPORTING.md) explain the two aggregation
units and the precision conventions.

Dataset S1 and Dataset S2 are supplied as supplementary archives with the study.
They are not bundled in this code repository. A public supplementary-material URL
or dataset DOI has not yet been assigned; [data access](docs/DATA_ACCESS.md) lists
the exact required files and the current availability boundary.

## Fit the study procedures from recorded inputs

Dataset S2 contains seven matched input directories, recorded sample/marker IDs,
cluster and identity labels, validation assignments, and 2,226 model-screen
archives. Extract `Ahn_CacaoCoffee_Dataset_S2_R1_v1.zip` beside this repository.

```text
python run.py verify-input --data ../Ahn_CacaoCoffee_Dataset_S2_R1_v1/matched/icgt_cacao
python run_study.py --input ../Ahn_CacaoCoffee_Dataset_S2_R1_v1/matched --config config/study.json --out study_output --recorded-screens ../Ahn_CacaoCoffee_Dataset_S2_R1_v1/model_screens/index.json
```

The first command verifies an input; the second fits models into a new output
directory. Recorded training-only screen order preserves feature ordering for
stochastic models. The code checks screen eligibility, input hashes and train/test
IDs. Some inner partitions are regenerated from the recorded labels and seeds;
the archived assignments support comparison. Use the recorded data and original
environment specification for numerical replay. See [input contracts](docs/INPUTS.md).

## Prepare new inputs from provider files

Obtain the files in [DATA_SOURCES.md](DATA_SOURCES.md) and fill in explicit local
paths in a copy of `config/sources.example.json`.

```text
python run.py prepare --sources sources.json --out decoded
python run.py design --matrices decoded --config config/study.json --out matched_new
python run_study.py --input matched_new --config config/study.json --out study_output_new
```

This route recomputes genotype-only structure and partitions. Library versions and
floating-point near-ties may affect regenerated memberships and stochastic-model
feature order. The recorded-input route is the reference for study replay.
Source positions with uncertain units are retained as metadata.

## Reporting and other commands

`scripts/prediction_reporting.py` summarizes new stored prediction tables within
outer folds. `collect-results` copies explicitly listed saved tables and creates
an Excel summary. `summarize-study`, `compare-predictions`, `marker-stability`, and
`export-workbook` also operate on saved outputs. Large Excel tables are split
across worksheets; precision-preserving TSV files remain available.

`predict-study`, `variance-study`, `qc-sensitivity`, `size-matched`, and
`tuning-diagnostic` execute the procedures used by `run_study.py`. Use
`python run.py COMMAND --help` for arguments. `--max-tasks` creates a limited test.
`--resume` checks inputs, settings, code and completed checkpoint hashes. An
interrupted checkpoint without `complete.json` requires a new destination or
removal of that incomplete checkpoint before resuming.

## Interpretation

Cluster-out evaluation pairs descriptive within-cluster Pearson correlation
(Fisher-z weights n−3, with coverage) with MSE skill relative to the training mean.
Random outer folds mix genetic groups. Their within-fold correlation and
cluster-out within-cluster correlation summarize different targets; comparison
within fixed genetic groups uses the same saved predictions and retains each
record's original training mean. Random predictions within a cluster can arise
from multiple fitted models. This comparison and the covariance decomposition do
not isolate causal effects of linkage disequilibrium or relatedness.

Undefined correlations remain unavailable while their observations remain in
error calculations. Repartitioning dispersion is not an independent-sample
confidence interval. The analyses evaluate marker-density procedures; they do not
validate a fixed assay or realized breeding gain.

## Files and licensing

- `config/`: source and study settings.
- `src/cacao_coffee_panels/`: readers, fitting procedures, drivers and exporters.
- `scripts/`: stored-prediction reporting and Dataset S1 reaggregation.
- `tests/`: numerical and interface checks.
- `docs/`: input contracts, data access, reporting definitions and validation.

The MIT license applies to the code. Third-party data retain their source licenses.
Record the immutable commit used for each analysis; citation metadata are in
`CITATION.cff`. Analysis commands write to the requested local destination and do
not download data or modify this repository remotely.
