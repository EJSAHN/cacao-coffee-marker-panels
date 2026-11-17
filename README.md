# cacao-coffee-marker-panels

Genomic prediction, marker-density comparisons, and validation-design sensitivity
for public cacao and coffee genotype–phenotype resources.

**Version 2.0.0rc1 is a code-review candidate, not a tagged release.** It connects
the source-specific readers, preserved numerical implementations, study drivers,
and tabular reporting. The approved study results are not overwritten by running
this code. See [validation scope](docs/VALIDATION.md) and [changes from version 1](CHANGELOG.md).

## Existing environment

Python 3.10 or later, NumPy, SciPy, pandas, scikit-learn, openpyxl and xlrd are
required. `requirements.txt` gives supported ranges; `requirements-reproduction.txt`
records the versions used in the original Windows analysis. Do not upgrade a
working shared environment merely to run this repository. No command installs
packages, downloads source data, or modifies remote repositories.

Run from the repository directory; editable installation is not necessary:

```text
python run.py verify-input --data matched/icgt_cacao
python -m unittest discover -s tests -v
```

## Two reproducibility routes

### 1. Recorded matched inputs and partitions

Use the supplied matched dosage matrices, source target values, sample/marker IDs,
and recorded validation assignments. This preserves the study cohorts and splits.

```text
python run.py import-matched --input supplied_matched_inputs --out matched
python run.py import-screens --input supplied_model_checkpoints --matched matched --out model_screens
python run_study.py --input matched --config config/study.json --out study_output --recorded-screens model_screens/index.json
```

`import-screens` is optional for new analyses. For replay of stochastic models it
preserves the recorded **training-only** F-ranking order. Near-tied scores can
otherwise reorder columns across numerical backends, changing a tree's random
feature draws. It does not insert fitted predictions or use held-out phenotypes.
Its IDs, eligibility and input hashes are checked. See [reproduction inputs](docs/INPUTS.md).

### 2. Source files to new matched inputs

Obtain the provider files in [DATA_SOURCES.md](DATA_SOURCES.md); fill in the explicit
paths in a copy of `config/sources.example.json`. No result is selected by filename
recency. Outputs must be outside the source folders.

```text
python run.py prepare --sources sources.json --out decoded
python run.py design --matrices decoded --config config/study.json --out matched_new
python run_study.py --input matched_new --config config/study.json --out study_output_new
```

This route recomputes genotype-only structure and partitions. Library versions and
floating-point near-ties may affect regenerated memberships and stochastic-model
feature order. The recorded-partition route is the reference for the reported
study. Source positions whose units are uncertain are not treated as base pairs.

## Existing results without refitting

An explicit catalog lists table paths and optional SHA-256 values:

```text
python run.py collect-results --catalog result_catalog.json --out result_export
```

The command copies tables and creates `analysis_summary.xlsx`. It does not refit,
reselect, recalibrate or reinterpret models. Large tables may be excluded from
Excel explicitly with `"workbook": false`; they remain in the TSV export. Sheets
exceeding the Excel row limit are split, never silently truncated.

## Separate analysis commands

`predict-study`, `variance-study`, `qc-sensitivity`, `size-matched`, and
`tuning-diagnostic` use the same implementations as `run_study.py`.
`summarize-study`, `compare-predictions`, `marker-stability`, and `export-workbook`
operate on saved outputs. Use `python run.py COMMAND --help` for arguments.
`--max-tasks` produces a **limited test**, not a complete study. `--resume` checks
input, settings, code and completed checkpoint hashes before reuse.

## Interpretation

The main reporting pair is descriptive within-fold Pearson correlation (Fisher-z,
weight n−3, with eligible sample/fold coverage) and error skill relative to the
training mean. Pooled correlations and within/between covariance contributions
remain diagnostics. Constant or numerically unresolved correlations are undefined,
not zero; their observations remain in error calculations. Repartitioning SD is
not an independent-sample confidence interval. No fixed assay or realized breeding
gain is validated by these simulations.

## Layout

```text
config/                     Explicit source and study settings
src/cacao_coffee_panels/     Readers, numerical kernels, drivers and exporters
scripts/prediction_reporting.py  Stored-prediction reporting entry point
run.py                      Command-line entry point
run_study.py                Ordered study execution with phase checkpoints
tests/                      Unit and interface tests
docs/                       Input contracts, reporting definitions and validation scope
```

No source data or generated results are bundled. Outputs are written only to the
user-specified destination. The MIT license applies to the code, not third-party
data. Record the immutable commit or reviewed release used for each analysis.
