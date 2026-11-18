# Reporting stored predictions

This reporting component evaluates existing out-of-fold predictions. It does not
fit, recalibrate, rank, or select prediction models. The study drivers call the same reporting implementation. Integration tests and
platform scope are recorded in [VALIDATION.md](VALIDATION.md).

## Input and execution

Use a tab-separated file, optionally gzip-compressed, with these columns:

- `dataset`, `scheme`, `repeat`, `trait`, `model`: the default evaluation key.
- `fold`, `sample_id`: the held-out fold and accession identifier.
- `observed`, `predicted`: the stored observation and prediction, on the same scale.
- `train_mean`: the phenotype mean calculated in that fold's training sample.

`train_mean` must not be calculated from the held-out observations. One accession
may occur only once per evaluation key. Each key's rows must be contiguous; fold
rows may occur in any order. An evaluation key must distinguish every setting
that changes a procedure. Include additional columns such as `design` or
`scaling` through `--group-columns` rather than combining different procedures.

From the repository root, with NumPy available:

```text
python scripts/prediction_reporting.py --input predictions.tsv.gz --output reporting
```

Example with additional settings:

```text
python scripts/prediction_reporting.py --input predictions.tsv --output reporting --group-columns dataset scheme design scaling repeat trait model
```

The output directory must not exist. The script writes `fold_metrics.tsv`,
`summary.tsv`, and `report.json`. On failure, `INCOMPLETE.txt` remains and no
completed report is certified. Input bytes are hashed before and after reading.
There are no downloads, dependency installations, or remote operations. Outputs are TSV and JSON. The separate export-workbook command creates an Excel
view without changing the numerical definitions.

## Within-fold association and coverage

For fold j, Pearson r is computed between the observed and predicted values.
Where n_j > 3 and correlation is numerically defined, the descriptive summary is

```text
z_j = atanh(clip(r_j, -1 + 1e-12, 1 - 1e-12))
r_within = tanh(sum((n_j - 3) * z_j) / sum(n_j - 3))
```

Clipping counts, eligible folds, eligible observations, and coverage are reported.
An undefined correlation is not replaced with zero. Exact constant observations
or predictions, and numerically unresolved variation, are excluded only from
correlation summaries. Error metrics continue to include these observations.
Numerical uncertainty is flagged when the centered L2 norm is below
`1e-13 * abs(mean)` or the prediction-to-observation SD ratio is below `1e-10`.
The second guard covers predictions whose mean is close to zero. These are
numerical-precision rules.
Small but resolved prediction variation remains eligible.

The prediction-to-observation SD ratio below 1e-3 is a separate descriptive flag;
it is not an exclusion rule. Spearman correlations use average ranks for ties.
They are also unavailable for numerically unresolved input variation.

Fisher weights are used for descriptive aggregation, not for independent-study
meta-analysis. Training sets overlap and accessions may be related. No confidence
interval or p-value is produced. Keep repeats separate. A standard deviation
across repartitions describes that repartitioning, not independent replication.

Compare model correlations on jointly eligible groups and report coverage. The standalone command reports each
procedure separately and does not implement paired model selection or comparisons.

## Fixed genetic groups for validation-design comparisons

Random outer test folds mix genetic groups. Their within-fold correlations and
cluster-out within-cluster correlations therefore measure different targets.
For comparisons, use each sample's saved out-of-fold prediction and evaluate it
within the same fixed genetic group, separately in each repeat. Keep the
corresponding training mean on every record for pooled MSE skill. Do not call
`fold_stats` on a combined genetic group with varying training means: that API
expects one fitted outer fold. `scripts/reaggregate_frozen_predictions.py` supplies this comparison for the
stored full-marker ridge predictions in Dataset S1.

The random within-group summary combines predictions from several trained
models. It is descriptive and does not isolate a causal contribution of
relatedness, linkage disequilibrium, ancestry or environmental differences.

## Reproducing Dataset S1

`scripts/reaggregate_frozen_predictions.py` is byte-identical to the script in
Dataset S1. It reads the two archived full-marker ridge prediction tables with
fixed cluster labels, evaluates all retained tests and common-reference test-ID
scopes, and optionally checks its repeat table against the supplied reference.
The README provides the complete command. Its output tables are
`grouped_metrics_by_group.tsv`, `grouped_metrics_by_repeat.tsv`, and
`grouped_metrics_summary.tsv`, with provenance in `reaggregation_report.json`.

The archived script retains the mean-relative numerical precision convention
used to generate those tables. The general reporting API additionally applies
the prediction-SD/observation-SD guard below 1e-10. An audit found zero further
exclusions in all 301,856 saved fold records and all supplied reaggregation
groups, so both conventions give the same eligibility on these data. Retaining
the standalone script preserves its exact archive identity. New input data
should use the general reporting API's precision rule.

## Prediction error

```text
MSE skill = 1 - sum((observed - predicted)^2)
                / sum((observed - training_mean_for_that_fold)^2)
```

The denominator is the total baseline squared error over the same test records,
not an average of fold-specific skill ratios. A zero baseline denominator yields
an undefined skill. RMSE, MAE, and R-squared are also reported. R-squared uses the
variance about the pooled held-out observation mean and is not the same as MSE
skill. Neither correlation nor a positive skill establishes realized breeding gain.

## Pooled diagnostics

Pooled Pearson r, within- and between-fold cross-products, and their contributions
to pooled correlation are retained. Within-centered correlation is not an
outcome-informed adjustment of model predictions. No predictions are centered
or recalibrated for error evaluation. Training-mean-only predictions have undefined
within-fold correlation even when their pooled correlation is strongly negative.

The covariance decomposition is algebraic. It does not partition the causal
contributions of linkage disequilibrium, ancestry, relatedness, or environment.
Pooled quantities remain diagnostic; near-constant pooled predictions can also
require numerical caution. Inspect fold precision and coverage flags before
interpreting association.

## Tests and validation scope

```text
python -m unittest discover -s tests -p test_prediction_reporting.py -v
```

The 34 tests cover constant and numerically unresolved predictions, preserved
error metrics, small resolved prediction spread, Fisher aggregation, covariance
decomposition, duplicate IDs, nonfinite inputs, group boundaries, and read-only
file processing. Tests also use SciPy as an independent correlation/rank check.

On archived predictions from 28 outer folds, the command reproduced 1,224 summaries
(612 full-marker and 612 training-mean baseline summaries) from 200,124 reporting
rows. These rows include reused subjects across designs and objectives; they are
not independent samples. The largest checked summary difference was approximately
1.14e-13. All 20 affected near-constant model evaluations were also checked from
stored predictions and remained ineligible for correlation.

These component tests ran on Linux. They do not constitute a fresh raw-data-to-
results run, a Windows command-line test, or validation of the complete version 2
pipeline. No predictive model was refitted in this reporting validation.
