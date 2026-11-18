# Changelog

## 2.0.0

- Add the exact Dataset S1 no-fit reaggregation script for saved full-marker
  predictions, including outer-fold and fixed-cluster summaries, coverage,
  training-mean MSE skill and common Agrosavia test-accession scopes.
- Add a scale-relative correlation precision guard to the general reporting API.
  The additional rule excludes no further defined correlations among the 301,856
  archived fold records or the supplied reaggregation groups.
- Preserve numeric Excel fields such as `residual_df` and `inner_validation_n`,
  while keeping accession identifiers and hashes as text.
- Connect documentation to the actual Dataset S1 and Dataset S2 file contracts.
  Dataset files are separate supplementary archives; public data-access links
  remain pending.
- Retain the model-fitting procedures, configuration grids and stored predictions
  used by the integrated code. No new model or marker-panel selection is introduced.

## 2.0.0rc1 — integrated study code

- Connect source decoding, recorded matched inputs, fixed partitions, prediction,
  density comparisons, variance components and validation sensitivities.
- Keep the numerical fitting functions unchanged from the analysis run and expose
  portable study drivers with explicit input and configuration contracts.
- Report within-fold correlations with coverage and training-mean MSE skill;
  retain pooled covariance diagnostics. The reporting decision followed inspection
  of earlier results and is not preregistration. Models are not reselected using
  the new reporting metrics.
- Exclude numerically unresolved correlations only from correlation summaries.
  Their observations and errors are retained. Small resolved prediction variation
  is not excluded merely because the fitted model is strongly shrunk.
- Add recorded training-only screen replay for stochastic models. Near-tied
  floating-point scores can change feature order across backends; archived screen
  order is an explicit input checked against cohort and train/test identities.
- Export paired comparisons, marker stability, read-only result catalogs and
  multisheet workbooks. No source data or generated outputs are embedded.

### Corrections relative to version 1.0.0

The ICGT reader included wholly empty source records as marker rows and discarded
valid observed genotype calls when the allele-description field was `*`. The
asterisk was metadata and did not by itself denote missing calls. This inflated
missingness and altered sample/marker filtering. Affected ICGT matrices and analyses
were reconstructed; changing only labels or a missingness percentage is insufficient.

Full-marker and panel ridge tuning and minimum evaluation-fold sizes previously
differed. Revised comparisons use matched evaluation records and explicit common
rules. Coffee targets are explicitly the source-adjusted FEM records, not a two-
environment mean. Continuous genotype values remain unrounded. Center-only primary
scaling follows the diagnosis of unstable predictions caused by extremely small
observed training standard deviations; allele-frequency scaling is retained as a
sensitivity. These changes do not imply all original low-density improvements were
reproduced.

### Validation boundary

Validation includes numerical/interface tests, source-reader checks and documented
representative integrated fits. It does not establish an all-model, all-fold
raw-data rerun on every platform or external-population validation. The legacy
version 1 implementation remains available in the repository history.
