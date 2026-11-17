# Changelog

## 2.0.0rc1 — integrated code-review candidate

- Connect source decoding, recorded matched inputs, fixed partitions, prediction,
  density comparisons, variance components and validation sensitivities.
- Preserve the eight established numerical kernels; expose portable study drivers
  and explicit input/configuration contracts rather than personal filesystem paths.
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

### Release scope

This is not a published/tagged version 2 release. Validation covers the documented
readers, numerical kernels and representative integrated fits, not an all-model,
all-fold raw-data rerun on every supported platform. Earlier commits and tags must
be preserved. The candidate does not overwrite approved result tables or claim
external population validation.
