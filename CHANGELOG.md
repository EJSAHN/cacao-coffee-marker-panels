# Changelog

## Unreleased — reporting component under review

### Added

- A standalone reader for stored out-of-fold predictions, with fold-specific
  correlations, descriptive Fisher-z summaries, explicit coverage, training-mean
  MSE skill, and pooled within/between covariance diagnostics.
- Numerical-precision guards for correlations on almost constant predictions.
  Observations and errors are retained; predictions are not modified.
- Input validation, read-only hashes, tests, and an explicit reporting contract.

### Known issue in version 1.0.0

The ICGT input reader treated wholly empty source records as markers and treated
valid observed genotype calls as missing when the allele-description field was
`*`. The asterisk was metadata, not necessarily a missing genotype call. These
behaviors inflated missingness and changed downstream marker/sample filtering.
Affected ICGT matrices and downstream analyses must be reconstructed; relabeling
legacy results or changing only the reported missingness percentage is insufficient.

The corrected analytical results are maintained separately. This reporting-only
change does not fix the legacy reader or replace the legacy model drivers. The
remaining source-reader and analysis-driver integration must be reviewed before
a complete version 2 release. Existing version 1 commits and tags are preserved;
this branch does not overwrite earlier results or create a release tag.

### Reporting scope

The within-fold correlation reporting decision was made after diagnostic review
of existing results. It is not preregistered and does not turn reused validation
samples into independent external validation. Fixed procedures and training-only
model selection remain distinct; new reporting metrics do not retune or reselect
models. This change contains no completed manuscript, private data, or rendering code.
