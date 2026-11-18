# Workflow

## Input preparation

The ICGT reader excludes wholly empty records and interprets observed diploid
calls when allele-description metadata are `*`. The inferred alphabet is an
analysis coding convention, not a genomic REF/ALT or strand assignment. Ambiguous
sample matches fail; multiple genotype records for a named accession use the
preserved exact-name/match-rank/source-ID rule and are documented for sensitivity.
The Agrosavia reader uses the published TOTAL/summary target rows. Matina and
Criollo are two genotype representations of the same phenotyped collection.
Coffee continuous source genotypes are shifted from [-1,1] to [0,2], never rounded.
The source-adjusted FEM targets are used. FES is not substituted or averaged in.

## Genotype-only design

The 20% sample-missingness gate determines primary cohorts. PCA and cluster
construction retain the empirical-SD scaling used to freeze the original design.
Prediction is a separate operation, primarily using center-only marker dosage.
High-similarity connected components define a group-aware random partition; they
are not confirmed clones. The same primary memberships are inherited by declared-
allele and discordant-source-ID sensitivity cohorts.

## Prediction and density

All training-derived marker filters, imputation, centering/scaling, F-ranking and
penalty tuning are refit within inner training splits. Outer test outcomes enter
only evaluation. Random and identity-grouped five-fold designs are repeated ten
times; the fixed cluster-out design is evaluated once. Panels are unavailable when the requested size exceeds the eligible training
marker set.
Core panels are evaluated under both genotype scalings. The expanded density and
additional model comparisons use center-only scaling on the five primary genotype
representations. Family comparisons share a maximum 1,000-marker training-only
screen, alongside the full-marker ridge and mean/PC baselines. Every ordered
candidate grid and tie rule is in `config/study.json` or its numerical implementation.

## Supplementary analyses

Genomic variance uses phenotype-blind full, random and high-MAF marker sets.
The fixed PC covariates are from the full centered genotype matrix for both
kernel scalings. Estimates describe the supplied adjusted/aggregated targets;
profile-likelihood support ranges and boundary estimates must be retained.
QC and representative-member sensitivities inherit primary splits.
Size-matched evaluation uses the same outer evaluation pool and a common random
inner rule. The separate tuning diagnostic holds outer samples fixed while
comparing inner membership, sample-size matching and MSE weighting. It does not
select the best outer-test design as a new primary analysis.

## Reporting and exports

`scripts/reaggregate_frozen_predictions.py` reproduces the Dataset S1 full-marker
ridge summaries from the two supplied frozen prediction tables. It separately
groups random predictions by outer fold and by fixed genetic cluster, retains
repeat identities and original per-record training means, and restricts the
common-reference scope by accession IDs. This command does not fit models.


The reporting layer is independent of fitting. It retains repeats separately,
uses common eligible folds for paired correlation summaries, and keeps all valid
observations for error metrics. `compare-predictions` restricts two procedures to
their common test IDs; differing cluster partitions are identified, not hidden.
This comparison does not isolate a causal reference-genome effect.
Marker overlap is descriptive, not final-panel selection. TSVs retain precision;
Excel is a bounded, readable export. Fixed SNP sets, inner-selected procedures and
post-evaluation best observations must remain distinguishable.

## Preservation

Existing source folders and results are read-only. New results are separate.
Phase checkpoint reuse requires matching signatures and output hashes. A missing
upstream checkpoint is an error, not a failed scientific result. No remote operation
is performed by an analysis command.
