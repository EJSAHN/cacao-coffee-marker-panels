# Input contracts

## Matched cohort directory

Each cohort directory contains:

- `genotypes.npz`: `dosage` (samples × markers), `sample_ids`, `marker_ids`.
  Dosages are [0,2] floats or NaN. No pickle or implicit rounding is allowed.
- `phenotypes.tsv`: `sample_id` followed by numeric target columns, in exact order.
- `samples.tsv`: `sample_id`, `cluster`, `identity_component` in exact order.
- `markers.tsv`: `marker_id`, `source_chrom`, source coordinates as metadata.
- `outer_test_assignments.tsv`: scheme, repeat, fold, sample_id. Every partition
  contains each sample exactly once before minimum-fold-size eligibility gates.
- `definition.json`: dataset, scenario, reference_scenario and role. Sensitivity
  records may inherit reference_scenario for fixed selection seeds.

IDs and deterministic seed labels are computational identities, not display
labels. They must not be renamed for presentation. Display names belong in a
separate trait dictionary. Both incomplete target values and duplicate IDs fail
rather than being silently dropped or imputed.

## Recorded training-only screens

`import-screens` reads original `screened_marker_indices.npz` archives and their
adjacent `status.json` metadata from completed model comparisons. It creates an
index bound to exact matched genotype, phenotype and marker-file hashes plus
train/test ID hashes. Arrays identify the training-selected feature order for each
inner split and the final outer-training fit. They contain no test phenotype
selection. Replay checks sizes, uniqueness and training-marker eligibility.

The original raw source-code correction, upstream phenotype adjustment and the
meaning of source environments remain provenance issues. Frozen screens do not
certify independent validation; they enable a numerical replay of a documented
selection history. Do not manufacture screen archives after looking at test outcomes.

## Prediction tables

Study drivers write compressed TSVs with analysis, dataset, scenario, scaling,
scheme, repeat, fold, trait, model, chosen_model, sample_id, observed, predicted,
train_mean. A model key distinguishes every candidate setting. Training means
come from the corresponding training sample, never from test outcomes. The
single-fold convenience API also retains its older y_true/y_pred aliases.

## Explicit table catalog

```json
{"title":"Analysis results","tables":[
 {"label":"Prediction summary","path":"tables/summary.tsv","filename":"summary.tsv","workbook":true},
 {"label":"Fold details","path":"tables/folds.tsv.gz","filename":"folds.tsv.gz","workbook":false}
]}
```

An optional sha256 pins each source. Relative paths are resolved from the catalog,
not the shell directory. Output filenames cannot escape the export directory.
