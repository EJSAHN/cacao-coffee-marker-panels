# Validation scope

## Scope

The source-specific readers and numerical fitting functions are connected to
portable study drivers. Core fitting functions are unchanged from the analysis
run. The recorded-screen adapter preserves training-only feature order.
Reported saved predictions remain the basis for the study tables.

## Recorded integration checks for version 2.0.0rc1

- 187 unit/interface tests passed on Linux. Tests cover numerical kernels, source
  decoding rules, input checks, recorded-screen replay, checkpoint mutation,
  reporting and paired evaluation support.
- Five decoded genotype representations were compared with the preserved matched
  matrices: 13,176,808 dosage cells agreed, including NaN positions. Coffee FEM
  targets (645 values) also agreed. The test read the original genotype TXT, VCF
  and CSV files; it did not reopen the source phenotype XLS/XLSX workbooks.
- Regeneration from the six verified decoded matrices reproduced all five primary
  sample sets, cluster/identity memberships and 15,750 recorded outer assignments.
  This tests the design builder, not the source phenotype-workbook reader.
- Integrated full-marker/core-panel, expanded-panel and model-family execution
  for one ICGT cluster holdout reproduced 3,770, 7,250 and 2,320 stored predictions,
  respectively. Maximum absolute prediction differences were below 1.0e-13 with
  recorded training-screen order used for the stochastic model-family comparison.
- Representative integrated QC sensitivity, size-matched comparison and tuning
  diagnostic reproduced 225, 130 and 2,210 stored prediction rows. The largest
  absolute difference was approximately 3.70e-10.
- Matina variance execution for the full marker set and four neutral 45-marker
  sets under two scalings/two fixed designs reproduced 100 estimates; the largest
  h-squared difference was approximately 2.63e-8.
- The reporting component retains its earlier 34 numerical tests. Integrated
  reporting of the representative outputs was also executed.
- The Excel serializer's row/cell-type, string-safety and read-only input contract
  was exercised with a stub writer. An actual Excel-engine export and rendering
  were not part of that integration check.

## Numerical reproducibility boundary

An initial regenerated F-ranking changed the order of nearly tied features on
the test numerical backend, despite matching genotype/target values. This changed
stochastic model predictions. Replaying the recorded training-only screen order
restored the stored predictions without changing outcomes, fitted algorithms or
chosen test cases. Both routes are retained and explicitly labelled. Use the
recorded screen index and recorded partitions for exact study replay; disclose
regenerated ordering when doing a fresh analysis. Screen order is fixed from training data only.

## Limitations

These checks are not an all-cohort/all-model/all-fold fresh rerun from every raw
file. Windows and Ubuntu CI test the unit/interface suite, not integrated study
execution. The new source-workbook check below was run on Linux. External
population validation is outside the current data.

## Reporting and data checks

- The published commit 39247f711a2a5344534658331013295484ddcfd6 passed
  the Windows and Ubuntu unit/interface jobs in
  [GitHub Actions run 44171515953](https://github.com/EJSAHN/cacao-coffee-marker-panels/actions/runs/44171515953).
- The version 2.0.0 source tree passed all 190 numerical/interface tests on Linux
  on 17 November 2026 (America/New_York). The three added tests cover near-zero-mean numerical noise,
  preserved ranking under resolved strong shrinkage, and actual openpyxl cell types.
- An audit of all 301,856 saved fold records found zero additional eligible
  correlations excluded by the new prediction-SD/observation-SD rule (<1e-10).
  See `precision_guard_audit.json`. Saved predictions and error metrics are unchanged.
- The public reader opened Agrosavia's original `File_S2.xls` with xlrd 2.0.1
  under Python 3.12.14. The official file MD5 matched. Both Matina and Criollo
  matched tables agreed exactly for 102 accessions and five traits each. See
  `source_xls_reader_check.json`. This is a reader check without model fitting.
- Dataset S2's seven matched matrices passed ID/order checks and exact TSV/NPZ
  value and missing-position comparisons. All 2,226 archived screen inputs passed
  original array-hash, train/test-ID and cohort-input-hash checks.

- The Dataset S1 standalone script was copied without changes and executed on all
  101,716 supplied prediction rows. Its 1,912 repeat-level summary rows matched
  the supplied reference exactly for every checked numerical field; the run
  produced 11,944 group rows. No model was fitted.
- A new audit confirmed zero additional exclusions from the SD-ratio guard in
  both the 301,856 archived fold records and all 11,944 reaggregation groups.
  See `frozen_reaggregation_validation.json`.

The linked remote CI run applies to commit `39247f7` and its 187-test suite.
The later 190-test and frozen-reaggregation checks above were run locally on Linux.
CI results for another commit must be checked on that commit's Actions page.

## Recheck

```text
python -m unittest discover -s tests -v
python run.py verify-input --data ../Ahn_CacaoCoffee_Dataset_S2_R1_v1/matched/icgt_cacao
python run_study.py --input ../Ahn_CacaoCoffee_Dataset_S2_R1_v1/matched --config config/study.json --out new_output --dry-run
```

For a bounded numerical test, run predict-study with --max-tasks 1 into a new
output directory. Such an output is explicitly PARTIAL_LIMITED_TEST and is not a
complete study. The reported stored results remain separate.
