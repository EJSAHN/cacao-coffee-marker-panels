# Validation scope

## Candidate scope

This candidate connects the preserved source-specific decoding functions and the
eight numerical modules to portable study drivers. The eight numerical files have
byte-hash regression guards. The recorded-screen adapter adds an explicit replay
input; it does not replace their fitting formulas. The code does not substitute
newly generated predictions for the approved saved results.

## Tests performed for this candidate

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
  were not part of this candidate's integration test.

## Numerical reproducibility boundary

An initial regenerated F-ranking changed the order of nearly tied features on
the test numerical backend, despite matching genotype/target values. This changed
stochastic model predictions. Replaying the recorded training-only screen order
restored the stored predictions without changing outcomes, fitted algorithms or
chosen test cases. Both routes are retained and explicitly labelled. Use the
recorded screen index and recorded partitions for exact study replay; disclose
regenerated ordering when doing a fresh analysis. No favorable test prediction is
used to choose a screen order.

## Not claimed

These checks do not constitute an all-cohort/all-model/all-fold fresh rerun from
all raw files, a Windows integration test of this candidate, a remote CI result,
or external biological validation. The current test environment did not have
xlrd, so the unchanged binary-workbook reader was not executed here. The earlier
Windows execution of that source function is historical evidence, not a new test.
The library-version record for the original run is provided separately; no
installation or upgrade is performed by these commands.

## Recheck

```text
python -m unittest discover -s tests -v
python run.py verify-input --data supplied_matched_inputs/icgt_cacao
python run_study.py --input supplied_matched_inputs --config config/study.json --out new_output --dry-run
```

For a bounded numerical test, run predict-study with --max-tasks 1 into a new
output directory. Such an output is explicitly PARTIAL_LIMITED_TEST and is not a
complete study. The approved stored results remain separate.
