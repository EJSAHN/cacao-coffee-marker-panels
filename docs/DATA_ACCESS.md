# Data access and reproduction files

## Availability

The provider datasets linked in [DATA_SOURCES.md](../DATA_SOURCES.md) are public.
This repository contains code. Dataset S1 and Dataset S2 are separately supplied
supplementary archives for this study. Their public journal supplement URL or
independent deposit DOI has not yet been assigned. Therefore downloading the code
repository alone does not currently supply the exact recorded study inputs.

Readers with the supplementary archives can use the recorded-input and no-fit
commands in the README. The source-file route can be used with public provider
files; it regenerates structure and partitions and may differ in membership or
stochastic feature order from the recorded study. A public data link should be
added here when the supplementary archives become accessible.

## Archive identities

| File | Bytes | SHA-256 |
|---|---:|---|
| Ahn_CacaoCoffee_Dataset_S1_Tables_R1_v1.zip | 68664388 | `3c3712283bb92ab15637fbca895446f878d44b82a4fea7f629f46641e4361b44` |
| Ahn_CacaoCoffee_Dataset_S2_R1_v1.zip | 42095205 | `6ef048b36c03586dbb95ca00099956643f203a8c2334eac096f2493aeb05cfbd` |

Each ZIP contains one top-level directory with the same stem as the filename.
Keep its internal filenames and identifiers unchanged.

Dataset S1 provides full numerical tables, the saved full-marker ridge
out-of-fold predictions with fixed genetic clusters, and a standalone metric
reaggregation script. The repository includes an identical script at
`scripts/reaggregate_frozen_predictions.py`.

Dataset S2 provides seven matched dosage/target directories, sample and marker
IDs, source-ID crosswalks, fixed cluster and identity labels, outer assignments,
archived inner assignments, 2,226 training-screen order archives with an index,
size-matched partitions, and the 98 common Agrosavia test IDs. Its
`verify_archive.py` checks its manifest without fitting models. Some inner folds
are regenerated from labels/seeds by the execution code; archived assignments
permit comparison.

Source licenses and attribution travel with Dataset S2. These archives do not
contain every fitted model, every original provider download or every historical
execution folder. The README and [validation report](VALIDATION.md) distinguish
archive verification, representative numerical replay, and complete study fitting.
