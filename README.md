# cacao-coffee-marker-panels

> **Development branch: reporting component only.** The new
> [`scripts/prediction_reporting.py`](scripts/prediction_reporting.py) summarizes
> stored predictions without fitting models. See
> [the reporting contract](docs/PREDICTION_REPORTING.md) and
> [known version 1 input issues](CHANGELOG.md).
> The analysis commands below are retained version 1 documentation. The legacy
> ICGT reader and model drivers are **not** the corrected version 2 pipeline and
> must not be used to reproduce the revised study. No version 2 release is
> declared on this branch.

## Retained version 1 documentation

Reproducible Python pipeline for identity-aware genomic prediction and low-density SNP panel evaluation in public cacao and coffee breeding datasets.

This repository contains table-oriented Python code for integrating public cacao and coffee genotype-phenotype resources. The workflow harmonizes genotype and phenotype matrices, audits sample identity and population structure, builds population-aware validation splits, benchmarks prediction models, and evaluates marker-panel size effects.

## What this repository does

1. Checks that the expected public input files are present.
2. Converts cacao and coffee genotype files into matched numeric matrices.
3. Audits genotype missingness, minor allele frequency, genetic structure, and high-similarity sample pairs.
4. Compares conventional random five-fold validation with genetic cluster-out validation.
5. Benchmarks full-marker prediction models.
6. Simulates low-density SNP panels under training-fold marker selection.
7. Exports tabular summaries and a single analysis workbook.

## Repository layout

```text
config/
  input_files_expected.tsv        Expected public input files and relative locations
resources/
  public_data_sources.tsv         Public source links for the input data
scripts/
  run_all.py                      Full workflow launcher
  01_inventory_raw_data.py        Raw-file inventory and table dimension checks
  02_clean_and_harmonize.py       Genotype and phenotype harmonization
  03_identity_structure_qc.py     Missingness, MAF, PCA, clusters, similarity audit
  04_prediction_benchmarks.py     Full-marker prediction benchmarks
  04b_summarize_prediction_results.py
                                  Prediction summary tables
  05_marker_panel_simulation.py   Low-density SNP panel simulations
  05b_panel_vs_full_ridge_digest.py
                                  Panel versus full-marker ridge comparison
  06_export_analysis_workbook.py  Multi-sheet analysis workbook export
src/
  tropical_perennial_breeding/    Small shared helpers and label dictionaries
```

## Input data

Raw data are not redistributed here. Download the public files described in `DATA_SOURCES.md` and place them under the relative paths listed in `config/input_files_expected.tsv`.

A project folder should have this structure before the pipeline is run:

```text
project_folder/
  data/raw/
    cacao_icgt_plos2022/
    cacao_agrosavia_g3_2020/
    cacao_identity_panels/
    coffee_canephora_dryad2018/
```

## Installation

Use an existing Python environment or create one from the included environment file.

```bash
conda env create -f environment.yml
conda activate cacao-coffee-marker-panels
```

A minimal pip-based setup is also provided.

```bash
pip install -r requirements.txt
```

## Run the workflow

From the repository root:

```bash
python scripts/run_all.py --project-root /path/to/project_folder
```

A faster run for checking installation and input layout is available:

```bash
python scripts/run_all.py --project-root /path/to/project_folder --quick
```

On Windows, from Anaconda Prompt:

```bat
run_pipeline.bat C:\path\to\project_folder
```

## Main outputs

The pipeline writes outputs into the user-specified project folder:

```text
analysis/intermediate/        Cleaned matrices, QC tables, PCA/clusters and validation splits
analysis/outputs/             Prediction tables, marker-panel tables and analysis workbook
analysis/logs/                Runtime logs
```

The final workbook is written to:

```text
analysis/outputs/analysis_workbook.xlsx
```

## Validation design

The workflow reports two complementary validation schemes:

- **Random five-fold validation:** conventional baseline for comparison with prior genomic prediction studies.
- **Cluster-out validation:** genetic clusters are withheld as test sets to evaluate prediction across population structure.

Marker panels selected by ANOVA-F are selected only within each training fold. They should be interpreted as trait-specific candidate panels for marker-density evaluation, not as finalized assays.

## Citation

Please cite the public release as:

Ahn, E. (2026). *cacao-coffee-marker-panels* (Version 1.0.0) [Computer software]. GitHub. https://github.com/EJSAHN/cacao-coffee-marker-panels

Machine-readable citation metadata are provided in `CITATION.cff`.
