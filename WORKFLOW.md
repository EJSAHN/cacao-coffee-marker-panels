# Workflow

The pipeline is designed to produce tabular outputs from public genotype-phenotype resources.

## 1. Raw-file inventory

`01_inventory_raw_data.py` checks expected files, records file sizes and table dimensions, and extracts VCF sample counts and variant counts.

## 2. Genotype-phenotype harmonization

`02_clean_and_harmonize.py` converts source-specific genotype formats into sample-by-marker matrices and writes matched phenotype tables. Cacao VCF calls are converted to alternate-allele dosage. Coffee SNP coding is retained in cleaned matrices and converted to dosage in downstream QC steps when needed.

## 3. Identity and structure audit

`03_identity_structure_qc.py` applies recorded missingness and MAF filters, runs PCA, assigns K-means clusters, computes missing-aware genotype similarity, identifies high-similarity sample pairs and creates validation split tables.

## 4. Prediction benchmark

`04_prediction_benchmarks.py` compares mean baseline, ridge regression, elastic net, random forest and gradient boosting under random five-fold and cluster-out validation. Feature processing is performed inside each training fold.

## 5. Prediction summary

`04b_summarize_prediction_results.py` identifies the best model for each dataset, trait and validation scheme, summarizes performance loss from random five-fold to cluster-out validation, and writes compact shortlist tables.

## 6. Marker-panel simulation

`05_marker_panel_simulation.py` evaluates 20, 45, 96, 219, 500 and 1000 SNP panels using ridge regression. Random panels are repeated. High-MAF panels and ANOVA-F selected panels are selected inside each training fold.

## 7. Panel versus full-marker comparison

`05b_panel_vs_full_ridge_digest.py` compares the best low-density panel result against the full-marker ridge benchmark and identifies minimum panel sizes reaching user-defined fractions of the full-marker ridge benchmark.

## 8. Workbook export

`06_export_analysis_workbook.py` combines key output tables into `analysis/outputs/analysis_workbook.xlsx`.
