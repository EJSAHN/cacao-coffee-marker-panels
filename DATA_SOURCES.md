# Data sources

Raw data are public but are not redistributed in this repository. The expected local file layout is listed in `config/input_files_expected.tsv`.

## ICGT cacao

- **Source:** Bekele et al. 2022, PLOS ONE, genome-wide association and genomic selection analysis of International Cocoa Genebank Trinidad accessions.
- **Used here:** phenotype workbook and HapMap-like SNP genotype table.
- **Files:** PLOS ONE supplementary S2 and S10.

## Agrosavia cacao

- **Source:** Osorio-Guarín et al. 2020, G3, cacao productivity and disease-resistance analysis.
- **Used here:** supplemental phenotype workbooks and VCF files aligned to Matina and Criollo references.
- **Files:** Figshare supplemental files S1-S6.

## Coffea canephora

- **Source:** Ferrão et al. 2019, *Heredity*, genomic prediction dataset for two *Coffea canephora* recurrent-selection populations (https://doi.org/10.1038/s41437-018-0105-y).
- **Used here:** SNP matrices and phenotype tables for production, leaf rust incidence and green bean yield.
- **Files:** `Pop_Int_SNPdata.csv`, `Pop_Int_PHENOdata.csv`, `Pop_Prec_SNPdata.csv`, and `Pop_Prec_PHENOdata.csv`.

## Cacao low-density SNP panel metadata

- **Source:** Gutiérrez et al. 2021, cacao 219-SNP identity and origin assignment panel.
- **Used here:** marker metadata for low-density cacao panel context.

## West Africa cacao KASP-SNP study

- **Source:** Bhattacharjee et al. 2024, Agronomy, Sierra Leone and Togo cacao germplasm fingerprinting with 20 KASP-SNPs.
- **Used here:** conceptual motivation for identity audit and low-density marker deployment. The pipeline itself does not require the raw KASP-SNP matrix from this study.
