# Data sources and target definitions

Third-party data are not redistributed in this repository. Obtain the provider
files and comply with their access and reuse conditions. Code licensing does not
supply a license for those files.

| Resource | Public identifier | Required source inputs | Analysis target |
|---|---|---|---|
| Bekele et al. (2022), ICGT cacao | https://doi.org/10.1371/journal.pone.0260907 | S2 phenotype XLSX and S10 HapMap-like TXT | The supplied accession-level floral, seed and pod traits |
| Osorio-Guarín et al. (2020), Agrosavia cacao | https://api.figshare.com/v2/articles/10028999 | File S2 XLS, File S5 Criollo VCF, File S6 Matina VCF | Published summary rows for productivity and disease AUDPC |
| Ferrão et al. (2019), Coffea canephora | https://doi.org/10.5061/dryad.1139fm7 | Pop_Int/Pop_Prec SNPdata and PHENOdata CSVs from data_set.zip | Source-adjusted FEM production, rust and green-bean targets |

The source manifest uses exact file paths, not the download URL. Source checksums
are recorded during decoding. Keep original README files with downloaded data.
Do not assume fold-specific upstream imputation or phenotype adjustment: this
pipeline controls its own training-only processing, not the original providers'
prior processing. Noninteger genotype values are preserved. Two reference mappings
of Agrosavia are not independent phenotype cohorts.

`config/sources.example.json` maps files to their roles. The two small cacao marker
resources used as background in the study are not training matrices or independent
validation datasets, and are not required to execute the revised prediction code.
Dataset S2 supplied with the study includes matched derivative data
and attribution. Bekele's PLOS supplementary data and the Agrosavia Figshare
dataset are licensed CC BY 4.0; the Ferrão Dryad dataset is CC0. The verified
license URLs and source metadata are included in Dataset S2. A public deposit URL
or DOI has not yet been assigned to that supplementary archive.

Archive identities and current access status are listed in [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).
