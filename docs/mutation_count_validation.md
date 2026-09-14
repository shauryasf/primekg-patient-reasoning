# Mutation-count validation

Neither `data_clinical_patient.txt` nor `data_clinical_sample.txt` contains a `MUTATION_COUNT` field, so there is no reported cBioPortal summary against which to compare. The pipeline therefore never labels its count as a reported clinical feature. `mutation_record_count` is explicitly the number of raw rows in `data_mutations.txt` grouped by `Tumor_Sample_Barcode`, including all MAF classifications represented in the source.

| Patient ID | Sample ID | Independent raw MAF row count | Reported-count comparison |
| --- | --- | --- | --- |
| TCGA-3C-AAAU | TCGA-3C-AAAU-01 | 50 | No MUTATION_COUNT field in this bundle |
| TCGA-A8-A08G | TCGA-A8-A08G-01 | 62 | No MUTATION_COUNT field in this bundle |
| TCGA-B6-A1KI | TCGA-B6-A1KI-01 | 33 | No MUTATION_COUNT field in this bundle |
| TCGA-E2-A14R | TCGA-E2-A14R-01 | 153 | No MUTATION_COUNT field in this bundle |
| TCGA-Z7-A8R6 | TCGA-Z7-A8R6-01 | 117 | No MUTATION_COUNT field in this bundle |

Each mutation edge is keyed by its original MAF row index. Automated tests assert that graph mutation-edge counts equal the independently grouped raw MAF-row count for deterministic source samples. This establishes the chosen representation without inventing a clinical mutation-burden formula.
