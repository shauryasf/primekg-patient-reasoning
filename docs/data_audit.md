# Raw data audit

This report streams each requested raw tab-delimited file (including the large molecular matrices). Comment-prefixed cBioPortal metadata lines are excluded; the first non-comment line is the header. Missingness counts treat blank, `NA`, `N/A`, `NULL`, `NONE`, and `Unknown` as explicit missing tokens. Full column lists and all missing-count fields are in `outputs/dataset_audit.json`.

| File | Rows | Columns | Schema / first columns | Largest missing counts | Malformed rows |
| --- | --- | --- | --- | --- | --- |
| data_clinical_patient.txt | 1084 | 38 | PATIENT_ID, SUBTYPE, CANCER_TYPE_ACRONYM, OTHER_PATIENT_ID, AGE … | WEIGHT:1084, PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT:364, NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT:199, ETHNICITY:169 | 0 |
| data_clinical_sample.txt | 1084 | 19 | PATIENT_ID, SAMPLE_ID, ONCOTREE_CODE, CANCER_TYPE, CANCER_TYPE_DETAILED … | GRADE:1084, MSI_SCORE_MANTIS:50, ANEUPLOIDY_SCORE:43, TMB_NONSYNONYMOUS:18 | 0 |
| data_mutations.txt | 126252 | 111 | Hugo_Symbol, Entrez_Gene_Id, Center, NCBI_Build, Chromosome … | COSMIC:120563, HGVSp:22038, Codons:21410, RefSeq:18363 | 0 |
| data_cna.txt | 25128 | 1072 | Hugo_Symbol, Entrez_Gene_Id, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | Entrez_Gene_Id:169 | 0 |
| data_log2_cna.txt | 25128 | 1072 | Hugo_Symbol, Entrez_Gene_Id, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | Entrez_Gene_Id:169 | 0 |
| data_armlevel_cna.txt | 39 | 1087 | ENTITY_STABLE_ID, NAME, DESCRIPTION, TCGA-3C-AAAU-01, TCGA-3C-AALI-01 … | TCGA-A1-A0SO-01:39, TCGA-A2-A0CZ-01:39, TCGA-A2-A0EP-01:39, TCGA-A2-A0ST-01:39 | 0 |
| data_mrna_seq_v2_rsem.txt | 20531 | 1084 | Hugo_Symbol, Entrez_Gene_Id, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | Hugo_Symbol:13 | 0 |
| data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt | 20531 | 1084 | Hugo_Symbol, Entrez_Gene_Id, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | TCGA-3C-AAAU-01:320, TCGA-3C-AALI-01:320, TCGA-3C-AALJ-01:320, TCGA-3C-AALK-01:320 | 0 |
| data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt | 20471 | 1084 | Hugo_Symbol, Entrez_Gene_Id, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | TCGA-3C-AAAU-01:457, TCGA-3C-AALI-01:457, TCGA-3C-AALJ-01:457, TCGA-3C-AALK-01:457 | 0 |
| data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt | 20531 | 1084 | Hugo_Symbol, Entrez_Gene_Id, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | TCGA-3C-AAAU-01:794, TCGA-3C-AALI-01:794, TCGA-3C-AALJ-01:794, TCGA-3C-AALK-01:794 | 0 |
| data_methylation_hm27_hm450_merged.txt | 22601 | 1070 | ENTITY_STABLE_ID, NAME, DESCRIPTION, TRANSCRIPT_ID, TCGA-3C-AAAU-01 … | TCGA-C8-A134-01:1906, TCGA-E2-A9RU-01:274, TCGA-3C-AAAU-01:269, TCGA-BH-AB28-01:269 | 0 |
| data_methylation_hm450.txt | 6597 | 780 | ENTITY_STABLE_ID, NAME, DESCRIPTION, TRANSCRIPT_ID, TCGA-3C-AAAU-01 … | DESCRIPTION:1503, NAME:1503, TRANSCRIPT_ID:1503, TCGA-E2-A9RU-01:125 | 0 |
| data_sv.txt | 5333 | 13 | Sample_Id, Site1_Hugo_Symbol, Site1_Chromosome, Site1_Position, Site2_Hugo_Symbol … | Site2_Effect_On_Frame:1802 | 0 |
| data_protein_quantification.txt | 9733 | 106 | Composite.Element.REF, TCGA-A2-A0D0-01, TCGA-BH-A0HK-01, TCGA-C8-A12T-01, TCGA-A2-A0D2-01 … | TCGA-A2-A0YM-01:1436, TCGA-BH-A0C7-01:1432, TCGA-A2-A0SX-01:1430, TCGA-AR-A0TY-01:1116 | 0 |
| data_protein_quantification_zscores.txt | 9733 | 106 | Composite.Element.REF, TCGA-A2-A0D0-01, TCGA-BH-A0HK-01, TCGA-C8-A12T-01, TCGA-A2-A0D2-01 … | TCGA-A2-A0YM-01:1436, TCGA-BH-A0C7-01:1432, TCGA-A2-A0SX-01:1430, TCGA-AR-A0TY-01:1116 | 0 |
| data_rppa.txt | 198 | 877 | Composite.Element.REF, TCGA-EW-A2FR-01, TCGA-EW-A2FS-01, TCGA-EW-A2FW-01, TCGA-EW-A2FV-01 … | TCGA-3C-AALI-01:8, TCGA-3C-AALK-01:8, TCGA-4H-AAAK-01:8, TCGA-5T-A9QA-01:8 | 0 |
| data_rppa_zscores.txt | 198 | 877 | Composite.Element.REF, TCGA-EW-A2FR-01, TCGA-EW-A2FS-01, TCGA-EW-A2FW-01, TCGA-EW-A2FV-01 … | TCGA-3C-AALI-01:8, TCGA-3C-AALK-01:8, TCGA-4H-AAAK-01:8, TCGA-5T-A9QA-01:8 | 0 |
| data_phosphoprotein_quantification.txt | 18806 | 111 | ENTITY_STABLE_ID, NAME, DESCRIPTION, GENE_SYMBOL, PHOSPHOSITE … | TCGA-BH-A0C7-01:11231, TCGA-A2-A0SX-01:11224, TCGA-A2-A0YM-01:11222, TCGA-A2-A0SW-01:10525 | 0 |
| data_genetic_ancestry.txt | 5 | 1038 | ENTITY_STABLE_ID, NAME, TCGA-3C-AAAU-01, TCGA-3C-AALI-01, TCGA-3C-AALJ-01 … | none | 0 |
| data_resource_definition.txt | 3 | 6 | RESOURCE_ID, DISPLAY_NAME, RESOURCE_TYPE, DESCRIPTION, OPEN_BY_DEFAULT … | none | 0 |
| data_resource_patient.txt | 1248 | 3 | PATIENT_ID, RESOURCE_ID, URL | none | 0 |
| data_timeline_sample_acquisition.txt | 1081 | 13 | PATIENT_ID, START_DATE, STOP_DATE, EVENT_TYPE, SAMPLE_ID … | STOP_DATE:1081, METHOD_OF_SAMPLE_PROCUREMENT:446, COUNTRY:395, TOP_SLIDE_SUBMITTED:169 | 0 |
| data_clinical_supp_hypoxia.txt | 1079 | 4 | PATIENT_ID, BUFFA_HYPOXIA_SCORE, WINTER_HYPOXIA_SCORE, RAGNUM_HYPOXIA_SCORE | none | 0 |

## Milestone relevance

- **Used:** `data_clinical_patient.txt` (patient context), `data_clinical_sample.txt` (sample context/link), `data_mutations.txt` (MAF mutation evidence), and `data_cna.txt` (discrete GISTIC gene-level CNA).
- **Inspected but excluded:** continuous/log2 CNA and arm-level CNA are alternate resolutions; mRNA and z-score files are high-dimensional expression matrices; methylation, protein/RPPA/phosphoprotein are separate high-dimensional modalities; SV is a separate alteration modality; ancestry, resources, timeline, and hypoxia are metadata/supplemental rather than necessary for the initial graph.
- `data_cna.txt` metadata defines values as -2 homozygous deletion, -1 hemizygous deletion, 0 neutral, 1 gain, 2 high-level amplification. `data_mutations.txt` is a 111-column MAF.
