# Initial feature selection

This policy is based on the actual fields in this study bundle, not a generic cBioPortal schema. The declarative version is in `config/features.yaml`.

## Retained evidence and context

| Original field/file | Level | Meaning/use | Graph representation | Kind |
| --- | --- | --- | --- | --- |
| `PATIENT_ID`, `data_clinical_patient.txt` | Patient | Stable cBioPortal case identifier | Patient node ID/attribute | Metadata |
| `AGE`, `SEX`, `CANCER_TYPE_ACRONYM`, `SUBTYPE`, `AJCC_PATHOLOGIC_TUMOR_STAGE`, `PATH_M_STAGE`, `data_clinical_patient.txt` | Patient | Diagnosis context; `SUBTYPE` supplies PanCan BRCA subtype | Patient attributes | Observed metadata |
| `SAMPLE_ID`, `PATIENT_ID`, `data_clinical_sample.txt` | Sample | Molecular specimen identity and patient linkage | Sample node ID/attribute and `has_sample` | Metadata |
| `ONCOTREE_CODE`, `CANCER_TYPE`, `CANCER_TYPE_DETAILED`, `TUMOR_TYPE`, `ANEUPLOIDY_SCORE`, `SAMPLE_TYPE`, `data_clinical_sample.txt` | Sample | Histology/tumor context and sample-level aneuploidy | Sample attributes | Observed metadata / summary statistic |
| `Hugo_Symbol`, `Entrez_Gene_Id`, `data_mutations.txt` | Gene/mutation | Standard symbol plus provided Entrez identifier | Gene node attributes | Observed fact identity |
| `Tumor_Sample_Barcode` plus `Variant_Classification`, `Variant_Type`, `Consequence`, `HGVSp_Short`, `HGVSc`, genomic coordinates, alleles, `IMPACT`, `FILTER`, `NCALLERS`, `data_mutations.txt` | Mutation | Source MAF call and useful annotation | `Sample -[has_mutation]-> Gene` MultiDiGraph edge attributes | Observed fact |
| `Hugo_Symbol`, `Entrez_Gene_Id`, and values -2/-1/1/2, `data_cna.txt` | Gene/sample | Non-neutral discrete GISTIC copy-number calls | `Sample -[has_CNA]-> Gene` edge with `cna_value` and `cna_category` | Observed fact |
| Raw MAF rows grouped by `Tumor_Sample_Barcode` | Sample | Reproducible record count, not a clinical TMB or reported mutation count | `mutation_record_count` sample attribute | Derived summary statistic |

`data_cna.txt` explicitly describes -2 as homozygous deletion, -1 as hemizygous deletion, 0 as neutral, 1 as gain, and 2 as high-level amplification. Neutral calls are intentionally omitted as non-evidence; the graph is not a dense copy-number matrix.

The supplied CNA matrix has duplicated gene rows. The loader deduplicates only exact same-symbol/same-Entrez/same-sample/same-value calls, logging every removed source row. When repeated rows give contradictory values for the same gene/sample, no arbitrary state is selected: all of that ambiguous gene/sample evidence is excluded and logged in the quality report. This keeps one unambiguous CNA relation per sample/gene while avoiding silent loss.

## Deliberately excluded

| File/fields | Decision and reason |
| --- | --- |
| `OS_*`, `DSS_*`, `DFS_*`, `PFS_*` in clinical patient data | Preserved in raw data only. They are outcome/evaluation variables and would leak targets if used as diagnostic evidence. |
| `RACE`, `ETHNICITY`, `GENETIC_ANCESTRY_LABEL`, `data_genetic_ancestry.txt` | Not required for the initial biological patient graph; retaining them would need a specific fairness/governance rationale. |
| `MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE`, `TMB_NONSYNONYMOUS`, `TBL_SCORE`, technical collection/site fields | Not selected for this BRCA graph milestone. TMB also overlaps conceptually with mutation burden but is not equivalent to raw MAF row count. |
| `GRADE`, `PATH_N_STAGE`, `PATH_T_STAGE`, dates, ICD/registry, treatment fields | Valid metadata but outside the initial compact feature set; some are redundant with the chosen stage/context attributes. |
| `data_log2_cna.txt`, `data_cna_hg19.seg`, `data_armlevel_cna.txt` | Alternate continuous/segment/arm-level representations. Using them with discrete gene-level GISTIC calls would duplicate CNA evidence. |
| mRNA/RSEM and all three z-score matrices | About 20k+ values per sample; intentionally excluded to keep the first graph interpretable and compact. |
| methylation, protein/RPPA, phosphoprotein, SV | Separate high-dimensional molecular modalities, deferred rather than silently merged into a graph with incompatible semantics. |
| resources, timeline sample acquisition, hypoxia | Supplemental imaging/procedural/signature metadata, not core patient/sample/mutation/CNA facts for this milestone. |

## Candidate fields absent in this release

Neither clinical table has `MUTATION_COUNT` or `FRACTION_GENOME_ALTERED`. `ANEUPLOIDY_SCORE` is present in `data_clinical_sample.txt` and retained. The pipeline does not invent the missing two summaries; it labels its MAF-row grouping as `mutation_record_count` and leaves fraction-genome-altered absent.
