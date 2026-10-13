# Dataset Acquisition Summary: GSE2034 and GSE2990

## Overview
This report summarizes the acquisition and clinical metadata structure of two external breast cancer benchmark datasets from the NCBI Gene Expression Omnibus (GEO).

## GSE2034
* **Source URL:** [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034)
* **Description:** Gene expression profiling of 286 lymph-node-negative breast cancer patients to predict tumor recurrence.
* **Survival Data:**
  * **Event Status:** Relapse status (0 = no relapse, 1 = relapse). 
  * **Time-to-event:** Relapse-free survival time (usually in months). 
* **Clinical Variables:** Age and ER status (Estrogen Receptor).

## GSE2990
* **Source URL:** [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2990](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2990)
* **Description:** Gene expression profiling in breast cancer (Sotiriou et al.). Includes 189 breast cancer patients.
* **Survival Data:**
  * **Event Status:** Relapse-free survival event / Overall survival status.
  * **Time-to-event:** Relapse-free survival time / Overall survival time.
* **Clinical Variables:** Contains ER status, Histological grade, and treatment status.

## Standardized Survival Mapping
To standardize these datasets for survival models (like CoxPH or Kaplan-Meier), map the raw phenotype strings as follows:

| Dataset | Standard Event Column (`event`) | Standard Time Column (`time`) |
|---------|---------------------------------|-------------------------------|
| GSE2034 | 1 (relapse), 0 (censored) | Follow-up time in months |
| GSE2990 | 1 (event), 0 (censored) | Follow-up time in months/years |

*(Note: The exact column names depend on the `GEOparse` phenotype matrix parsing, which commonly assigns these to properties like `characteristics_ch1.1`. Post-processing scripts will be needed to regex/split these string columns into numerical targets).*
