"""Extract clean target outcome variables for recurrence and progression prediction."""
from pathlib import Path
import pandas as pd

def load_target_outcomes(data_dir: Path) -> pd.DataFrame:
    """Extract clinical outcome targets from data_clinical_patient.txt without data leakage.
    
    Primary Target:
      - 'progression_event': 1 if PFS_STATUS is '1:PROGRESSION', 0 if '0:CENSORED'
    
    Secondary Targets (for validation / sensitivity):
      - 'recurrence_event': 1 if NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT == 'Yes', 0 if 'No' (NaN if missing)
      - 'disease_free_event': 1 if DFS_STATUS == '1:Recurred/Progressed', 0 if '0:DiseaseFree' (NaN if missing)
      - 'overall_survival_event': 1 if OS_STATUS == '1:DECEASED', 0 if '0:LIVING'
    """
    patient_file = data_dir / "data_clinical_patient.txt"
    with open(patient_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)), sep="\t")
    
    records = []
    for _, row in df.iterrows():
        pid = str(row["PATIENT_ID"]).strip()
        
        # PFS (Progression-Free Survival) - Primary Target
        pfs = str(row.get("PFS_STATUS", "")).strip()
        if "1:PROGRESSION" in pfs:
            prog_label = 1
        elif "0:CENSORED" in pfs:
            prog_label = 0
        else:
            prog_label = None
            
        # New Tumor Event (Direct Recurrence indicator)
        nte = str(row.get("NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT", "")).strip().lower()
        if nte == "yes":
            nte_label = 1
        elif nte == "no":
            nte_label = 0
        else:
            nte_label = None
            
        # DFS (Disease-Free Survival)
        dfs = str(row.get("DFS_STATUS", "")).strip()
        if "1:recurred" in dfs.lower() or "progressed" in dfs.lower():
            dfs_label = 1
        elif "0:diseasefree" in dfs.lower():
            dfs_label = 0
        else:
            dfs_label = None
            
        # OS (Overall Survival)
        os_stat = str(row.get("OS_STATUS", "")).strip()
        if "1:deceased" in os_stat.lower():
            os_label = 1
        elif "0:living" in os_stat.lower():
            os_label = 0
        else:
            os_label = None
            
        records.append({
            "patient_id": pid,
            "progression_event": prog_label,
            "recurrence_event": nte_label,
            "disease_free_event": dfs_label,
            "overall_survival_event": os_label,
            "pfs_months": pd.to_numeric(row.get("PFS_MONTHS"), errors="coerce"),
            "os_months": pd.to_numeric(row.get("OS_MONTHS"), errors="coerce")
        })
        
    out_df = pd.DataFrame(records)
    return out_df

if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path("brca_tcga_pan_can_atlas_2018")
    targets = load_target_outcomes(data_dir)
    print("Target Outcomes Extracted:")
    print(targets.info())
    print("\nProgression Event distribution (Primary Target):")
    print(targets["progression_event"].value_counts(dropna=False))
    print("\nRecurrence Event distribution:")
    print(targets["recurrence_event"].value_counts(dropna=False))
