import pandas as pd
import requests
import io
import os

def run():
    print("Fetching GSE2990 supplementary info...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2990/suppl/GSE2990_suppl_info.txt"
    resp = requests.get(url)
    resp.raise_for_status()
    
    # Read text
    df = pd.read_csv(io.StringIO(resp.text), sep='\t')
    
    # Filter columns for survival
    survival_cols = ['geo_accn', 'event.rfs', 'time.rfs', 'event.dmfs', 'time.dmfs']
    df_surv = df[survival_cols].copy()
    
    # Check valid RFS
    df_surv['time.rfs'] = pd.to_numeric(df_surv['time.rfs'], errors='coerce')
    df_surv['event.rfs'] = pd.to_numeric(df_surv['event.rfs'], errors='coerce')
    
    df_valid = df_surv.dropna(subset=['time.rfs', 'event.rfs'])
    
    print(f"Total GSE2990 Samples: {len(df)}")
    print(f"Valid RFS Samples: {len(df_valid)}")
    print(f"RFS Events: {df_valid['event.rfs'].sum()}")
    
    os.makedirs('outputs/gse2034_final', exist_ok=True)
    df_surv.to_csv('outputs/gse2034_final/GSE2990_survival.csv', index=False)
    
if __name__ == "__main__":
    run()
