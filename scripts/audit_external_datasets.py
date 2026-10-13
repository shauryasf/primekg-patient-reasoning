import os
import pandas as pd
import numpy as np

def parse_series_matrix(filepath):
    metadata = {}
    expression_data = []
    in_matrix = False
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('!Sample_'):
                key = line.split('\t')[0].strip('!').strip('"')
                values = [v.strip().strip('"') for v in line.split('\t')[1:]]
                if key in metadata:
                    if isinstance(metadata[key], list) and isinstance(metadata[key][0], list):
                        metadata[key].append(values)
                    else:
                        metadata[key] = [metadata[key], values]
                else:
                    metadata[key] = values
    
    # Transpose metadata into a DataFrame
    # Note: !Sample_characteristics_ch1 often appears multiple times.
    df_meta = pd.DataFrame()
    if 'Sample_title' in metadata:
        df_meta['Sample_ID'] = metadata['Sample_title']
    
    chars = metadata.get('Sample_characteristics_ch1', [])
    if isinstance(chars, list) and len(chars) > 0 and isinstance(chars[0], list):
        for i, char_list in enumerate(chars):
            df_meta[f'char_{i}'] = char_list
    elif isinstance(chars, list) and len(chars) > 0:
        df_meta['char_0'] = chars
        
    return df_meta

if __name__ == "__main__":
    datasets = {
        "GSE2034": "data/external_benchmark/raw/GSE2034/GSE2034_series_matrix.txt",
        "GSE2990": "data/external_benchmark/raw/GSE2990/GSE2990_series_matrix.txt"
    }
    
    for gse_id, filepath in datasets.items():
        if os.path.exists(filepath):
            df = parse_series_matrix(filepath)
            print(f"\n=== Audit for {gse_id} ===")
            print(f"Total Patients/Samples: {len(df)}")
            print("Clinical Variables Available:")
            for col in df.columns:
                if 'char' in col:
                    print(f"  {col}: {df[col].iloc[0].split(':')[0] if ':' in df[col].iloc[0] else 'Mixed'}")
            
            # Show a sample of the extracted characteristics
            char_cols = [c for c in df.columns if 'char' in c]
            if char_cols:
                print("\nSample Data (First 3 rows):")
                print(df[char_cols].head(3))
        else:
            print(f"File not found: {filepath}")
