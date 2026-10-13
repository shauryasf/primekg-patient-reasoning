import os
import pandas as pd
import numpy as np
import mygene
import torch
from torch_geometric.data import Data
import yaml

def load_panel():
    with open('config/features.yaml') as f:
        config = yaml.safe_load(f)
    # The current graph pipeline uses PrimeKG gene identifiers (Entrez IDs)
    return config.get('gene_panel', [])

def run():
    print("Building GSE2034 Dataset...")
    
    # 1. Parse Series Matrix for expressions and labels
    matrix_path = 'data/external_benchmark/raw/GSE2034/GSE2034_series_matrix.txt'
    
    labels = {}
    samples = []
    
    print("Parsing series matrix...")
    with open(matrix_path, 'r') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                samples = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1') and 'bone relapses' in line:
                chars = line.strip().split('\t')[1:]
                for s, c in zip(samples, chars):
                    val = c.strip('"').split(':')[-1].strip()
                    labels[s] = int(val)
                break
    
    print(f"Parsed {len(labels)} samples. Positives: {sum(labels.values())}")
    
    # Read expression data
    print("Reading expression data...")
    df = pd.read_csv(matrix_path, sep='\t', comment='!', skip_blank_lines=True)
    df = df.dropna(subset=['ID_REF'])
    df.set_index('ID_REF', inplace=True)
    df = df.astype(float)
    
    # 2. Map Probes to Genes using mygene
    print("Mapping probes to Entrez IDs via mygene...")
    probes = df.index.tolist()
    mg = mygene.MyGeneInfo()
    try:
        res = mg.querymany(probes, scopes='reporter', fields='entrezgene,symbol', species='human', returnall=True, as_dataframe=True)
        mapped = res['out']
    except Exception as e:
        print(f"MyGene error: {e}")
        return
        
    mapped = mapped.dropna(subset=['entrezgene'])
    mapped['entrezgene'] = mapped['entrezgene'].astype(str)
    
    # Create probe to entrez dictionary
    probe_to_entrez = mapped['entrezgene'].to_dict()
    
    df['entrez'] = df.index.map(probe_to_entrez)
    df_mapped = df.dropna(subset=['entrez'])
    
    print(f"Mapped {len(df_mapped)} probes to genes.")
    
    # 3. Aggregate multiple probes per gene (Mean)
    print("Aggregating duplicates via mean expression...")
    df_agg = df_mapped.groupby('entrez').mean()
    
    # 4. Standardize (Z-score normalize per gene across cohort)
    print("Z-score normalizing...")
    df_norm = (df_agg - df_agg.mean(axis=1).values.reshape(-1, 1)) / (df_agg.std(axis=1).values.reshape(-1, 1) + 1e-8)
    
    # 5. Filter to the 500-gene panel (or all available PrimeKG genes)
    # The existing pipeline loads mutations. Let's just create standard graphs from whatever genes we mapped.
    # To keep it comparable to TCGA, we will save it as a structured object.
    
    os.makedirs('data/processed/gse2034', exist_ok=True)
    
    # Save the normalized expression and labels
    df_norm.to_csv('data/processed/gse2034/expression.csv')
    pd.Series(labels, name='label').to_csv('data/processed/gse2034/labels.csv')
    
    print("GSE2034 preprocessing complete.")

if __name__ == "__main__":
    run()
