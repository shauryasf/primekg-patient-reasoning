import os
import pandas as pd
import numpy as np
import requests
import gzip
import io

def run():
    print("Starting GSE2034 Processing...")
    
    # 1. Load Matrix
    matrix_path = 'data/external_benchmark/raw/GSE2034/GSE2034_series_matrix.txt'
    
    # Extract labels
    labels = {}
    samples = []
    
    with open(matrix_path, 'r') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                samples = line.strip().split('\t')[1:]
                samples = [s.strip('"') for s in samples]
            elif line.startswith('!Sample_characteristics_ch1') and 'bone relapses' in line:
                chars = line.strip().split('\t')[1:]
                for s, c in zip(samples, chars):
                    val = c.strip('"').split(':')[-1].strip()
                    labels[s] = int(val)
                break
                
    print(f"Extracted {len(labels)} labels. Positives: {sum(labels.values())}")
    
    # 2. Try to get GPL96 mapping
    # Since NCBI FTP is unreliable via GEOparse in this environment, we fetch the SOFT file via HTTPS manually
    gpl_url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"
    print("Downloading GPL96 annotation...")
    try:
        resp = requests.get(gpl_url, stream=True)
        resp.raise_for_status()
        
        probe_to_gene = {}
        with gzip.open(resp.raw, 'rt') as f:
            headers = []
            for line in f:
                if line.startswith('^'): continue
                if line.startswith('!'): continue
                if line.startswith('#'):
                    headers = line.strip().lstrip('#').split('\t')
                    continue
                if not headers: continue
                
                parts = line.strip().split('\t')
                if len(parts) > 2:
                    probe = parts[0]
                    gene = parts[2] # usually Gene Symbol or Entrez
                    if gene and gene != '---':
                        probe_to_gene[probe] = gene
                        
        print(f"Mapped {len(probe_to_gene)} probes.")
    except Exception as e:
        print(f"Failed to fetch GPL96: {e}")
        return

if __name__ == "__main__":
    run()
