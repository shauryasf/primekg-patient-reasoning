import json
import pandas as pd
from pathlib import Path

# Established high-priority breast cancer / pan-cancer drivers (Knowledge-driven pillar)
KNOWN_DRIVERS = {
    "TP53", "PIK3CA", "GATA3", "CDH1", "ERBB2", "PTEN", "MAP3K1", "BRCA1", "BRCA2", 
    "RB1", "CCND1", "MYC", "ATM", "ESR1", "AKT1", "MAP2K4", "KMT2C", "ARID1A", 
    "NF1", "EGFR", "FGFR1", "FOXA1", "CBFB", "RUNX1", "PIK3R1", "SF3B1", "BAP1"
}

def generate_panel(data_dir: Path, output_file: Path, target_size: int = 500):
    print("Loading mutation data...")
    mut_path = data_dir / 'data_mutations.txt'
    with open(mut_path, 'r', encoding='utf-8') as f:
        mut_lines = [l for l in f if not l.startswith('#')]
    from io import StringIO
    df_mut = pd.read_csv(StringIO(''.join(mut_lines)), sep='\t', usecols=['Hugo_Symbol', 'Entrez_Gene_Id', 'Tumor_Sample_Barcode'])
    df_mut = df_mut.dropna(subset=['Hugo_Symbol'])
    
    # 1. Mutated samples per gene
    mut_freq = df_mut.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique().to_dict()
    entrez_map = df_mut.dropna(subset=['Entrez_Gene_Id']).groupby('Hugo_Symbol')['Entrez_Gene_Id'].first().to_dict()
    
    print("Streaming CNA data for extreme alterations (-2, 2)...")
    cna_path = data_dir / 'data_cna.txt'
    cna_header = pd.read_csv(cna_path, sep='\t', nrows=0).columns.tolist()
    sample_cols = [c for c in cna_header if c.startswith('TCGA')]
    
    cna_extreme = {}
    cna_entrez = {}
    
    for chunk in pd.read_csv(cna_path, sep='\t', chunksize=3000):
        for _, row in chunk.iterrows():
            sym = row['Hugo_Symbol']
            if pd.isna(sym): continue
            if sym not in cna_entrez and pd.notna(row['Entrez_Gene_Id']):
                cna_entrez[sym] = row['Entrez_Gene_Id']
            
            # Count only -2 and +2
            vals = pd.to_numeric(row[sample_cols], errors='coerce')
            ext_count = int(((vals == -2) | (vals == 2)).sum())
            cna_extreme[sym] = cna_extreme.get(sym, 0) + ext_count

    print("Combining modalities and ranking...")
    all_syms = set(mut_freq.keys()).union(cna_extreme.keys())
    records = []
    
    for sym in all_syms:
        m = mut_freq.get(sym, 0)
        c = cna_extreme.get(sym, 0)
        e = entrez_map.get(sym, cna_entrez.get(sym, None))
        # Ensure Entrez ID is integer if possible, else string/null
        if pd.notna(e):
            try:
                e = int(float(e))
            except:
                e = str(e)
        else:
            e = None
            
        score = m + c
        is_known = sym in KNOWN_DRIVERS
        
        # Give known drivers a massive score boost so they are guaranteed inclusion
        boosted_score = score + 10000 if is_known else score
        
        records.append({
            'Hugo_Symbol': sym,
            'Entrez_Gene_Id': e,
            'mut_samples': m,
            'cna_extreme_samples': c,
            'total_alteration_score': score,
            'boosted_score': boosted_score,
            'is_known_driver': is_known
        })
        
    df_ranked = pd.DataFrame(records).sort_values('boosted_score', ascending=False)
    
    # Select top target_size genes
    top_genes_df = df_ranked.head(target_size).copy()
    
    # Create the final panel dictionary
    panel = {}
    for _, row in top_genes_df.iterrows():
        panel[row['Hugo_Symbol']] = {
            'entrez_id': row['Entrez_Gene_Id'],
            'alteration_score': row['total_alteration_score'],
            'is_known_driver': row['is_known_driver']
        }
    
    print(f"Selected {len(panel)} genes.")
    print(f"Included {sum(top_genes_df['is_known_driver'])} known drivers.")
    print(f"Lowest included empirical score (excluding known drivers): {top_genes_df[~top_genes_df['is_known_driver']]['total_alteration_score'].min()}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(panel, f, indent=2)
    print(f"Saved gene panel to {output_file}")

if __name__ == '__main__':
    generate_panel(
        data_dir=Path('brca_tcga_pan_can_atlas_2018'),
        output_file=Path('config/gene_panel_500.json'),
        target_size=500
    )
