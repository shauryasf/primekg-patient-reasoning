"""Stream PrimeKG from Harvard Dataverse and extract the biomedical subnetwork
connecting our 500 breast cancer panel genes to:
1. Protein-Protein Interactions (PPI) between panel genes and key interactors
2. Reactome Pathways (pathway_protein)
3. Targeted Drugs (drug_protein)
4. Diseases (disease_protein, focusing on breast neoplasm/carcinoma)
"""
import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "gene_panel_500.json"
OUTPUT_PATH = ROOT / "data" / "primekg_cancer_subnetwork.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"

def load_panel_identifiers():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        panel = json.load(f)
    symbols = set(panel.keys())
    entrez_ids = {str(v["entrez_id"]) for v in panel.values() if v.get("entrez_id") is not None}
    print(f"Loaded {len(symbols)} symbols and {len(entrez_ids)} Entrez IDs from 500-gene panel.")
    return symbols, entrez_ids

def extract_subnetwork():
    symbols, entrez_ids = load_panel_identifiers()
    print("Connecting to Harvard Dataverse PrimeKG stream...")
    req = urllib.request.Request(PRIMEKG_URL, headers={"User-Agent": "Mozilla/5.0"})
    
    extracted_count = 0
    total_lines = 0
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        import io
        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")
        reader = csv.reader(text_stream)
        
        header = next(reader)
        # Expected: relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source
        rel_idx = header.index("relation")
        x_id_idx = header.index("x_id")
        x_type_idx = header.index("x_type")
        x_name_idx = header.index("x_name")
        y_id_idx = header.index("y_id")
        y_type_idx = header.index("y_type")
        y_name_idx = header.index("y_name")
        
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            
            for row in reader:
                total_lines += 1
                if total_lines % 500000 == 0:
                    print(f"Processed {total_lines:,} lines... matched {extracted_count:,} edges.")
                    
                x_id = row[x_id_idx]
                x_name = row[x_name_idx]
                x_type = row[x_type_idx]
                y_id = row[y_id_idx]
                y_name = row[y_name_idx]
                y_type = row[y_type_idx]
                rel = row[rel_idx]
                
                # Check if x or y is in our panel
                x_is_panel = (x_id in entrez_ids) or (x_name in symbols)
                y_is_panel = (y_id in entrez_ids) or (y_name in symbols)
                
                # Filter rules:
                # 1. PPI between panel genes, or panel gene to interactor
                # 2. Drug-protein interaction involving a panel gene
                # 3. Pathway-protein interaction involving a panel gene
                # 4. Disease-protein interaction involving a panel gene
                match = False
                if x_is_panel or y_is_panel:
                    if rel in ["protein_protein", "ppi"]:
                        # Keep if either is panel gene
                        match = True
                    elif "drug" in x_type or "drug" in y_type:
                        match = True
                    elif "pathway" in x_type or "pathway" in y_type:
                        match = True
                    elif "disease" in x_type or "disease" in y_type:
                        # Prioritize oncology / breast neoplasm or keep all disease associations for panel
                        match = True
                    elif rel in ["bioprocess_protein", "molfunc_protein"]:
                        # Function annotations - keep if panel
                        match = True
                        
                if match:
                    writer.writerow(row)
                    extracted_count += 1
                    
    print(f"Done! Extracted {extracted_count:,} PrimeKG edges from {total_lines:,} total records.")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_subnetwork()
