import os
import GEOparse

def download_geo(dataset_id, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {dataset_id} to {dest_dir}...")
    try:
        # Download the SOFT file
        gse = GEOparse.get_GEO(geo=dataset_id, destdir=dest_dir, silent=True)
        print(f"Successfully downloaded {dataset_id}")
        
        # Save clinical metadata
        metadata_path = os.path.join(dest_dir, f"{dataset_id}_metadata.csv")
        gse.phenotype_data.to_csv(metadata_path)
        print(f"Metadata saved to {metadata_path}")
        
        # Save expression matrix
        matrix_path = os.path.join(dest_dir, f"{dataset_id}_expression.csv")
        gse.pivot_samples('VALUE').to_csv(matrix_path)
        print(f"Expression matrix saved to {matrix_path}")
        
    except Exception as e:
        print(f"Error downloading {dataset_id}: {e}")

if __name__ == "__main__":
    base_dir = "data/external_benchmark/raw"
    datasets = {
        "GSE2034": os.path.join(base_dir, "GSE2034"),
        "GSE2990": os.path.join(base_dir, "GSE2990")
    }
    
    for gse_id, path in datasets.items():
        download_geo(gse_id, path)
