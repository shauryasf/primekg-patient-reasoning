import os
import requests
import gzip

def download_series_matrix(dataset_id, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    # GEO uses a bucket system: GSE2034 is in GSE2nnn bucket
    prefix = dataset_id[:4] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{dataset_id}/matrix/{dataset_id}_series_matrix.txt.gz"
    
    print(f"Downloading {dataset_id} Series Matrix from {url}...")
    dest_file_gz = os.path.join(dest_dir, f"{dataset_id}_series_matrix.txt.gz")
    dest_file_txt = os.path.join(dest_dir, f"{dataset_id}_series_matrix.txt")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_file_gz, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Extracting {dest_file_gz}...")
        with gzip.open(dest_file_gz, 'rb') as f_in:
            with open(dest_file_txt, 'wb') as f_out:
                f_out.write(f_in.read())
                
        print(f"Successfully downloaded and extracted {dataset_id}")
        
    except Exception as e:
        print(f"Error downloading {dataset_id}: {e}")

if __name__ == "__main__":
    base_dir = "data/external_benchmark/raw"
    datasets = {
        "GSE2034": os.path.join(base_dir, "GSE2034"),
        "GSE2990": os.path.join(base_dir, "GSE2990")
    }
    
    for gse_id, path in datasets.items():
        download_series_matrix(gse_id, path)
