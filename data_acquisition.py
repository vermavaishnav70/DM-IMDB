"""
IMDb Data Acquisition Script

Downloads and extracts IMDb datasets from the official non-commercial datasets page.
Datasets include: title basics, ratings, crew, principals, and name basics.
"""

import os
import gzip
import shutil
import urllib.request
from pathlib import Path

# IMDb dataset URLs
IMDB_DATASETS = {
    'title_basics': 'https://datasets.imdbws.com/title.basics.tsv.gz',
    'title_ratings': 'https://datasets.imdbws.com/title.ratings.tsv.gz',
    'title_crew': 'https://datasets.imdbws.com/title.crew.tsv.gz',
    'title_principals': 'https://datasets.imdbws.com/title.principals.tsv.gz',
    'name_basics': 'https://datasets.imdbws.com/name.basics.tsv.gz'
}

def download_file(url, filename):
    """
    Download a file from a URL with progress indicator.
    
    Args:
        url (str): URL to download from
        filename (str): Local filename to save to
    """
    print(f"Downloading {filename}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100.0) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, filename, reporthook=report_progress)
    print(f"\n{filename} downloaded successfully!")

def extract_gz_file(gz_filename, output_filename):
    """
    Extract a gzip compressed file.
    
    Args:
        gz_filename (str): Input .gz file
        output_filename (str): Output extracted file
    """
    print(f"Extracting {gz_filename}...")
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(output_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"{output_filename} extracted successfully!")

def download_imdb_datasets(data_dir='data'):
    """
    Download and extract all IMDb datasets.
    
    Args:
        data_dir (str): Directory to store downloaded data
    """
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    print("Starting IMDb dataset download process...")
    print("="*60)
    
    for dataset_name, url in IMDB_DATASETS.items():
        gz_filename = os.path.join(data_dir, f"{dataset_name}.tsv.gz")
        tsv_filename = os.path.join(data_dir, f"{dataset_name}.tsv")
        
        # Check if TSV file already exists
        if os.path.exists(tsv_filename):
            print(f"\n{dataset_name} already exists. Skipping download.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Download the file
            download_file(url, gz_filename)
            
            # Extract the file
            extract_gz_file(gz_filename, tsv_filename)
            
            # Remove the .gz file to save space
            os.remove(gz_filename)
            print(f"Removed compressed file: {gz_filename}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("All datasets downloaded successfully!")
    print(f"Data stored in: {os.path.abspath(data_dir)}")
    print("="*60)

def get_dataset_info(data_dir='data'):
    """
    Display information about downloaded datasets.
    
    Args:
        data_dir (str): Directory containing the data
    """
    print("\nDataset Information:")
    print("="*60)
    
    total_size = 0
    for dataset_name in IMDB_DATASETS.keys():
        tsv_filename = os.path.join(data_dir, f"{dataset_name}.tsv")
        if os.path.exists(tsv_filename):
            size_mb = os.path.getsize(tsv_filename) / (1024 * 1024)
            total_size += size_mb
            print(f"{dataset_name:20s}: {size_mb:8.2f} MB")
        else:
            print(f"{dataset_name:20s}: Not downloaded")
    
    print("="*60)
    print(f"Total size: {total_size:.2f} MB")
    print("="*60)

if __name__ == "__main__":
    # Download all datasets
    download_imdb_datasets()
    
    # Display dataset information
    get_dataset_info()
    
    print("\nNext steps:")
    print("1. Run data_preprocessing.py to clean and prepare the data")
    print("2. Run exploratory_analysis.py to perform EDA")
    print("3. Run visualization.py to create interactive visualizations")
    print("4. Run predictive_models.py to build prediction models")
