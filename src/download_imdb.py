# src/download_imdb.py
import os
import requests
from tqdm import tqdm

DOWNLOAD_LIST = {
    "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
    "title.crew.tsv.gz": "https://datasets.imdbws.com/title.crew.tsv.gz",
    "title.principals.tsv.gz": "https://datasets.imdbws.com/title.principals.tsv.gz",
    "title.akas.tsv.gz": "https://datasets.imdbws.com/title.akas.tsv.gz",
    "name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz",
}

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

os.makedirs(RAW_DIR, exist_ok=True)

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"Already exists: {local_path}")
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(total=total, unit='iB', unit_scale=True, desc=os.path.basename(local_path)) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def main():
    print("Downloading IMDb datasets into:", RAW_DIR)
    for fname, url in DOWNLOAD_LIST.items():
        local_path = os.path.join(RAW_DIR, fname)
        try:
            download_file(url, local_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            print("If download fails, you can manually download from https://datasets.imdbws.com/ and place files into data/raw/")
    print("Done.")

if __name__ == "__main__":
    main()
