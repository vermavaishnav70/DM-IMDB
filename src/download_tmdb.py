import os
import shutil
import glob
import kagglehub

def download_tmdb_dataset(target_dir="data/raw"):
    """
    Download TMDB dataset using kagglehub and move to target directory.
    """
    # Create folder if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if dataset already exists
    existing_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if existing_files:
        print(f"Dataset already exists in {target_dir}. Skipping download.")
        print(f"Found: {existing_files}")
        return

    print("Downloading dataset using kagglehub...")
    try:
        # Download latest version to cache
        path = kagglehub.dataset_download("alanvourch/tmdb-movies-daily-updates")
        print(f"Dataset downloaded to cache: {path}")

        # Move files to target_dir
        print(f"Moving files to {target_dir}...")
        for file_path in glob.glob(os.path.join(path, "*")):
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(target_dir, file_name)
            # Copy instead of move to keep cache intact or just copy
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_name} to {target_dir}")
            
        print("Download and move complete! 🎉")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you are authenticated with Kaggle.")

if __name__ == "__main__":
    download_tmdb_dataset()
