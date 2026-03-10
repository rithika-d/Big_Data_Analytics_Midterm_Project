import os
import shutil
import random
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Ensure credentials from .env are in the environment for the kaggle library
from dotenv import load_dotenv
load_dotenv()

def download_test_data(num_samples=5):
    api = KaggleApi()
    api.authenticate()

    dataset = "paultimothymooney/chest-xray-pneumonia"
    temp_dir = Path("temp_kaggle_data")
    dest_dir = Path("test_images")
    
    temp_dir.mkdir(exist_ok=True)
    dest_dir.mkdir(exist_ok=True)

    print(f"Downloading samples from {dataset}...")
    
    # Download the test set specifically
    api.dataset_download_files(dataset, path=temp_dir, unzip=True)

    # The dataset structure is typically: chest_xray/test/NORMAL and chest_xray/test/PNEUMONIA
    test_root = temp_dir / "chest_xray" / "test"
    if not test_root.exists():
        # Sometimes there's an extra nested folder
        test_root = temp_dir / "chest_xray" / "chest_xray" / "test"

    for category in ["NORMAL", "PNEUMONIA"]:
        cat_dir = test_root / category
        out_dir = dest_dir / category.lower()
        out_dir.mkdir(exist_ok=True)
        
        images = list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.jpg"))
        selected = random.sample(images, min(num_samples, len(images)))
        
        for img in selected:
            shutil.copy(img, out_dir / img.name)
            print(f"Saved {category} sample: {img.name}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"Done! Test images saved to {dest_dir.resolve()}")

if __name__ == "__main__":
    download_test_data()
