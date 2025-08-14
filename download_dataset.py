#!/usr/bin/env python3
"""Simple script to download CAPTURe dataset for testing."""

import os
import sys

def download_capture():
    """Download CAPTURe dataset using huggingface_hub."""
    try:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        
        from huggingface_hub import hf_hub_download
        
        print("Downloading CAPTURe dataset files...")
        
        # Create directories
        os.makedirs("data/capture/real", exist_ok=True)
        os.makedirs("data/capture/synthetic", exist_ok=True)
        
        # Download metadata files
        print("Downloading metadata...")
        real_meta = hf_hub_download(
            repo_id="atinp/CAPTURe", 
            filename="real_metadata.json", 
            repo_type="dataset"
        )
        
        synthetic_meta = hf_hub_download(
            repo_id="atinp/CAPTURe", 
            filename="synthetic_metadata.json", 
            repo_type="dataset"
        )
        
        # Copy metadata to proper locations
        import shutil
        shutil.copy(real_meta, "data/capture/real/metadata.json")
        shutil.copy(synthetic_meta, "data/capture/synthetic/metadata.json")
        
        print("✓ Downloaded metadata files")
        print("✓ Dataset setup complete!")
        print("Next: Run the Streamlit app or notebooks to test with your API keys")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You can continue with the demo using your own images")
        return False

if __name__ == "__main__":
    download_capture()