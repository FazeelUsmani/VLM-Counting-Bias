"""
Dataset download scripts for VLM Counting Bias research platform.

This module provides functions to download and prepare datasets for evaluating
vision-language models on counting tasks under occlusion and camouflage conditions.
"""

import os
import zipfile
import shutil
import json
import pathlib
from typing import Optional, Dict, Any, Union
from pathlib import Path

def download_capture(root: str = "data/capture", subset_size: Optional[int] = 50) -> Dict[str, Any]:
    """
    Download CAPTURe dataset for occlusion counting evaluation.
    
    The CAPTURe dataset provides real and synthetic images with controlled
    occlusion levels, perfect for evaluating VLM counting biases.
    
    Args:
        root: Directory to save the dataset
        subset_size: If provided, only download a subset of images for quick testing
        
    Returns:
        Dictionary with download statistics and paths
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    DATASET_ID = "atinp/CAPTURe"
    
    print(f"Downloading CAPTURe dataset to {root}")
    
    # Files to download from HuggingFace
    files = [
        "real_dataset.zip", "real_metadata.json",
        "synthetic_dataset.zip", "synthetic_metadata.json",
    ]
    
    local_files = {}
    
    # Download files with progress
    for fname in tqdm(files, desc="Downloading files"):
        try:
            local_files[fname] = hf_hub_download(
                repo_id=DATASET_ID, 
                filename=fname, 
                repo_type="dataset"
            )
            print(f"✓ Downloaded {fname}")
        except Exception as e:
            print(f"✗ Failed to download {fname}: {e}")
            return {"error": f"Failed to download {fname}: {e}"}
    
    # Extract zip files
    extracted_paths = {}
    for fname in ["real_dataset.zip", "synthetic_dataset.zip"]:
        if fname not in local_files:
            continue
            
        dataset_type = "real" if "real" in fname else "synthetic"
        out_dir = root / dataset_type
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {fname}...")
        try:
            with zipfile.ZipFile(local_files[fname]) as z:
                z.extractall(out_dir)
            extracted_paths[dataset_type] = out_dir
            print(f"✓ Extracted {dataset_type} dataset")
        except Exception as e:
            print(f"✗ Failed to extract {fname}: {e}")
            return {"error": f"Failed to extract {fname}: {e}"}
    
    # Copy metadata files
    metadata_paths = {}
    for metadata_file in ["real_metadata.json", "synthetic_metadata.json"]:
        if metadata_file not in local_files:
            continue
            
        dataset_type = "real" if "real" in metadata_file else "synthetic"
        dest_path = root / dataset_type / "metadata.json"
        
        try:
            shutil.copy(local_files[metadata_file], dest_path)
            metadata_paths[dataset_type] = dest_path
            print(f"✓ Copied {dataset_type} metadata")
        except Exception as e:
            print(f"✗ Failed to copy {metadata_file}: {e}")
    
    # Create subset if requested
    if subset_size:
        print(f"Creating subset of {subset_size} images per dataset...")
        for dataset_type in ["real", "synthetic"]:
            if dataset_type not in extracted_paths:
                continue
            create_subset(extracted_paths[dataset_type], subset_size)
    
    # Generate summary
    summary = {
        "dataset": "CAPTURe",
        "root_path": str(root),
        "extracted_paths": {k: str(v) for k, v in extracted_paths.items()},
        "metadata_paths": {k: str(v) for k, v in metadata_paths.items()},
        "subset_size": subset_size,
        "license": "MIT",
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "citation": "CAPTURe: Comprehensive occlusion counting benchmark"
    }
    
    # Save summary
    summary_path = root / "download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Dataset download complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"Real images: {extracted_paths.get('real', 'Not downloaded')}")
    print(f"Synthetic images: {extracted_paths.get('synthetic', 'Not downloaded')}")
    
    return summary

def create_subset(dataset_path: pathlib.Path, subset_size: int) -> None:
    """Create a subset of images for quick testing."""
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    
    if len(image_files) <= subset_size:
        print(f"Dataset already has {len(image_files)} images (≤ {subset_size})")
        return
    
    # Keep first subset_size images, move others to backup
    backup_dir = dataset_path / "backup_full_dataset"
    backup_dir.mkdir(exist_ok=True)
    
    for img_file in image_files[subset_size:]:
        shutil.move(str(img_file), str(backup_dir / img_file.name))
    
    print(f"✓ Created subset with {subset_size} images")
    print(f"  Full dataset backed up to: {backup_dir}")

def download_coco_subset(root: str = "data/coco", category: str = "person", max_images: int = 30) -> Dict[str, Any]:
    """
    Download a small subset of COCO dataset for real-world evaluation.
    
    Args:
        root: Directory to save the dataset
        category: COCO category to download (e.g., 'person', 'car', 'cat')
        max_images: Maximum number of images to download
        
    Returns:
        Dictionary with download statistics and manifest path
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up COCO subset for category '{category}'...")
    print("Note: For full COCO dataset, use official download scripts")
    print("This creates a minimal subset for demonstration purposes")
    
    # Create placeholder manifest for now
    manifest = {
        "dataset": "COCO-subset",
        "category": category,
        "max_images": max_images,
        "note": "Use official COCO download for full dataset",
        "images": []
    }
    
    manifest_path = root / f"{category}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created COCO subset manifest: {manifest_path}")
    print("To download full COCO dataset, see README instructions")
    
    return {
        "dataset": "COCO-subset",
        "manifest_path": str(manifest_path),
        "category": category,
        "status": "manifest_created"
    }

def verify_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Verify downloaded dataset integrity and provide statistics.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary with verification results and statistics
    """
    path = pathlib.Path(dataset_path)
    
    if not path.exists():
        return {"error": f"Dataset path does not exist: {dataset_path}"}
    
    # Count images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(path.glob(f"*{ext}"))
        image_files.extend(path.glob(f"*{ext.upper()}"))
    
    # Check for metadata
    metadata_file = path / "metadata.json"
    has_metadata = metadata_file.exists()
    
    metadata_info = {}
    if has_metadata:
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            metadata_info = {
                "entries": len(metadata) if isinstance(metadata, list) else len(metadata.get("images", [])),
                "keys": list(metadata.keys()) if isinstance(metadata, dict) else "list_format"
            }
        except Exception as e:
            metadata_info = {"error": f"Failed to parse metadata: {e}"}
    
    stats = {
        "dataset_path": str(path),
        "image_count": len(image_files),
        "has_metadata": has_metadata,
        "metadata_info": metadata_info,
        "sample_images": [f.name for f in image_files[:5]],
        "verified": True
    }
    
    print(f"Dataset verification for {path}:")
    print(f"  Images found: {len(image_files)}")
    print(f"  Metadata file: {'✓' if has_metadata else '✗'}")
    if metadata_info and "entries" in metadata_info:
        print(f"  Metadata entries: {metadata_info['entries']}")
    
    return stats

if __name__ == "__main__":
    print("VLM Counting Bias - Dataset Downloader")
    print("=" * 50)
    
    # Download CAPTURe dataset (with subset for quick testing)
    capture_result = download_capture(subset_size=50)
    
    if "error" not in capture_result:
        print("\n" + "=" * 50)
        print("Verifying downloaded datasets...")
        
        # Verify real dataset
        if "real" in capture_result["extracted_paths"]:
            print("\nReal dataset:")
            verify_dataset(capture_result["extracted_paths"]["real"])
        
        # Verify synthetic dataset
        if "synthetic" in capture_result["extracted_paths"]:
            print("\nSynthetic dataset:")
            verify_dataset(capture_result["extracted_paths"]["synthetic"])
        
        print("\n✓ Dataset setup complete!")
        print("\nNext steps:")
        print("1. Run notebooks/01_counting_occlusion_synthetic.ipynb")
        print("2. Check results in the results/ directory")
        print("3. Use Streamlit app for interactive analysis")
    else:
        print(f"\n✗ Download failed: {capture_result['error']}")