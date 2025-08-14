#!/usr/bin/env python3
"""Create sample test data for the VLM Counting Bias research platform."""

import os
import json
from PIL import Image, ImageDraw
import random

def create_sample_images():
    """Create sample images with known object counts for testing."""
    
    # Create directories
    os.makedirs("data/capture/real", exist_ok=True)
    os.makedirs("data/capture/synthetic", exist_ok=True)
    
    # Sample metadata for real dataset
    real_metadata = {
        "dataset": "CAPTURe-real-sample",
        "description": "Sample real images for testing VLM counting bias",
        "images": []
    }
    
    # Sample metadata for synthetic dataset  
    synthetic_metadata = {
        "dataset": "CAPTURe-synthetic-sample", 
        "description": "Sample synthetic images for testing VLM counting bias",
        "images": []
    }
    
    # Create a few simple test images
    for i in range(5):
        # Create synthetic image with circles
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        num_circles = random.randint(2, 8)
        
        for j in range(num_circles):
            x = random.randint(20, 350)
            y = random.randint(20, 250)
            radius = random.randint(15, 30)
            color = random.choice(['red', 'blue', 'green', 'yellow', 'orange'])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        
        filename = f"test_synthetic_{i+1}.png"
        img.save(f"data/capture/synthetic/{filename}")
        
        synthetic_metadata["images"].append({
            "filename": filename,
            "true_count": num_circles,
            "object_type": "circles",
            "occlusion_level": random.randint(0, 30),
            "difficulty": "easy"
        })
        
        print(f"Created {filename} with {num_circles} circles")
    
    # Save metadata
    with open("data/capture/real/metadata.json", "w") as f:
        json.dump(real_metadata, f, indent=2)
    
    with open("data/capture/synthetic/metadata.json", "w") as f:
        json.dump(synthetic_metadata, f, indent=2)
    
    print(f"\n✓ Created {len(synthetic_metadata['images'])} test images")
    print("✓ Saved metadata files")
    print("✓ Ready to test your VLM counting bias research platform!")
    
    return True

if __name__ == "__main__":
    create_sample_images()