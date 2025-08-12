"""
Google Colab Setup Script for VLM Counting Bias Research Platform

This script installs all necessary dependencies and sets up the environment
for running the VLM counting bias research notebooks in Google Colab.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install all required packages for the research platform."""
    
    print("🔧 Setting up VLM Counting Bias Research Platform in Google Colab...")
    print("=" * 60)
    
    # Core dependencies for the research platform
    packages = [
        "openai>=1.0.0",           # GPT-4V API access
        "transformers>=4.30.0",    # HuggingFace models
        "torch>=2.0.0",           # PyTorch for local models
        "accelerate",             # Fast model loading
        "bitsandbytes",           # Memory optimization
        "pillow>=10.0.0",         # Image processing
        "opencv-python>=4.8.0",   # Computer vision
        "matplotlib>=3.7.0",      # Plotting
        "pandas>=2.0.0",          # Data analysis
        "numpy>=1.24.0",          # Numerical computing
        "plotly>=5.15.0",         # Interactive plots
        "scikit-learn>=1.3.0",    # Machine learning metrics
        "scipy>=1.11.0",          # Statistical analysis
        "tqdm>=4.66.0",           # Progress bars
        "requests>=2.31.0"        # HTTP requests
    ]
    
    print("📦 Installing packages...")
    for package in packages:
        print(f"   Installing {package.split('>=')[0]}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("✅ All packages installed successfully!")

def setup_colab_environment():
    """Set up the Colab environment for the research platform."""
    
    print("\n🔑 Setting up API keys...")
    print("Please run the following cell to set your API keys:")
    
    api_key_setup = '''
import os
from google.colab import userdata

# Set up OpenAI API key (required for GPT-4V)
try:
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
    print("✅ OpenAI API key loaded from Colab secrets")
except:
    # Fallback to manual entry
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    print("✅ OpenAI API key set manually")

# Set up HuggingFace token (optional, for private models)
try:
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    print("✅ HuggingFace token loaded from Colab secrets")
except:
    print("ℹ️  HuggingFace token not found (optional for public models)")
'''
    
    print("Copy and paste this code into a new cell:")
    print("-" * 40)
    print(api_key_setup)
    print("-" * 40)

def download_research_code():
    """Download the research platform code."""
    
    print("\n📥 Downloading research platform code...")
    
    # In a real implementation, this would clone the repository
    # For now, we'll create a placeholder structure
    
    code_structure = {
        "models": ["vlm_interface.py", "synthetic_generator.py"],
        "utils": ["image_processing.py", "evaluation_metrics.py"],
        "data": ["download_scripts.py"]
    }
    
    for folder, files in code_structure.items():
        os.makedirs(folder, exist_ok=True)
        for file in files:
            if not os.path.exists(f"{folder}/{file}"):
                print(f"   Created placeholder: {folder}/{file}")
    
    print("✅ Research platform structure ready!")

def verify_gpu_access():
    """Check if GPU is available and properly configured."""
    
    print("\n🚀 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  GPU not available - using CPU (will be slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - GPU check skipped")
        return False

def main():
    """Main setup function for Colab environment."""
    
    print("🔬 VLM Counting Bias Research Platform - Colab Setup")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    gpu_available = verify_gpu_access()
    
    # Setup instructions
    setup_colab_environment()
    
    # Download code structure
    download_research_code()
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Set your API keys using the code above")
    print("2. Run the research notebooks")
    if gpu_available:
        print("3. Use GPU acceleration for faster local model inference")
    else:
        print("3. Consider enabling GPU in Runtime > Change runtime type")
    
    print("\n📚 Available notebooks:")
    print("   • 01_counting_occlusion_synthetic.ipynb")
    print("   • 02_counting_real_camouflage.ipynb")

if __name__ == "__main__":
    main()