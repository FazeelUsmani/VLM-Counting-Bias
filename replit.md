# Overview

This is a research platform for evaluating vision-language model (VLM) counting biases under challenging conditions like occlusion and camouflage. The platform provides systematic tools for analyzing how state-of-the-art VLMs perform on object counting tasks when objects are partially hidden or blend into backgrounds. It includes both synthetic data generation capabilities and real-world evaluation tools, with support for multiple VLMs including GPT-4V, BLIP-2, and LLaVA.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Dashboard**: Interactive web interface for real-time single-image analysis with multi-model comparison
- **Jupyter Notebooks**: Research-oriented analysis environment for comprehensive experiments and reproducible results
- **Modular UI Components**: Sidebar configuration panels for API keys, model selection, and experimental parameters

## Backend Architecture
- **Unified VLM Interface**: Abstract base class system (`VLMInterface`) providing consistent API across different vision-language models
- **Model Manager**: Centralized `VLMManager` class handling model availability, initialization, and routing
- **Synthetic Data Pipeline**: Controlled image generation with precise occlusion levels and object properties
- **Evaluation Engine**: Comprehensive metrics calculation including accuracy, bias analysis, and statistical significance testing

## Data Processing Architecture
- **Image Processing Pipeline**: PIL and OpenCV-based preprocessing with smart resizing, quality enhancement, and format conversion
- **Base64 Encoding**: Standardized image representation for API communication
- **Batch Processing**: Systematic evaluation across large image sets with progress tracking
- **Metadata Management**: JSON-based storage of ground truth, experimental parameters, and results

## Model Integration Strategy
- **API-Based Models**: OpenAI GPT-4V integration via REST API with authentication handling
- **Local Models**: HuggingFace Transformers integration for BLIP-2 and LLaVA with GPU acceleration support
- **Extensible Design**: Plugin-style architecture allowing easy addition of new VLM implementations
- **Error Handling**: Robust retry mechanisms and graceful degradation for API failures

# External Dependencies

## Vision-Language Model APIs
- **OpenAI API**: GPT-4V access with API key configured in Replit Secrets (OPENAI_API_KEY)
- **HuggingFace Inference API**: BLIP-2 and LLaVA model access with token configured in Replit Secrets (HF_TOKEN)
- **HuggingFace Transformers**: Local model execution for open-source VLMs (optional for advanced usage)

## Data Sources
- **MS COCO Dataset**: Real-world image evaluation with verified object counts
- **Synthetic Generation**: Custom image creation with controlled occlusion patterns
- **Curated Test Sets**: Hand-selected challenging scenarios for bias evaluation

## Core Libraries
- **Streamlit**: Web dashboard framework for interactive analysis
- **PIL/Pillow**: Image processing and manipulation
- **OpenCV**: Advanced computer vision operations
- **NumPy/Pandas**: Numerical computing and data analysis
- **Matplotlib**: Visualization and plotting
- **SciPy/Scikit-learn**: Statistical analysis and evaluation metrics

## Development Tools
- **PyTorch**: Deep learning framework for local model execution
- **Requests**: HTTP client for API communication
- **tqdm**: Progress bars for batch processing
- **pytest**: Testing framework for smoke tests and validation