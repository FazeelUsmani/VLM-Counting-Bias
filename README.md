# Vision-Language Model Counting Bias Research Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FazeelUsmani/VLM-Counting-Bias/blob/main/notebooks/01_counting_occlusion_synthetic.ipynb)

A comprehensive research platform for evaluating vision-language model (VLM) counting biases under occlusion and camouflage conditions. This platform provides systematic tools for analyzing how state-of-the-art VLMs perform object counting tasks when objects are partially hidden, occluded, or camouflaged.

**Repository**: [https://github.com/FazeelUsmani/VLM-Counting-Bias](https://github.com/FazeelUsmani/VLM-Counting-Bias)

## 🔬 Research Overview

Large vision-language models (VLMs) can describe images impressively, but they often struggle with precise tasks like counting objects, especially when some objects are partially hidden or blend into their surroundings. This platform enables systematic evaluation of these limitations across multiple models and scenarios.

### Key Research Questions
- How does object occlusion affect VLM counting accuracy?
- Do different VLMs exhibit systematic biases (over-counting vs under-counting)?
- How well do models handle camouflaged objects?
- Can we quantify and predict counting performance degradation?

## 📱 Platform Interface

![VLM Counting Bias Research Platform](assets/app-screenshot.png)

The interactive Streamlit dashboard provides real-time analysis with multiple VLM comparison, accuracy metrics, and comprehensive visualizations for systematic evaluation of counting biases.

## 🚀 Features

### Interactive Analysis
- **Streamlit Dashboard**: Real-time single-image analysis with multiple VLM comparison
- **Jupyter Notebooks**: Comprehensive experiments with reproducible results
- **Batch Processing**: Systematic evaluation across large image sets

### Synthetic Data Generation
- **Controlled Occlusion**: Generate images with precise occlusion levels (0% to 75%)
- **Multiple Object Types**: Circles, rectangles, triangles with customizable properties
- **Background Variations**: Plain, gradient, textured, and noisy backgrounds
- **Occlusion Patterns**: Random patches, strips, and blocks

### Real-World Evaluation
- **Curated COCO Dataset**: Hand-selected images with verified object counts
- **Camouflage Scenarios**: Challenging cases where objects blend with backgrounds
- **Difficulty Stratification**: Easy, medium, hard, and extreme scenarios

### Model Support
- **GPT-4V (GPT-4 with Vision)**: Latest multimodal model from OpenAI
- **Claude-Vision**: Claude 3 Vision via Anthropic API
- **Gemini-Vision**: Google Gemini Vision via Google AI API
- **Extensible Architecture**: Easy to add new models

### Comprehensive Metrics
- **Accuracy Metrics**: Exact match, within-N accuracy, correlation
- **Bias Analysis**: Over-counting vs under-counting tendencies
- **Confidence Calibration**: How well model confidence correlates with accuracy
- **Statistical Significance**: Rigorous comparison between models

## 🔧 Quick Start

### 1. Environment Setup

#### Option A: GitHub + Replit (Recommended)
1. Fork the [VLM-Counting-Bias repository](https://github.com/FazeelUsmani/VLM-Counting-Bias) on GitHub
2. Import your forked repository into Replit
3. Set up your API keys in Replit Secrets:
   - Go to the Secrets tab in your Repl
   - Add `OPENAI_API_KEY` with your OpenAI API key
   - Add `ANTHROPIC_API_KEY` for Claude Vision analysis
   - Add `GEMINI_API_KEY` for Gemini Vision analysis
4. Create test data: `python create_test_data.py`
5. Run the Streamlit app: The workflow will start automatically

#### Option B: Google Colab (for GPU support)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FazeelUsmani/VLM-Counting-Bias/blob/main/notebooks/01_counting_occlusion_synthetic.ipynb)

1. Click the Colab badge above or open any notebook directly
2. Run the first cell to install dependencies
3. Set your API keys in the notebook's environment

#### Option C: Local Installation
```bash
git clone https://github.com/FazeelUsmani/VLM-Counting-Bias.git
cd VLM-Counting-Bias
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"  # Optional
export GEMINI_API_KEY="your-gemini-key-here"        # Optional
python create_test_data.py
streamlit run app.py --server.address 0.0.0.0 --server.port 5000
```

### 2. API Key Setup

#### Replit Secrets (Secure Method)
API keys are stored securely using Replit's Secrets feature:
- **OPENAI_API_KEY**: Required for GPT-4V analysis (get from https://platform.openai.com)
- **ANTHROPIC_API_KEY**: Required for Claude Vision analysis (get from https://console.anthropic.com)
- **GEMINI_API_KEY**: Required for Gemini Vision analysis (get from https://ai.google.dev)

The application automatically reads these from environment variables, so no hardcoding is needed.

#### Manual Environment Setup
If not using Replit, set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
```

### 3. Running Experiments

#### Interactive Analysis (Streamlit)
- Single image analysis with real-time VLM comparison
- Batch processing of multiple images
- Results dashboard with comprehensive metrics

#### Research Notebooks (Jupyter)
- `01_counting_occlusion_synthetic.ipynb`: Synthetic data experiments
- `02_counting_real_camouflage.ipynb`: Real-world COCO evaluation

## 🧪 Testing

The platform includes comprehensive automated tests to ensure reliability:

```bash
# Run all tests
python -m pytest tests/test_smoke.py -v

# Run specific test categories
python -m pytest tests/test_smoke.py::TestVLMInterface -v
python -m pytest tests/test_smoke.py::TestSyntheticGenerator -v
```

### Test Coverage
- **VLM Interface Tests**: Model initialization, availability checks
- **Synthetic Generator Tests**: Image generation, occlusion application
- **Image Processing Tests**: Preprocessing, base64 conversion, analysis
- **Evaluation Metrics Tests**: Accuracy calculations, bias analysis
- **Integration Tests**: End-to-end pipeline validation

## 🚀 Continuous Integration

GitHub Actions automatically test the platform:
- Multiple Python versions (3.9, 3.10, 3.11)
- Notebook validation and syntax checking
- Streamlit application startup verification
- Automated dependency management

## 📁 Repository Structure

```
├── app.py                              # Streamlit dashboard
├── models/
│   ├── vlm_interface.py               # VLM abstraction layer
│   └── synthetic_generator.py         # Synthetic data generation
├── utils/
│   ├── image_processing.py           # Image utilities
│   └── evaluation_metrics.py         # Performance metrics
├── notebooks/
│   ├── 01_counting_occlusion_synthetic.ipynb
│   ├── 02_counting_real_camouflage.ipynb
│   └── colab_setup.py                # Google Colab setup script
├── data/
│   └── download_scripts.py           # Dataset management
├── tests/
│   └── test_smoke.py                 # Automated test suite
├── .github/workflows/
│   └── ci.yml                        # Continuous integration
├── create_test_data.py                # Sample data generator
└── .streamlit/
    └── config.toml                   # Streamlit configuration
```

## 🔗 Links

- **GitHub Repository**: [https://github.com/FazeelUsmani/VLM-Counting-Bias](https://github.com/FazeelUsmani/VLM-Counting-Bias)
- **Run on Replit**: Fork from GitHub and import to Replit  
- **Open in Colab**: Use the badges above to run notebooks in Google Colab

## 📄 Citation

If you use this platform in your research, please cite:

```bibtex
@misc{vlm-counting-bias-2024,
  title={VLM Counting Bias Research Platform},
  author={Fazeel Usmani},
  year={2024},
  url={https://github.com/FazeelUsmani/VLM-Counting-Bias}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4V API access
- Anthropic for Claude Vision API access
- Google for Gemini Vision API access
- The COCO dataset maintainers
- CAPTURe dataset creators for occlusion counting benchmarks
- Streamlit team for the excellent dashboard framework

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions about this research platform:
- Open an issue on [GitHub](https://github.com/FazeelUsmani/VLM-Counting-Bias/issues)
- Check the documentation for setup instructions
