# Contributing to VLM Counting Bias Research Platform

Thank you for your interest in contributing to the VLM Counting Bias Research Platform! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine or open in Replit
3. **Set up the environment** following the README instructions
4. **Configure API keys** in Replit Secrets or environment variables

## Development Setup

### Using Replit (Recommended)
1. Fork this Repl
2. Add your API keys to Replit Secrets:
   - `OPENAI_API_KEY` for GPT-4V access
   - `HF_TOKEN` for HuggingFace models
3. Run the application with `streamlit run app.py`

### Local Development
```bash
pip install -r requirements.txt  # Use pyproject.toml dependencies
export OPENAI_API_KEY="your-key"
export HF_TOKEN="your-token"
streamlit run app.py --server.port 5000
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

### Testing
- Run the test suite before submitting: `pytest tests/test_smoke.py -v`
- Add tests for new functionality
- Ensure all tests pass

### Documentation
- Update README.md for significant changes
- Add docstrings to new functions
- Update notebooks with clear explanations

## Types of Contributions

### 1. New VLM Integrations
- Add new vision-language models to `models/vlm_interface.py`
- Follow the existing interface pattern
- Include proper error handling

### 2. Evaluation Metrics
- Add new bias analysis methods to `utils/evaluation_metrics.py`
- Include statistical significance tests
- Provide clear documentation

### 3. Synthetic Data Generation
- Enhance `models/synthetic_generator.py`
- Add new occlusion patterns or object types
- Ensure reproducibility with seed control

### 4. UI Improvements
- Improve the Streamlit dashboard in `app.py`
- Add new visualization options
- Enhance user experience

### 5. Research Notebooks
- Add new experimental notebooks
- Include clear methodology and results
- Ensure Colab compatibility

## Pull Request Process

1. **Create a feature branch** from main
2. **Make your changes** following the guidelines above
3. **Test thoroughly** using the test suite
4. **Update documentation** as needed
5. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots for UI changes

## Research Ethics

This platform is designed for academic research on VLM limitations. Please ensure:
- Proper attribution of datasets and models
- Responsible use of API resources
- Consideration of potential biases in research design

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions about the research methodology
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.