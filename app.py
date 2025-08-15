import streamlit as st
import os
import json
import base64
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.vlm_interface import VLMManager
from utils.image_processing import preprocess_image
from utils.evaluation_metrics import calculate_accuracy_metrics

# Configure page
st.set_page_config(
    page_title="VLM Counting Bias Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []

def main():
    st.title("🔬 Vision-Language Model Counting Bias Research Platform")
    st.markdown("### Interactive Analysis of Object Counting Under Occlusion and Camouflage")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key configuration
        st.subheader("API Keys")
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Required for GPT-4V analysis"
        )
        
        anthropic_key = st.text_input(
            "Anthropic API Key", 
            type="password", 
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Required for Claude Vision analysis"
        )
        
        gemini_key = st.text_input(
            "Gemini API Key", 
            type="password", 
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Required for Gemini Vision analysis"
        )
        
        hf_token = st.text_input(
            "HuggingFace Token", 
            type="password", 
            value=os.getenv("HF_TOKEN", ""),
            help="Optional: For private model access and LLaVA"
        )
        
        # Model selection
        st.subheader("Model Selection")
        
        # Set environment variables for API keys
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        
        # Get available models dynamically
        try:
            temp_vlm = VLMManager(openai_key=openai_key, hf_token=hf_token)
            available_models = temp_vlm.get_available_models()
        except Exception as e:
            st.error(f"Error initializing VLM manager: {str(e)}")
            # Show all models anyway - they'll show appropriate errors if API keys missing
            available_models = ["GPT-4V", "Claude-Vision", "Gemini-Vision", "BLIP-2", "LLaVA"]
        
        st.info(f"Available models: {', '.join(available_models)}")
        
        # Model selection with better UX
        # Initialize selection state
        if 'selected_models_state' not in st.session_state:
            st.session_state.selected_models_state = available_models[:1] if available_models else []
        
        # Control buttons above dropdown
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", key="select_all_models"):
                st.session_state.selected_models_state = available_models
        with col2:
            if st.button("Clear All", key="clear_all_models"):
                st.session_state.selected_models_state = []
        
        selected_models = st.multiselect(
            "Choose VLMs to evaluate:",
            available_models,
            default=st.session_state.selected_models_state,
            key="model_selector"
        )
        
        # Update the state when selection changes
        if selected_models != st.session_state.selected_models_state:
            st.session_state.selected_models_state = selected_models
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.7, 0.1
        )
        
        max_retries = st.number_input(
            "Max API Retries", 
            min_value=1, max_value=5, value=3
        )
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["Single Image Analysis", "Batch Analysis", "Results Dashboard", "Documentation"])
    
    with tab1:
        single_image_analysis(selected_models, openai_key, hf_token, confidence_threshold, max_retries)
    
    with tab2:
        batch_analysis(selected_models, openai_key, hf_token)
    
    with tab3:
        results_dashboard()
    
    with tab4:
        documentation()

def single_image_analysis(selected_models, openai_key, hf_token, confidence_threshold, max_retries):
    st.header("Single Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to analyze object counting"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Object type selection
            object_type = st.text_input(
                "What objects should be counted?", 
                placeholder="e.g., people, cars, birds",
                help="Specify the type of objects to count in the image"
            )
            
            # Ground truth (optional)
            ground_truth = st.number_input(
                "Ground Truth Count (optional)", 
                min_value=0, 
                value=0,
                help="If you know the actual count, enter it for comparison"
            )
            
            if st.button("Analyze Image", type="primary"):
                if not object_type:
                    st.error("Please specify what objects to count")
                    return
                
                if not selected_models:
                    st.error("Please select at least one model")
                    return
                
                analyze_single_image(image, object_type, selected_models, openai_key, hf_token, ground_truth, confidence_threshold, max_retries)
    
    with col2:
        st.subheader("Analysis Results")
        if st.session_state.get('current_analysis'):
            display_analysis_results(st.session_state.current_analysis)

def analyze_single_image(image, object_type, selected_models, openai_key, hf_token, ground_truth, confidence_threshold, max_retries):
    """Analyze a single image with selected VLMs"""
    
    # Initialize VLM Manager
    try:
        vlm_manager = VLMManager(
            openai_key=openai_key,
            hf_token=hf_token,
            confidence_threshold=confidence_threshold,
            max_retries=max_retries
        )
    except Exception as e:
        st.error(f"Failed to initialize VLM interface: {str(e)}")
        return
    
    results = {}
    
    with st.spinner("Analyzing image with selected models..."):
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(selected_models):
            try:
                st.info(f"Running {model_name}...")
                
                # Convert image to base64 for API calls
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Get prediction from model
                prediction = vlm_manager.count_objects(
                    model_name, img_base64, object_type
                )
                
                results[model_name] = prediction
                
                progress_bar.progress((i + 1) / len(selected_models))
                
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
                results[model_name] = {
                    'count': 0,
                    'confidence': 0.0,
                    'error': str(e)
                }
    
    # Store results
    analysis_result = {
        'object_type': object_type,
        'ground_truth': ground_truth if ground_truth > 0 else None,
        'predictions': results,
        'image_size': image.size,
        'timestamp': pd.Timestamp.now()
    }
    
    st.session_state.current_analysis = analysis_result
    st.session_state.results.append(analysis_result)
    
    st.success("Analysis complete!")

def display_analysis_results(analysis):
    """Display results of image analysis"""
    
    st.subheader("Model Predictions")
    
    # Create results table
    results_data = []
    for model_name, prediction in analysis['predictions'].items():
        if 'error' in prediction:
            results_data.append({
                'Model': model_name,
                'Count': -1,  # Use -1 to indicate error
                'Status': 'Error',
                'Confidence': 'N/A',
                'Error': prediction['error']
            })
        else:
            results_data.append({
                'Model': model_name,
                'Count': prediction['count'],
                'Status': 'Success',
                'Confidence': f"{prediction['confidence']:.2f}",
                'Error': None
            })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Ground truth comparison
    if analysis.get('ground_truth'):
        st.subheader("Accuracy Analysis")
        
        accurate_models = []
        for model_name, prediction in analysis['predictions'].items():
            if 'error' not in prediction:
                error = abs(prediction['count'] - analysis['ground_truth'])
                accuracy = 1 - (error / max(analysis['ground_truth'], 1))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{model_name} Prediction", prediction['count'])
                with col2:
                    st.metric("Absolute Error", error)
                with col3:
                    st.metric("Accuracy", f"{accuracy:.1%}")
                
                if error == 0:
                    accurate_models.append(model_name)
        
        if accurate_models:
            st.success(f"Exact matches: {', '.join(accurate_models)}")
        else:
            st.warning("No model achieved exact match with ground truth")
    
    # Visualization
    if len(analysis['predictions']) > 1:
        st.subheader("Model Comparison")
        
        # Create comparison chart
        model_names = []
        counts = []
        confidences = []
        
        for model_name, prediction in analysis['predictions'].items():
            if 'error' not in prediction:
                model_names.append(model_name)
                counts.append(prediction['count'])
                confidences.append(prediction['confidence'])
        
        if model_names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Count comparison
            bars1 = ax1.bar(model_names, counts)
            ax1.set_title('Predicted Counts by Model')
            ax1.set_ylabel('Count')
            ax1.set_ylim(0, max(counts) * 1.2 if counts else 1)
            
            if analysis.get('ground_truth'):
                ax1.axhline(y=analysis['ground_truth'], color='r', linestyle='--', label='Ground Truth')
                ax1.legend()
            
            # Add value labels on bars
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
            
            # Confidence comparison
            bars2 = ax2.bar(model_names, confidences)
            ax2.set_title('Model Confidence')
            ax2.set_ylabel('Confidence Score')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, conf in zip(bars2, confidences):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{conf:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

def batch_analysis(selected_models, openai_key, hf_token):
    st.header("Batch Analysis")
    st.info("Upload multiple images for systematic evaluation")
    
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images for analysis")
        
        object_type = st.text_input(
            "Object type to count", 
            placeholder="e.g., people, cars, birds"
        )
        
        # Option to upload CSV with ground truth
        ground_truth_file = st.file_uploader(
            "Ground truth CSV (optional)",
            type=['csv'],
            help="CSV with columns: filename, count"
        )
        
        ground_truth_dict = {}
        if ground_truth_file:
            gt_df = pd.read_csv(ground_truth_file)
            ground_truth_dict = dict(zip(gt_df['filename'], gt_df['count']))
            st.success(f"Loaded ground truth for {len(ground_truth_dict)} images")
        
        if st.button("Start Batch Analysis", type="primary"):
            if not object_type:
                st.error("Please specify object type")
                return
            
            run_batch_analysis(uploaded_files, object_type, selected_models, openai_key, hf_token, ground_truth_dict)

def run_batch_analysis(uploaded_files, object_type, selected_models, openai_key, hf_token, ground_truth_dict):
    """Run analysis on batch of images"""
    
    try:
        vlm_manager = VLMManager(
            openai_key=openai_key,
            hf_token=hf_token
        )
    except Exception as e:
        st.error(f"Failed to initialize VLM interface: {str(e)}")
        return
    
    batch_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            image = Image.open(uploaded_file)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            file_results = {
                'filename': uploaded_file.name,
                'object_type': object_type,
                'ground_truth': ground_truth_dict.get(uploaded_file.name),
                'predictions': {},
                'timestamp': pd.Timestamp.now()
            }
            
            for model_name in selected_models:
                try:
                    prediction = vlm_manager.count_objects(
                        model_name, img_base64, object_type
                    )
                    file_results['predictions'][model_name] = prediction
                except Exception as e:
                    file_results['predictions'][model_name] = {
                        'count': 0,
                        'confidence': 0.0,
                        'error': str(e)
                    }
            
            batch_results.append(file_results)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Store batch results
    st.session_state.batch_results = batch_results
    status_text.text("Batch analysis complete!")
    
    # Display summary
    display_batch_summary(batch_results)

def display_batch_summary(batch_results):
    """Display summary of batch analysis results"""
    
    st.subheader("Batch Analysis Summary")
    
    # Create summary statistics
    summary_data = []
    
    for result in batch_results:
        for model_name, prediction in result['predictions'].items():
            if 'error' not in prediction:
                row = {
                    'filename': result['filename'],
                    'model': model_name,
                    'predicted_count': prediction['count'],
                    'confidence': prediction['confidence'],
                    'ground_truth': result.get('ground_truth')
                }
                
                if result.get('ground_truth'):
                    row['absolute_error'] = abs(prediction['count'] - result['ground_truth'])
                    row['accuracy'] = 1 - (row['absolute_error'] / max(result['ground_truth'], 1))
                
                summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Overall statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_confidence = summary_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col2:
            if 'accuracy' in summary_df.columns:
                avg_accuracy = summary_df['accuracy'].mean()
                st.metric("Average Accuracy", f"{avg_accuracy:.3f}")
        
        with col3:
            total_predictions = len(summary_df)
            st.metric("Total Predictions", total_predictions)
        
        # Model comparison
        if len(summary_df['model'].unique()) > 1:
            st.subheader("Model Performance Comparison")
            
            model_stats = summary_df.groupby('model').agg({
                'confidence': 'mean',
                'accuracy': 'mean' if 'accuracy' in summary_df.columns else lambda x: None,
                'absolute_error': 'mean' if 'absolute_error' in summary_df.columns else lambda x: None
            }).round(3)
            
            st.dataframe(model_stats, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Results")
        st.dataframe(summary_df, use_container_width=True)
        
        # Download results
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            csv_data,
            "batch_analysis_results.csv",
            "text/csv"
        )

def results_dashboard():
    st.header("Results Dashboard")
    
    if not st.session_state.results:
        st.info("No analysis results yet. Run some analyses to see results here.")
        return
    
    # Aggregate all results
    all_results = []
    for result in st.session_state.results:
        for model_name, prediction in result['predictions'].items():
            if 'error' not in prediction:
                row = {
                    'timestamp': result['timestamp'],
                    'object_type': result['object_type'],
                    'model': model_name,
                    'predicted_count': prediction['count'],
                    'confidence': prediction['confidence'],
                    'ground_truth': result.get('ground_truth')
                }
                
                if result.get('ground_truth'):
                    row['absolute_error'] = abs(prediction['count'] - result['ground_truth'])
                    row['bias'] = prediction['count'] - result['ground_truth']
                
                all_results.append(row)
    
    if not all_results:
        st.warning("No valid results to display")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Summary metrics
    st.subheader("Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_analyses = len(results_df)
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        avg_confidence = results_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col3:
        if 'absolute_error' in results_df.columns:
            avg_error = results_df['absolute_error'].mean()
            st.metric("Avg Absolute Error", f"{avg_error:.2f}")
    
    with col4:
        unique_models = results_df['model'].nunique()
        st.metric("Models Tested", int(unique_models))
    
    # Visualizations
    if len(results_df) > 1:
        st.subheader("Analysis Visualizations")
        
        # Model performance comparison
        if results_df['model'].nunique() > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Confidence by model
            model_confidence = results_df.groupby('model')['confidence'].mean()
            axes[0, 0].bar(model_confidence.index, model_confidence.values)
            axes[0, 0].set_title('Average Confidence by Model')
            axes[0, 0].set_ylabel('Confidence')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Error distribution (if available)
            if 'absolute_error' in results_df.columns:
                results_df.boxplot(column='absolute_error', by='model', ax=axes[0, 1])
                axes[0, 1].set_title('Error Distribution by Model')
                axes[0, 1].set_ylabel('Absolute Error')
            
            # Bias analysis (if available)
            if 'bias' in results_df.columns:
                model_bias = results_df.groupby('model')['bias'].mean()
                colors = ['red' if x < 0 else 'green' for x in model_bias.values]
                axes[1, 0].bar(model_bias.index, model_bias.values, color=colors)
                axes[1, 0].set_title('Average Bias by Model (Red=Under, Green=Over)')
                axes[1, 0].set_ylabel('Bias (Predicted - Truth)')
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Confidence vs Error correlation (if available)
            if 'absolute_error' in results_df.columns:
                axes[1, 1].scatter(results_df['confidence'], results_df['absolute_error'], alpha=0.6)
                axes[1, 1].set_xlabel('Confidence')
                axes[1, 1].set_ylabel('Absolute Error')
                axes[1, 1].set_title('Confidence vs Error Correlation')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Data table
    st.subheader("All Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Export functionality
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        "Download All Results",
        csv_data,
        "vlm_counting_results.csv",
        "text/csv"
    )

def documentation():
    st.header("Documentation")
    
    st.markdown("""
    ## Vision-Language Model Counting Bias Research Platform
    
    This platform enables systematic evaluation of object counting capabilities in Vision-Language Models (VLMs),
    with particular focus on biases that occur under occlusion and camouflage conditions.
    
    ### Features
    
    #### 🔍 Single Image Analysis
    - Upload individual images for immediate analysis
    - Select multiple VLMs for comparison
    - Specify object types to count
    - Optional ground truth comparison
    - Real-time confidence scoring
    
    #### 📊 Batch Analysis  
    - Process multiple images simultaneously
    - Upload CSV with ground truth counts
    - Statistical summaries and model comparisons
    - Export results for further analysis
    
    #### 📈 Results Dashboard
    - Aggregate view of all analyses
    - Performance metrics and visualizations
    - Bias detection and error analysis
    - Export functionality
    
    ### Supported Models
    
    #### GPT-4V (GPT-4 with Vision)
    - Latest multimodal model from OpenAI
    - Requires OpenAI API key
    - Strong general vision understanding
    - Known limitations in precise counting
    
    #### BLIP-2
    - Open-source VLM from Salesforce
    - Uses HuggingFace Inference API
    - Specialized for image-text tasks
    - Good balance of speed and accuracy
    
    #### LLaVA (Large Language and Vision Assistant)
    - Open-source alternative to GPT-4V
    - Based on LLaMA architecture
    - Fine-tuned for visual instruction following
    - Research-focused capabilities
    
    ### Setup Instructions
    
    #### API Keys
    1. **OpenAI API Key**: Required for GPT-4V
       - Get from: https://platform.openai.com/api-keys
       - Set as environment variable: `OPENAI_API_KEY`
       - Or enter in sidebar configuration
    
    2. **HuggingFace Token**: Optional for private models
       - Get from: https://huggingface.co/settings/tokens
       - Set as environment variable: `HF_TOKEN`
       - Or enter in sidebar configuration
    
    #### Environment Setup
    ```bash
    # Clone repository
    git clone <repository-url>
    cd vlm-counting-bias
    
    # Install dependencies (handled by replit.nix)
    # Or manually: pip install -r requirements.txt
    
    # Set API keys
    export OPENAI_API_KEY="your-key-here"
    export HF_TOKEN="your-token-here"
    
    # Run Streamlit app
    streamlit run app.py
    ```
    
    ### Research Applications
    
    #### Counting Bias Evaluation
    - Systematic testing of counting accuracy
    - Occlusion level impact analysis
    - Camouflage detection capabilities
    - Cross-model performance comparison
    
    #### Data Collection
    - Structured result logging
    - Ground truth comparison
    - Statistical significance testing
    - Export for academic publication
    
    #### Model Development
    - Baseline establishment for new models
    - Prompt engineering optimization
    - Error pattern identification
    - Robustness evaluation
    
    ### Jupyter Notebooks
    
    For more comprehensive experiments, see the included notebooks:
    
    - `01_counting_occlusion_synthetic.ipynb`: Synthetic data with controlled occlusion
    - `02_counting_real_camouflage.ipynb`: Real-world images with camouflage scenarios
    
    ### Troubleshooting
    
    #### API Errors
    - Verify API keys are correctly set
    - Check rate limits and quotas
    - Ensure network connectivity
    - Review model availability
    
    #### Performance Issues  
    - Use batch analysis for multiple images
    - Consider API rate limiting
    - Monitor memory usage with large images
    - Optimize image sizes if needed
    
    #### Result Accuracy
    - Verify ground truth data format
    - Check object type specification
    - Review confidence thresholds
    - Validate image preprocessing
    
    ### Citation
    
    If you use this platform in your research, please cite:
    
    ```bibtex
    @software{vlm_counting_bias_platform,
        title={Vision-Language Model Counting Bias Research Platform},
        author={[Your Name]},
        year={2024},
        url={https://github.com/your-repo/vlm-counting-bias}
    }
    ```
    """)

if __name__ == "__main__":
    main()
