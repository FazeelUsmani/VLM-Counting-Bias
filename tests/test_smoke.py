"""
Smoke Tests for VLM Counting Bias Research Platform

These tests verify that the basic functionality of the platform works
without errors. They are designed to run quickly and catch major issues.
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
from models.vlm_interface import VLMManager, create_vlm_interface
from models.synthetic_generator import SyntheticDataGenerator, create_default_dataset
from utils.image_processing import (
    preprocess_image, image_to_base64, base64_to_image, 
    analyze_image_properties, create_image_grid
)
from utils.evaluation_metrics import (
    calculate_accuracy_metrics, calculate_bias_metrics,
    generate_evaluation_report
)
from data.download_scripts import DatasetManager


class TestVLMInterface:
    """Test VLM interface functionality."""
    
    def test_vlm_manager_initialization(self):
        """Test that VLM manager can be initialized."""
        # Should work without API keys (models may not be available but interface should initialize)
        vlm_manager = VLMManager()
        assert isinstance(vlm_manager, VLMManager)
        
        # Should have some basic methods
        assert hasattr(vlm_manager, 'get_available_models')
        assert hasattr(vlm_manager, 'count_objects')
    
    def test_get_available_models(self):
        """Test getting available models."""
        vlm_manager = VLMManager()
        models = vlm_manager.get_available_models()
        assert isinstance(models, list)
        # May be empty if no API keys provided, but should be a list
    
    def test_model_info(self):
        """Test getting model information."""
        vlm_manager = VLMManager()
        info = vlm_manager.get_model_info()
        assert isinstance(info, dict)
    
    def test_create_vlm_interface(self):
        """Test convenience function for creating VLM interface."""
        vlm = create_vlm_interface()
        assert isinstance(vlm, VLMManager)


class TestSyntheticGenerator:
    """Test synthetic data generation."""
    
    def test_generator_initialization(self):
        """Test that generator can be initialized."""
        generator = SyntheticDataGenerator()
        assert generator.image_size == (512, 512)
        assert hasattr(generator, 'color_map')
        assert hasattr(generator, 'default_configs')
    
    def test_background_generation(self):
        """Test background generation."""
        generator = SyntheticDataGenerator(image_size=(100, 100))
        
        # Test plain background
        bg = generator.generate_background('plain', 'white')
        assert isinstance(bg, Image.Image)
        assert bg.size == (100, 100)
        
        # Test gradient background
        bg_grad = generator.generate_background('gradient', 'white')
        assert isinstance(bg_grad, Image.Image)
        assert bg_grad.size == (100, 100)
    
    def test_object_generation(self):
        """Test object generation on image."""
        generator = SyntheticDataGenerator(image_size=(200, 200))
        background = generator.generate_background('plain', 'white')
        
        config = generator.default_configs['circles']
        image_with_objects, objects = generator.generate_objects(background, config, 3)
        
        assert isinstance(image_with_objects, Image.Image)
        assert len(objects) == 3
        assert all('shape' in obj for obj in objects)
        assert all('color' in obj for obj in objects)
    
    def test_occlusion_application(self):
        """Test applying occlusion to images."""
        generator = SyntheticDataGenerator(image_size=(200, 200))
        
        # Create simple image with objects
        background = generator.generate_background('plain', 'white')
        config = generator.default_configs['circles']
        image_with_objects, objects = generator.generate_objects(background, config, 2)
        
        # Apply occlusion
        occluded_image, updated_objects = generator.apply_occlusion(
            image_with_objects, objects, 0.3, 'random_patches'
        )
        
        assert isinstance(occluded_image, Image.Image)
        assert len(updated_objects) == len(objects)
    
    def test_dataset_generation(self):
        """Test generating a small dataset."""
        generator = SyntheticDataGenerator(image_size=(100, 100), seed=42)
        
        # Generate very small dataset for testing
        dataset = generator.generate_dataset(
            object_configs=[generator.default_configs['circles']],
            object_counts=[2, 3],
            occlusion_levels=[0.0, 0.5],
            num_images_per_condition=1,
            background_types=['plain'],
            occlusion_types=['random_patches']
        )
        
        # Should generate 2 * 2 * 1 * 1 * 1 = 4 images
        assert len(dataset) == 4
        assert all(hasattr(img, 'image') for img in dataset)
        assert all(hasattr(img, 'metadata') for img in dataset)
        assert all(hasattr(img, 'ground_truth') for img in dataset)
    
    def test_dataset_saving(self):
        """Test saving dataset to disk."""
        generator = SyntheticDataGenerator(image_size=(50, 50), seed=42)
        
        # Generate tiny dataset
        dataset = generator.generate_dataset(
            object_configs=[generator.default_configs['circles']],
            object_counts=[2],
            occlusion_levels=[0.0],
            num_images_per_condition=1
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator.save_dataset(dataset, temp_dir)
            
            # Check that files were created
            assert (Path(temp_dir) / "images").exists()
            assert (Path(temp_dir) / "metadata.json").exists()
            assert (Path(temp_dir) / "summary.json").exists()
            
            # Check that metadata is valid JSON
            with open(Path(temp_dir) / "metadata.json") as f:
                metadata = json.load(f)
                assert isinstance(metadata, list)
                assert len(metadata) == len(dataset)


class TestImageProcessing:
    """Test image processing utilities."""
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test image
        test_image = Image.new('RGB', (300, 200), color='red')
        
        # Test basic preprocessing
        processed = preprocess_image(test_image)
        assert isinstance(processed, Image.Image)
        assert processed.mode == 'RGB'
        
        # Test with target size
        processed_resized = preprocess_image(test_image, target_size=(400, 400))
        assert processed_resized.size == (400, 400)
    
    def test_base64_conversion(self):
        """Test base64 conversion."""
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        # Convert to base64
        base64_str = image_to_base64(test_image)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        
        # Convert back
        restored_image = base64_to_image(base64_str)
        assert isinstance(restored_image, Image.Image)
        assert restored_image.size == test_image.size
    
    def test_image_analysis(self):
        """Test image property analysis."""
        test_image = Image.new('RGB', (150, 100), color='green')
        
        properties = analyze_image_properties(test_image)
        assert isinstance(properties, dict)
        assert 'dimensions' in properties
        assert 'color_properties' in properties
        assert 'quality_metrics' in properties
        
        # Check specific values
        assert properties['dimensions']['width'] == 150
        assert properties['dimensions']['height'] == 100
    
    def test_image_grid_creation(self):
        """Test creating image grids."""
        # Create test images
        images = [
            Image.new('RGB', (50, 50), color='red'),
            Image.new('RGB', (50, 50), color='green'),
            Image.new('RGB', (50, 50), color='blue'),
            Image.new('RGB', (50, 50), color='yellow')
        ]
        
        grid = create_image_grid(images, grid_size=(2, 2), image_size=(60, 60))
        assert isinstance(grid, Image.Image)
        # Grid should be larger than individual images
        assert grid.size[0] > 60
        assert grid.size[1] > 60


class TestEvaluationMetrics:
    """Test evaluation metrics calculations."""
    
    def test_accuracy_metrics(self):
        """Test accuracy metric calculations."""
        predictions = [3, 4, 2, 7, 2]
        ground_truth = [3, 5, 2, 8, 1]
        
        metrics = calculate_accuracy_metrics(predictions, ground_truth)
        assert isinstance(metrics, dict)
        assert 'exact_match_accuracy' in metrics
        assert 'mean_absolute_error' in metrics
        assert 'correlation' in metrics
        
        # Check reasonable values
        assert 0 <= metrics['exact_match_accuracy'] <= 1
        assert metrics['mean_absolute_error'] >= 0
    
    def test_bias_metrics(self):
        """Test bias metric calculations."""
        predictions = [5, 6, 4, 9, 3]  # Generally over-counting
        ground_truth = [3, 5, 2, 8, 1]
        
        metrics = calculate_bias_metrics(predictions, ground_truth)
        assert isinstance(metrics, dict)
        assert 'mean_bias' in metrics
        assert 'over_counting_rate' in metrics
        assert 'under_counting_rate' in metrics
        
        # Should detect over-counting bias
        assert metrics['mean_bias'] > 0
        assert metrics['over_counting_rate'] > 0
    
    def test_evaluation_report_generation(self):
        """Test comprehensive evaluation report generation."""
        # Create sample results data
        results_data = [
            {
                'predicted_count': 3,
                'true_count': 3,
                'confidence': 0.9,
                'model': 'test_model',
                'difficulty': 'easy'
            },
            {
                'predicted_count': 4,
                'true_count': 5,
                'confidence': 0.7,
                'model': 'test_model',
                'difficulty': 'hard'
            },
            {
                'predicted_count': 2,
                'true_count': 2,
                'confidence': 0.8,
                'model': 'test_model',
                'difficulty': 'medium'
            }
        ]
        
        report = generate_evaluation_report(results_data)
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'confidence_analysis' in report
        
        # Check summary contains expected metrics
        summary = report['summary']
        assert 'total_samples' in summary
        assert 'exact_match_accuracy' in summary
        assert summary['total_samples'] == 3


class TestDataDownloadScripts:
    """Test data download script functionality."""
    
    def test_dataset_manager_initialization(self):
        """Test dataset manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(temp_dir)
            assert isinstance(manager, DatasetManager)
            assert hasattr(manager, 'coco_downloader')
            assert hasattr(manager, 'camouflage_downloader')
    
    def test_dataset_summary_generation(self):
        """Test dataset summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(temp_dir)
            
            # This will generate empty summary since no data is downloaded
            # but should not crash
            manager.generate_dataset_summary()
            
            # Check that directory structure exists
            assert Path(temp_dir).exists()


class TestIntegration:
    """Integration tests that combine multiple components."""
    
    def test_synthetic_to_evaluation_pipeline(self):
        """Test complete pipeline from synthetic generation to evaluation."""
        # Generate small synthetic dataset
        generator = SyntheticDataGenerator(image_size=(100, 100), seed=42)
        dataset = generator.generate_dataset(
            object_configs=[generator.default_configs['circles']],
            object_counts=[2, 3],
            occlusion_levels=[0.0],
            num_images_per_condition=1
        )
        
        # Simulate model predictions (mock data for testing)
        results_data = []
        for synthetic_img in dataset:
            # Simulate prediction (add some noise to true count)
            true_count = synthetic_img.ground_truth['visible_count']
            predicted_count = max(0, true_count + np.random.randint(-1, 2))
            
            results_data.append({
                'predicted_count': predicted_count,
                'true_count': true_count,
                'confidence': np.random.uniform(0.5, 1.0),
                'model': 'mock_model',
                'difficulty': 'medium'
            })
        
        # Generate evaluation report
        report = generate_evaluation_report(results_data)
        assert isinstance(report, dict)
        assert len(results_data) == len(dataset)
    
    def test_image_processing_pipeline(self):
        """Test image processing pipeline."""
        # Create test image
        test_image = Image.new('RGB', (200, 150), color='purple')
        
        # Process image
        processed = preprocess_image(test_image, target_size=(256, 256))
        
        # Convert to base64 (as would be done for API calls)
        base64_str = image_to_base64(processed)
        
        # Analyze properties
        properties = analyze_image_properties(processed)
        
        # All steps should complete without errors
        assert isinstance(processed, Image.Image)
        assert isinstance(base64_str, str)
        assert isinstance(properties, dict)


# Pytest configuration and runner
def run_smoke_tests():
    """Run all smoke tests."""
    pytest.main([
        __file__,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '-x',  # Stop on first failure
    ])


if __name__ == "__main__":
    print("Running VLM Counting Bias Platform Smoke Tests...")
    print("=" * 60)
    
    # Run the tests
    run_smoke_tests()
    
    print("\n" + "=" * 60)
    print("Smoke tests completed!")
    print("\nNote: Some tests may show warnings if API keys are not configured.")
    print("This is expected behavior for smoke tests.")
