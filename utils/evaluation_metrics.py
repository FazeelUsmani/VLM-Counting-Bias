"""
Evaluation Metrics for VLM Counting Performance

This module provides comprehensive metrics for evaluating the performance
of Vision-Language Models on object counting tasks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import math


def calculate_accuracy_metrics(predictions: List[int], 
                             ground_truth: List[int]) -> Dict[str, float]:
    """Calculate comprehensive accuracy metrics for counting predictions.
    
    Args:
        predictions: List of predicted counts
        ground_truth: List of true counts
        
    Returns:
        Dictionary with various accuracy metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Basic accuracy metrics
    exact_matches = np.sum(predictions == ground_truth)
    total_samples = len(predictions)
    exact_match_accuracy = exact_matches / total_samples
    
    # Error metrics
    absolute_errors = np.abs(predictions - ground_truth)
    relative_errors = np.where(ground_truth != 0, 
                              absolute_errors / ground_truth, 
                              np.where(predictions == 0, 0, 1))
    
    mean_absolute_error = np.mean(absolute_errors)
    median_absolute_error = np.median(absolute_errors)
    mean_relative_error = np.mean(relative_errors)
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.where(ground_truth != 0,
                           np.abs((ground_truth - predictions) / ground_truth) * 100,
                           0))
    
    # Within-N accuracy (percentage of predictions within N of true value)
    within_1_accuracy = np.mean(absolute_errors <= 1)
    within_2_accuracy = np.mean(absolute_errors <= 2)
    within_3_accuracy = np.mean(absolute_errors <= 3)
    
    # Correlation
    correlation = np.corrcoef(predictions, ground_truth)[0, 1] if len(predictions) > 1 else 0
    
    return {
        'exact_match_accuracy': float(exact_match_accuracy),
        'mean_absolute_error': float(mean_absolute_error),
        'median_absolute_error': float(median_absolute_error),
        'mean_relative_error': float(mean_relative_error),
        'rmse': float(rmse),
        'mape': float(mape),
        'within_1_accuracy': float(within_1_accuracy),
        'within_2_accuracy': float(within_2_accuracy),
        'within_3_accuracy': float(within_3_accuracy),
        'correlation': float(correlation),
        'total_samples': int(total_samples)
    }


def calculate_bias_metrics(predictions: List[int], 
                          ground_truth: List[int]) -> Dict[str, float]:
    """Calculate bias metrics to identify systematic over/under-counting.
    
    Args:
        predictions: List of predicted counts
        ground_truth: List of true counts
        
    Returns:
        Dictionary with bias metrics
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Bias calculation
    bias = predictions - ground_truth
    mean_bias = np.mean(bias)
    median_bias = np.median(bias)
    bias_std = np.std(bias)
    
    # Directional bias analysis
    over_counting = np.sum(bias > 0)
    under_counting = np.sum(bias < 0)
    exact_counting = np.sum(bias == 0)
    total_samples = len(bias)
    
    over_counting_rate = over_counting / total_samples
    under_counting_rate = under_counting / total_samples
    exact_counting_rate = exact_counting / total_samples
    
    # Magnitude of bias
    mean_positive_bias = np.mean(bias[bias > 0]) if over_counting > 0 else 0
    mean_negative_bias = np.mean(bias[bias < 0]) if under_counting > 0 else 0
    
    # Bias consistency (how consistently biased)
    bias_consistency = 1 - (bias_std / (np.abs(mean_bias) + 1e-8))
    
    return {
        'mean_bias': float(mean_bias),
        'median_bias': float(median_bias),
        'bias_std': float(bias_std),
        'over_counting_rate': float(over_counting_rate),
        'under_counting_rate': float(under_counting_rate),
        'exact_counting_rate': float(exact_counting_rate),
        'mean_positive_bias': float(mean_positive_bias),
        'mean_negative_bias': float(mean_negative_bias),
        'bias_consistency': float(bias_consistency)
    }


def calculate_confidence_metrics(predictions: List[int],
                               ground_truth: List[int],
                               confidences: List[float]) -> Dict[str, float]:
    """Calculate metrics related to model confidence calibration.
    
    Args:
        predictions: List of predicted counts
        ground_truth: List of true counts
        confidences: List of confidence scores (0-1)
        
    Returns:
        Dictionary with confidence metrics
    """
    if len(predictions) != len(confidences):
        raise ValueError("Predictions and confidences must have same length")
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    confidences = np.array(confidences)
    
    # Basic confidence statistics
    mean_confidence = np.mean(confidences)
    median_confidence = np.median(confidences)
    confidence_std = np.std(confidences)
    
    # Accuracy vs confidence correlation
    accuracies = (predictions == ground_truth).astype(float)
    confidence_accuracy_correlation = np.corrcoef(confidences, accuracies)[0, 1]
    
    # Confidence calibration (ECE - Expected Calibration Error)
    ece = expected_calibration_error(predictions, ground_truth, confidences)
    
    # High confidence accuracy
    high_conf_threshold = 0.8
    high_conf_mask = confidences >= high_conf_threshold
    if np.sum(high_conf_mask) > 0:
        high_conf_accuracy = np.mean(accuracies[high_conf_mask])
        high_conf_samples = np.sum(high_conf_mask)
    else:
        high_conf_accuracy = 0.0
        high_conf_samples = 0
    
    # Low confidence accuracy
    low_conf_threshold = 0.5
    low_conf_mask = confidences <= low_conf_threshold
    if np.sum(low_conf_mask) > 0:
        low_conf_accuracy = np.mean(accuracies[low_conf_mask])
        low_conf_samples = np.sum(low_conf_mask)
    else:
        low_conf_accuracy = 0.0
        low_conf_samples = 0
    
    return {
        'mean_confidence': float(mean_confidence),
        'median_confidence': float(median_confidence),
        'confidence_std': float(confidence_std),
        'confidence_accuracy_correlation': float(confidence_accuracy_correlation),
        'expected_calibration_error': float(ece),
        'high_confidence_accuracy': float(high_conf_accuracy),
        'low_confidence_accuracy': float(low_conf_accuracy),
        'high_confidence_samples': int(high_conf_samples),
        'low_confidence_samples': int(low_conf_samples)
    }


def expected_calibration_error(predictions: np.ndarray,
                             ground_truth: np.ndarray,
                             confidences: np.ndarray,
                             n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).
    
    Args:
        predictions: Predicted counts
        ground_truth: True counts
        confidences: Confidence scores
        n_bins: Number of bins for calibration
        
    Returns:
        Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (predictions == ground_truth).astype(float)
    ece = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_difficulty_stratified_metrics(predictions: List[int],
                                          ground_truth: List[int],
                                          difficulty_labels: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate metrics stratified by difficulty level.
    
    Args:
        predictions: List of predicted counts
        ground_truth: List of true counts
        difficulty_labels: List of difficulty labels (e.g., 'easy', 'medium', 'hard')
        
    Returns:
        Dictionary with metrics for each difficulty level
    """
    if len(predictions) != len(difficulty_labels):
        raise ValueError("Predictions and difficulty labels must have same length")
    
    results = {}
    unique_difficulties = sorted(set(difficulty_labels))
    
    for difficulty in unique_difficulties:
        # Filter data for this difficulty level
        mask = np.array(difficulty_labels) == difficulty
        diff_predictions = np.array(predictions)[mask]
        diff_ground_truth = np.array(ground_truth)[mask]
        
        if len(diff_predictions) > 0:
            # Calculate accuracy metrics for this difficulty
            accuracy_metrics = calculate_accuracy_metrics(
                diff_predictions.tolist(), diff_ground_truth.tolist()
            )
            
            # Calculate bias metrics for this difficulty
            bias_metrics = calculate_bias_metrics(
                diff_predictions.tolist(), diff_ground_truth.tolist()
            )
            
            # Combine metrics
            combined_metrics = {**accuracy_metrics, **bias_metrics}
            results[difficulty] = combined_metrics
        else:
            results[difficulty] = {}
    
    return results


def calculate_count_stratified_metrics(predictions: List[int],
                                     ground_truth: List[int],
                                     count_bins: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Dict[str, float]]:
    """Calculate metrics stratified by true object count ranges.
    
    Args:
        predictions: List of predicted counts
        ground_truth: List of true counts
        count_bins: List of (min, max) tuples for count ranges
        
    Returns:
        Dictionary with metrics for each count range
    """
    if count_bins is None:
        # Default count bins
        count_bins = [(1, 3), (4, 6), (7, 10), (11, float('inf'))]
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    results = {}
    
    for min_count, max_count in count_bins:
        # Create mask for this count range
        if max_count == float('inf'):
            mask = ground_truth >= min_count
            bin_name = f"{min_count}+"
        else:
            mask = (ground_truth >= min_count) & (ground_truth <= max_count)
            bin_name = f"{min_count}-{max_count}"
        
        # Filter data for this count range
        bin_predictions = predictions[mask]
        bin_ground_truth = ground_truth[mask]
        
        if len(bin_predictions) > 0:
            # Calculate metrics for this count range
            accuracy_metrics = calculate_accuracy_metrics(
                bin_predictions.tolist(), bin_ground_truth.tolist()
            )
            
            bias_metrics = calculate_bias_metrics(
                bin_predictions.tolist(), bin_ground_truth.tolist()
            )
            
            # Add count-specific metrics
            combined_metrics = {**accuracy_metrics, **bias_metrics}
            combined_metrics['sample_count'] = len(bin_predictions)
            combined_metrics['mean_true_count'] = float(np.mean(bin_ground_truth))
            
            results[bin_name] = combined_metrics
        else:
            results[bin_name] = {'sample_count': 0}
    
    return results


def calculate_statistical_significance(predictions1: List[int],
                                     ground_truth1: List[int],
                                     predictions2: List[int],
                                     ground_truth2: List[int],
                                     metric: str = 'accuracy') -> Dict[str, Any]:
    """Calculate statistical significance between two model performances.
    
    Args:
        predictions1: Predictions from model 1
        ground_truth1: Ground truth for model 1
        predictions2: Predictions from model 2
        ground_truth2: Ground truth for model 2
        metric: Metric to compare ('accuracy', 'mae', 'bias')
        
    Returns:
        Dictionary with statistical test results
    """
    # Calculate metrics for both models
    if metric == 'accuracy':
        scores1 = (np.array(predictions1) == np.array(ground_truth1)).astype(float)
        scores2 = (np.array(predictions2) == np.array(ground_truth2)).astype(float)
    elif metric == 'mae':
        scores1 = np.abs(np.array(predictions1) - np.array(ground_truth1))
        scores2 = np.abs(np.array(predictions2) - np.array(ground_truth2))
    elif metric == 'bias':
        scores1 = np.array(predictions1) - np.array(ground_truth1)
        scores2 = np.array(predictions2) - np.array(ground_truth2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Perform statistical tests
    # T-test (assumes normal distribution)
    t_stat, t_pvalue = stats.ttest_ind(scores1, scores2)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                         (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                        (len(scores1) + len(scores2) - 2))
    
    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
    
    # Bootstrap confidence interval for difference in means
    n_bootstrap = 1000
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(scores1, size=len(scores1), replace=True)
        sample2 = np.random.choice(scores2, size=len(scores2), replace=True)
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return {
        'metric': metric,
        'model1_mean': float(np.mean(scores1)),
        'model2_mean': float(np.mean(scores2)),
        'difference': float(np.mean(scores1) - np.mean(scores2)),
        't_test': {
            'statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'significant': bool(t_pvalue < 0.05)
        },
        'mannwhitney_test': {
            'statistic': float(u_stat),
            'p_value': float(u_pvalue),
            'significant': bool(u_pvalue < 0.05)
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'magnitude': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        },
        'confidence_interval_95': {
            'lower': float(ci_lower),
            'upper': float(ci_upper)
        }
    }


def generate_evaluation_report(results_data: List[Dict[str, Any]],
                             include_confidence: bool = True) -> Dict[str, Any]:
    """Generate comprehensive evaluation report from results data.
    
    Args:
        results_data: List of result dictionaries with predictions and ground truth
        include_confidence: Whether to include confidence metrics
        
    Returns:
        Comprehensive evaluation report
    """
    # Extract data
    predictions = [r.get('predicted_count', 0) for r in results_data]
    ground_truth = [r.get('true_count', 0) for r in results_data]
    
    if include_confidence:
        confidences = [r.get('confidence', 0.5) for r in results_data]
    
    models = [r.get('model', 'unknown') for r in results_data]
    difficulties = [r.get('difficulty', 'unknown') for r in results_data if 'difficulty' in r]
    
    # Basic metrics
    accuracy_metrics = calculate_accuracy_metrics(predictions, ground_truth)
    bias_metrics = calculate_bias_metrics(predictions, ground_truth)
    
    report = {
        'summary': {
            'total_samples': len(results_data),
            'unique_models': len(set(models)),
            'models_tested': list(set(models)),
            **accuracy_metrics,
            **bias_metrics
        }
    }
    
    # Confidence metrics if available
    if include_confidence and confidences:
        confidence_metrics = calculate_confidence_metrics(predictions, ground_truth, confidences)
        report['confidence_analysis'] = confidence_metrics
    
    # Model comparison if multiple models
    if len(set(models)) > 1:
        model_results = {}
        for model in set(models):
            model_mask = np.array(models) == model
            model_predictions = np.array(predictions)[model_mask]
            model_ground_truth = np.array(ground_truth)[model_mask]
            
            if len(model_predictions) > 0:
                model_accuracy = calculate_accuracy_metrics(
                    model_predictions.tolist(), model_ground_truth.tolist()
                )
                model_bias = calculate_bias_metrics(
                    model_predictions.tolist(), model_ground_truth.tolist()
                )
                
                model_results[model] = {**model_accuracy, **model_bias}
        
        report['model_comparison'] = model_results
    
    # Difficulty stratification if available
    if difficulties and len(difficulties) == len(predictions):
        difficulty_metrics = calculate_difficulty_stratified_metrics(
            predictions, ground_truth, difficulties
        )
        report['difficulty_analysis'] = difficulty_metrics
    
    # Count stratification
    count_metrics = calculate_count_stratified_metrics(predictions, ground_truth)
    report['count_analysis'] = count_metrics
    
    return report


def export_metrics_to_dataframe(evaluation_report: Dict[str, Any]) -> pd.DataFrame:
    """Export evaluation metrics to a pandas DataFrame for analysis.
    
    Args:
        evaluation_report: Report from generate_evaluation_report
        
    Returns:
        DataFrame with flattened metrics
    """
    rows = []
    
    # Summary metrics
    summary = evaluation_report.get('summary', {})
    for metric, value in summary.items():
        rows.append({
            'category': 'overall',
            'subcategory': 'summary',
            'metric': metric,
            'value': value
        })
    
    # Model comparison
    model_comp = evaluation_report.get('model_comparison', {})
    for model, metrics in model_comp.items():
        for metric, value in metrics.items():
            rows.append({
                'category': 'model',
                'subcategory': model,
                'metric': metric,
                'value': value
            })
    
    # Difficulty analysis
    diff_analysis = evaluation_report.get('difficulty_analysis', {})
    for difficulty, metrics in diff_analysis.items():
        for metric, value in metrics.items():
            rows.append({
                'category': 'difficulty',
                'subcategory': difficulty,
                'metric': metric,
                'value': value
            })
    
    # Count analysis
    count_analysis = evaluation_report.get('count_analysis', {})
    for count_range, metrics in count_analysis.items():
        for metric, value in metrics.items():
            rows.append({
                'category': 'count_range',
                'subcategory': count_range,
                'metric': metric,
                'value': value
            })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing evaluation metrics...")
    
    # Generate sample data
    np.random.seed(42)
    ground_truth = [3, 5, 2, 8, 1, 7, 4, 6, 9, 3]
    predictions = [3, 4, 2, 7, 2, 6, 4, 6, 10, 3]  # Some errors
    confidences = [0.9, 0.7, 0.8, 0.6, 0.5, 0.8, 0.9, 0.95, 0.7, 0.85]
    
    # Test accuracy metrics
    accuracy = calculate_accuracy_metrics(predictions, ground_truth)
    print(f"Accuracy metrics: {accuracy}")
    
    # Test bias metrics
    bias = calculate_bias_metrics(predictions, ground_truth)
    print(f"Bias metrics: {bias}")
    
    # Test confidence metrics
    confidence = calculate_confidence_metrics(predictions, ground_truth, confidences)
    print(f"Confidence metrics: {confidence}")
    
    print("Evaluation metrics test completed!")
