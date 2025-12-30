"""
Error Analysis Module

Functions for analyzing model errors and categorizing failure modes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from collections import Counter


def categorize_errors(
    texts: List[str],
    true_labels: List[int],
    predictions: List[int],
    probabilities: List[float] = None
) -> pd.DataFrame:
    """
    Categorize prediction errors by linguistic features.
    
    Args:
        texts: List of input texts
        true_labels: Ground truth labels
        predictions: Model predictions
        probabilities: Optional prediction probabilities
    
    Returns:
        DataFrame with error analysis
    """
    from src.data_processing import detect_linguistic_features
    
    error_data = []
    
    for i, (text, true_label, pred) in enumerate(zip(texts, true_labels, predictions)):
        # Only analyze errors
        if true_label != pred:
            features = detect_linguistic_features(text)
            
            # Determine primary error type
            if features['has_negation']:
                error_type = 'negation'
            elif features['has_mixed_sentiment']:
                error_type = 'mixed_sentiment'
            else:
                error_type = 'other'
            
            error_data.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'true_label': 'positive' if true_label == 1 else 'negative',
                'predicted': 'positive' if pred == 1 else 'negative',
                'error_type': error_type,
                'confidence': probabilities[i] if probabilities else None,
                **features
            })
    
    return pd.DataFrame(error_data)


def compute_error_statistics(error_df: pd.DataFrame) -> Dict:
    """
    Compute statistics on error distribution.
    
    Args:
        error_df: DataFrame from categorize_errors()
    
    Returns:
        Dictionary of error statistics
    """
    total_errors = len(error_df)
    
    error_counts = error_df['error_type'].value_counts()
    
    stats = {
        'total_errors': total_errors,
        'negation_count': error_counts.get('negation', 0),
        'negation_pct': (error_counts.get('negation', 0) / total_errors) * 100,
        'mixed_sentiment_count': error_counts.get('mixed_sentiment', 0),
        'mixed_sentiment_pct': (error_counts.get('mixed_sentiment', 0) / total_errors) * 100,
        'other_count': error_counts.get('other', 0),
        'other_pct': (error_counts.get('other', 0) / total_errors) * 100
    }
    
    return stats


def analyze_errors(test_data, model=None) -> Dict:
    """
    Main error analysis function.
    
    Args:
        test_data: Test dataset
        model: Optional trained model
    
    Returns:
        Dictionary with analysis results
    """
    # For demonstration, load sample errors
    from src.data_processing import load_sample_errors
    
    sample_errors = load_sample_errors()
    
    # Count error types
    error_counts = sample_errors['error_type'].value_counts()
    
    results = {
        'summary': {
            'total_examples': len(test_data) if test_data else 5,
            'total_errors': len(sample_errors),
            'error_rate': len(sample_errors) / (len(test_data) if test_data else 5) * 100
        },
        'error_distribution': error_counts.to_dict(),
        'sample_errors': sample_errors.to_dict('records')
    }
    
    print("\n=== Error Analysis Results ===")
    print(f"Total errors analyzed: {results['summary']['total_errors']}")
    print("\nError distribution:")
    for error_type, count in results['error_distribution'].items():
        print(f"  {error_type}: {count}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Running error analysis on sample data...")
    
    # Simulate some data
    sample_texts = [
        "This film is not without its flaws but overall excellent",
        "Oh wonderful another masterpiece",
        "Great acting but terrible plot"
    ]
    sample_labels = [1, 0, 0]
    sample_preds = [0, 1, 1]
    
    # Categorize errors
    errors = categorize_errors(sample_texts, sample_labels, sample_preds)
    print("\nError categorization:")
    print(errors[['text', 'true_label', 'predicted', 'error_type']])
    
    # Statistics
    stats = compute_error_statistics(errors)
    print("\nError statistics:")
    for key, value in stats.items():
        if 'pct' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")
