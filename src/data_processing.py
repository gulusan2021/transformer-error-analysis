"""
Data Processing Module

Functions for loading and preprocessing the IMDB sentiment dataset.
"""

from datasets import load_dataset
from typing import Tuple, List
import pandas as pd


def load_data(split: str = None) -> dict:
    """
    Load IMDB movie review dataset.
    
    Args:
        split: Optional split to load ('train' or 'test')
    
    Returns:
        Dataset dictionary or specific split
    """
    print("Loading IMDB dataset from Hugging Face...")
    
    if split:
        dataset = load_dataset("imdb", split=split)
        return dataset
    else:
        dataset = load_dataset("imdb")
        return {
            'train': dataset['train'],
            'test': dataset['test']
        }


def extract_texts_and_labels(dataset) -> Tuple[List[str], List[int]]:
    """
    Extract texts and labels from dataset.
    
    Args:
        dataset: Hugging Face dataset object
    
    Returns:
        Tuple of (texts, labels)
    """
    texts = [example['text'] for example in dataset]
    labels = [example['label'] for example in dataset]
    return texts, labels


def detect_linguistic_features(text: str) -> dict:
    """
    Detect linguistic features in text.
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary of detected features
    """
    text_lower = text.lower()
    
    # Negation words
    negation_words = ['not', 'never', 'no', "n't", 'neither', 'nor']
    has_negation = any(word in text_lower for word in negation_words)
    
    # Positive and negative sentiment words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']
    
    has_positive = any(word in text_lower for word in positive_words)
    has_negative = any(word in text_lower for word in negative_words)
    has_mixed_sentiment = has_positive and has_negative
    
    return {
        'has_negation': has_negation,
        'has_positive': has_positive,
        'has_negative': has_negative,
        'has_mixed_sentiment': has_mixed_sentiment,
        'word_count': len(text.split())
    }


def load_sample_errors() -> pd.DataFrame:
    """
    Load sample error cases from CSV.
    
    Returns:
        DataFrame with error examples
    """
    return pd.read_csv('data/sample_errors.csv')


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    data = load_data()
    
    print(f"Train size: {len(data['train'])}")
    print(f"Test size: {len(data['test'])}")
    
    # Show example
    example = data['train'][0]
    print(f"\nExample review: {example['text'][:100]}...")
    print(f"Label: {'Positive' if example['label'] == 1 else 'Negative'}")
    
    # Detect features
    features = detect_linguistic_features(example['text'])
    print(f"\nDetected features: {features}")
