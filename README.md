# Transformer Error Analysis for Sentiment Classification

A systematic investigation of failure modes in transformer-based sentiment classifiers.

## Overview

This project analyzes how DistilBERT models fail on sentiment classification, examining error patterns beyond aggregate accuracy metrics. The goal is to understand which linguistic structures cause systematic misclassification.

## Key Findings

- 27.3% of errors involve negation constructs (e.g., "not bad", "never good")
- 19.4% of errors contain mixed sentiment (positive and negative words together)
- Misclassified examples show 2x higher variance in embedding space
- Domain shift causes 18% accuracy drop (movie reviews to product reviews)

## Dataset

IMDB Movie Reviews
- 50,000 reviews (25k train, 25k test)
- Binary sentiment labels (positive/negative)
- Publicly available via Hugging Face datasets

## Project Structure

transformer-error-analysis/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   └── sample_errors.csv
├── src/
│   ├── data_processing.py
│   └── error_analysis.py
└── notebooks/
    └── error_analysis.ipynb

## Installation

git clone https://github.com/gulusan2021/transformer-error-analysis.git
cd transformer-error-analysis
pip install -r requirements.txt

## Quick Start

from src.data_processing import load_data
from src.error_analysis import analyze_errors

# Load IMDB dataset
train_data, test_data = load_data()

# Analyze errors
results = analyze_errors(test_data)
print(results)

## Results

Error Distribution:
- Negation: 634 errors (27.3%)
- Mixed Sentiment: 451 errors (19.4%)
- Sarcasm: 274 errors (11.8%)
- Domain-Specific: 365 errors (15.7%)

Example Errors:

Negation mishandling:
Review: "This film is not without its flaws, but overall excellent."
True Label: Positive
Predicted: Negative

Sarcasm:
Review: "Oh wonderful, another masterpiece. Just what we needed."
True Label: Negative
Predicted: Positive

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn
- pandas
- matplotlib



## Contact

Gulusan Erdogan-Ozgul
Email: e.gulusan@gmail.com
Website: https://gulusan2021.github.io/gulusan-website/


