# Assignment 1: SMS Spam Classification

**Course**: Applied Machine Learning
**Institute**: Chennai Mathematical Institute
**Student**: Jugaad Singh Sohal (MDS202421)

## Problem Statement

Build a spam filter for SMS messages using machine learning. The goal is to classify messages as either "ham" (legitimate) or "spam".

## Dataset

**UCI SMS Spam Collection**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Total samples: 5,572 messages
- Class distribution: ~87% ham, ~13% spam (imbalanced)

## Approach

### Data Preparation (`prepare.ipynb`)
1. Download and load the dataset
2. Exploratory Data Analysis (EDA)
   - Class distribution analysis
   - Message length distribution
   - Word frequency analysis
3. Train/Validation/Test split (70/15/15, stratified)

### Model Training (`train.ipynb`)
1. **Feature Extraction**: TF-IDF vectorization (5000 features)
2. **Baseline Model**: Rule-based keyword matching
3. **Benchmark Models** (with hyperparameter tuning):
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear SVM
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1, AUPRC, AUROC

## Results

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1 | AUPRC | FP | FN |
|-------|----------|-----------|--------|------|-------|----|----|
| Logistic Regression | 98.21% | 94.50% | 91.96% | 0.9321 | 0.9638 | 6 | 9 |
| **Naive Bayes** | **98.33%** | **96.23%** | 91.07% | **0.9358** | **0.9708** | **4** | 10 |
| SVM | 98.21% | 94.50% | 91.96% | 0.9321 | 0.9673 | 6 | 9 |

### Best Model: Naive Bayes

Selected based on:
- Highest F1 score (0.9358)
- Highest AUPRC (0.9708)
- Lowest false positives (4)
- Highest precision (96.23%)

### Cost-Weighted Analysis

In spam filtering, false positives (marking legitimate email as spam) are more costly than false negatives. Using a 10:1 cost ratio (FP 10x worse than FN):

| Model | Weighted Cost (10×FP + FN) |
|-------|---------------------------|
| Logistic Regression | 69 |
| **Naive Bayes** | **50** |
| SVM | 69 |

## Conclusion

This assignment was about building an SMS spam filter. The problem has two key characteristics: **class imbalance** (~13% spam vs ~87% ham) and **asymmetric misclassification costs** (false positives are 10x more costly than false negatives, since losing a legitimate message is worse than seeing spam).

We need a model that effectively filters spam while ensuring legitimate messages are not incorrectly blocked. **Naive Bayes** is the best model for this task because:
1. It achieves the highest precision (96.23%), meaning when it flags a message as spam, it's almost always correct
2. It has the fewest false positives (4 out of 724 ham messages), minimizing the risk of losing legitimate messages
3. It has the lowest weighted cost (50) when accounting for the 10x penalty on false positives
4. It achieves the highest AUPRC (0.9708), which is the appropriate metric for imbalanced datasets

While SVM and Logistic Regression have slightly higher recall, the trade-off of more false positives makes them less suitable for a production spam filter where user trust is critical.

## Files

```
Assignment-01/
├── README.md           # This file
├── prepare.ipynb       # Data preparation and EDA
├── train.ipynb         # Model training and evaluation
└── data/
    ├── SMSSpamCollection   # Raw dataset
    ├── train.csv           # Training set (70%)
    ├── validation.csv      # Validation set (15%)
    └── test.csv            # Test set (15%)
```

## How to Run

1. Run `prepare.ipynb` to download data and create train/val/test splits
2. Run `train.ipynb` to train models and see results

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
