# SMS Spam Classification using TF-IDF and Machine Learning

## Project Overview

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and Machine Learning.

The project demonstrates a complete text classification workflow including:

- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression
- Naive Bayes
- Pipeline implementation
- Hyperparameter tuning using GridSearchCV
- Model evaluation and comparison

---

# Technologies Used

- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorizer
- Logistic Regression
- Multinomial Naive Bayes
- Pipeline
- GridSearchCV

---

# Dataset

Dataset Used from **Kaggle**:

- SMS Spam Collection Dataset

Features:

- **message** → SMS text
- **label** → ham / spam

Target Encoding:

| Original | Encoded |
|----------|----------|
| ham | 0 |
| spam | 1 |

---

# Preprocessing Steps

The following preprocessing steps were performed:

- Loaded dataset
- Removed unnecessary columns
- Renamed columns for readability
- Converted labels into numerical format
- Split dataset into training and testing sets

Training/Test Split:

- Train : 80%
- Test : 20%

Random State:

```
42
```

---

# Feature Engineering

Text data cannot be directly used by machine learning models.

TF-IDF (Term Frequency – Inverse Document Frequency) was used to convert text into numerical vectors.

The vectorizer learns:

- Important words
- Rare informative words
- Word frequencies
- N-grams

---

# Models Used

## 1. Logistic Regression

Pipeline:

- TF-IDF Vectorizer
- Logistic Regression

Hyperparameter tuning performed using GridSearchCV.

Parameters searched:

- max_features
- ngram_range
- Regularization parameter (C)

Scoring Metric:

```
F1 Score
```

---

## 2. Multinomial Naive Bayes

Pipeline:

- TF-IDF Vectorizer
- Multinomial Naive Bayes

Used as a baseline model for comparison.

---

# Hyperparameter Tuning

GridSearchCV was used to find the best parameter combination.

Parameters searched:

```python
max_features = [1000, 3000, 5000]

ngram_range = [(1,1), (1,2)]

C = [0.1, 1, 10]
```

Cross Validation:

```
5 Fold Cross Validation
```

Optimization Metric:

```
F1 Score
```

---

# Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

Both models were evaluated on the unseen test dataset.

---

# Model Comparison

| Model | Purpose |
|--------|----------|
| Logistic Regression | Main classifier |
| Multinomial Naive Bayes | Baseline comparison |

This comparison demonstrates how different algorithms perform on the same NLP task.

---

# Key Learning

This project reinforced several important NLP and Machine Learning concepts:

- Text data must be converted into numerical form before model training.
- TF-IDF gives higher importance to informative words while reducing the weight of common words.
- Logistic Regression performs exceptionally well for binary text classification.
- Naive Bayes provides a strong baseline for NLP problems.
- Pipelines prevent preprocessing mistakes and simplify workflows.
- GridSearchCV helps automatically find better hyperparameters.

---

# Conclusion

This project demonstrates a complete NLP classification pipeline using TF-IDF feature extraction and two supervised machine learning algorithms.

By comparing Logistic Regression and Multinomial Naive Bayes, the project highlights how different classifiers perform on text classification tasks while emphasizing the importance of feature engineering, hyperparameter tuning, and proper model evaluation.

It serves as a strong foundation for more advanced Natural Language Processing and Large Language Model (LLM) projects.
