# Customer Churn Prediction using Random Forest

## Project Overview

This project predicts customer churn using a **Random Forest Classifier** built with **Scikit-Learn**.

The project demonstrates a complete machine learning workflow including:

- Data preprocessing
- One-Hot Encoding
- Pipeline implementation
- Model training
- Model evaluation
- Performance analysis
- Investigation of poor generalization

---

# Technologies Used

- Python
- Pandas
- Scikit-Learn
- Random Forest Classifier
- Pipeline
- StandardScaler

---

# Dataset

The project uses two separate CSV files from **Kaggle**:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Both datasets contain customer information such as:

- Age
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction
- Gender
- Subscription Type
- Contract Length

Target Variable:

- **Churn**

---

# Preprocessing Steps

The following preprocessing steps were performed:

- Removed missing values
- Removed CustomerID
- Applied One-Hot Encoding to categorical features
- Used `drop_first=True`
- Selected relevant numerical and encoded features

---

# Model

A Scikit-Learn Pipeline was used.

Pipeline:

- StandardScaler
- RandomForestClassifier

Model Parameters:

- n_estimators = 100
- max_depth = 3
- random_state = 42

---

# Results

Test Accuracy

```
0.534
```

Classification Report

```
              precision    recall    f1-score

Class 0          0.98       0.12      0.21
Class 1          0.50       1.00      0.67

Accuracy                               0.53
```

---

# Investigation Performed

Since the test accuracy appeared unusually low, several verification steps were performed before concluding that the issue was related to the dataset rather than the implementation.

## 1. Verified Preprocessing

- No missing values remained
- Correct One-Hot Encoding
- CustomerID removed
- Proper feature selection
- No data leakage
- Pipeline used correctly

---

## 2. Checked Feature Correlation

```python
print(df1.corr()["Churn"].sort_values(ascending=False))
```

Purpose:

- Verify whether meaningful relationships exist between features and the target.

---

## 3. Checked Dataset Statistics

```python
print(df1.describe())
print(df2.describe())
```

Purpose:

- Compare train and test distributions.

---

## 4. Checked Class Distribution

```python
print(df1["Churn"].value_counts())
```

Purpose:

- Detect class imbalance.

---

## 5. Visualized Feature Distributions

```python
df1.hist(figsize=(12,10))
plt.show()
```

Purpose:

- Identify unusual or synthetic feature patterns.

---

# Observation

After verifying the preprocessing pipeline and inspecting the data, it was observed that the model heavily favored one class during prediction.

This suggested that the poor performance was **not caused by coding mistakes**, but rather by differences between the training and testing datasets.

The training dataset contains **440,832 rows**, while the testing dataset contains **64,374 rows**.

Both files were generated **separately using different synthetic distributions**, resulting in a distribution mismatch.

This causes the model to generalize poorly despite correct preprocessing and implementation.

---

# Key Learning

This project reinforced several important machine learning concepts:

- Low accuracy is **not always caused by bad code**.
- Always inspect the **classification report**, not just accuracy.
- Verify preprocessing before changing the model.
- Compare train and test distributions.
- Detect possible data leakage.
- Investigate dataset quality before excessive hyperparameter tuning.

---

# Future Improvements

Possible improvements include:

- Hyperparameter tuning using GridSearchCV
- Trying Gradient Boosting models
- XGBoost
- LightGBM
- Better feature engineering
- Cross-validation on datasets with similar distributions
- Testing on real-world customer churn datasets

---

# Skills Demonstrated

- Data Cleaning
- Feature Engineering
- One-Hot Encoding
- Pipeline Creation
- Random Forest Classification
- Model Evaluation
- Classification Report Analysis
- Debugging Machine Learning Models
- Understanding Distribution Shift
- Investigating Dataset Quality

---

# Conclusion

Although the model achieved only **53.4% test accuracy**, the investigation showed that the primary limitation was the synthetic distribution mismatch between the training and testing datasets rather than the implementation itself.

This project demonstrates the importance of evaluating data quality and understanding dataset characteristics in addition to building machine learning models.