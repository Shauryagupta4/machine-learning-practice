import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error
)


# LOAD DATASET

df = pd.read_csv("Housing_Price_Dataset.csv")

print("Original Dataset Shape:", df.shape)
print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# DATA CLEANING

# Remove duplicate rows
df = df.drop_duplicates()

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Ensure all columns are numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Separate features and target
target = "price"

# Remove rows where target is missing
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

# Handle missing feature values using median imputation
imputer = SimpleImputer(strategy="median")
X_clean = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print("\nCleaned Dataset Shape:", X_clean.shape)
print("\nRemaining Missing Values:")
print(X_clean.isnull().sum())


# SPLIT DATA INTO TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(
    X_clean,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# DEFINE MODELS

models = {
    "Linear Regression": Pipeline([
        ("model", LinearRegression())
    ]),

    "Multiple Linear Regression": Pipeline([
        ("model", LinearRegression())
    ]),

    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(
            degree=2,
            include_bias=False
        )),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
}


# TRAIN MODELS AND EVALUATE

results = []
predictions = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    if name == "Linear Regression":
        model.fit(X_train[["area"]], y_train)
        y_pred = model.predict(X_test[["area"]])

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Number of model input features/terms
    if name == "Linear Regression":
        k = 1

    elif name == "Multiple Linear Regression":
        k = X_train.shape[1]

    else:
        k = model.named_steps["poly"].n_output_features_

    n = len(y_test)

    # Adjusted R-squared
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2 Score": r2,
        "Adjusted R2": adj_r2,
        "Explained Variance": evs
    })


# CREATE COMPARISON TABLE

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


# SELECT BEST MODEL

best_model_name = results_df.loc[
    results_df["R2 Score"].idxmax(),
    "Model"
]

print("\nBest Model Based on Test R2:", best_model_name)


# VISUALIZE MODEL COMPARISON

# R2 comparison
plt.figure(figsize=(9, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)
plt.title("R2 Score Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# MAE comparison
plt.figure(figsize=(9, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="MAE"
)
plt.title("Mean Absolute Error Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# RMSE comparison
plt.figure(figsize=(9, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)
plt.title("Root Mean Squared Error Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# 10. ACTUAL VS PREDICTED VALUES
plt.figure(figsize=(8, 6))

for name, y_pred in predictions.items():
    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5,
        label=name
    )

min_val = min(y_test.min(), min(p.min() for p in predictions.values()))
max_val = max(y_test.max(), max(p.max() for p in predictions.values()))

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()


# 11. SAVE RESULTS

results_df.to_csv("regression_model_comparison.csv", index=False)

print("\nComparison results saved successfully!")


# # Select best model based on R2 score
# best_model_name = results_df.loc[
#     results_df["R2 Score"].idxmax(),
#     "Model"
# ]

# print("Selected Model:", best_model_name)

# # Get the selected model
# final_model = models[best_model_name]

# # Retrain the selected model on the complete dataset
# if best_model_name == "Linear Regression":
#     final_model.fit(X_clean[["area"]], y)
# else:
#     final_model.fit(X_clean, y)

# # Save the fully trained model
# os.makedirs("model", exist_ok=True)

# model_path = "model/best_housing_price_model.pkl"

# joblib.dump(final_model, model_path)

# # Save model metadata
# model_info = {
#     "model_name": best_model_name,
#     "features": (
#         ["area"]
#         if best_model_name == "Linear Regression"
#         else list(X_clean.columns)
#     ),
#     "target": "price"
# }

# joblib.dump(
#     model_info,
#     "model/model_info.pkl"
# )

# print("\nModel successfully trained and saved!")
# print("Model path:", model_path)

# # Verify that the saved model can be loaded
# loaded_model = joblib.load(model_path)

# print("Model loaded successfully!")
# print("Model type:", type(loaded_model))

# # Test prediction
# sample = X_clean.iloc[[0]][model_info["features"]]

# prediction = loaded_model.predict(sample)

# print("Test Prediction:", prediction[0])
# print("Actual Price:", y.iloc[0])

# # Download in Google Colab
# try:
#     from google.colab import files
#     files.download(model_path)
#     files.download("model/model_info.pkl")
# except ImportError:
#     print("Files are saved in the model folder.")

import joblib

model = joblib.load("best_housing_price_model.pkl")

print("Model Type:", type(model))
print("\nModel Contents:")
print(model)

# Check if it is a trained Scikit-learn model
if hasattr(model, "named_steps"):
    print("\nPipeline Steps:")
    print(model.named_steps)

    final_estimator = model.steps[-1][1]

    if hasattr(final_estimator, "coef_"):
        print("\nLearned Coefficients:")
        print(final_estimator.coef_)

        print("\nLearned Intercept:")
        print(final_estimator.intercept_)

elif hasattr(model, "coef_"):
    print("\nLearned Coefficients:")
    print(model.coef_)

    print("\nLearned Intercept:")
    print(model.intercept_)

else:
    print("\nThis object does not expose Linear Regression parameters.")