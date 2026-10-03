# 🏠 Housing Price Prediction

A machine learning web application that predicts house prices based on property features such as area, number of bedrooms, bathrooms, stories, and parking spaces.

Built using **Python, Scikit-learn, and Streamlit**, this project explores multiple regression techniques and provides an interactive interface for real-time house price estimation.

## 🚀 Live Demo

**[Try the Housing Price Prediction App](https://house-prediction-021026.streamlit.app/)**

Enter property details and get an estimated house price instantly.

## 📌 Project Overview

The objective of this project is to develop a machine learning model capable of estimating housing prices using property-related features.

The project covers the complete machine learning workflow:

* Data loading and exploratory data analysis
* Data preprocessing and feature selection
* Training and comparison of regression models
* Model evaluation using multiple performance metrics
* Model serialization using Joblib
* Deployment of an interactive web application using Streamlit

## 📊 Dataset

The dataset contains **545 records and 6 columns**, including five input features and one target variable.

| Feature     | Description                   |
| ----------- | ----------------------------- |
| `area`      | Area of the property          |
| `bedrooms`  | Number of bedrooms            |
| `bathrooms` | Number of bathrooms           |
| `stories`   | Number of floors              |
| `parking`   | Number of parking spaces      |
| `price`     | Target variable (house price) |

The dataset was checked for missing values before model training.

## 🤖 Machine Learning Models

Three regression approaches were implemented and evaluated:

1. **Simple Linear Regression** – Uses area as the sole predictor.
2. **Multiple Linear Regression** – Uses all five property features.
3. **Polynomial Regression (Degree 2)** – Captures potential nonlinear relationships between features and house prices.

### Model Comparison

The following results were obtained on an 80/20 train-test split using `random_state=42`.

| Metric   | Linear Regression | Multiple Linear Regression | Polynomial Regression |
| -------- | ----------------: | -------------------------: | --------------------: |
| MAE      |         1,474,748 |                  1,127,483 |             1,111,159 |
| RMSE     |         1,917,104 |                  1,514,174 |             1,525,169 |
| R² Score |            0.2729 |                     0.5464 |                0.5398 |
| MAPE     |            32.28% |                     25.50% |                24.39% |

**Final Model: Multiple Linear Regression**

Multiple Linear Regression was selected based on its overall evaluation results, particularly its R² score and RMSE. Although Polynomial Regression achieved slightly lower MAE and MAPE, Multiple Linear Regression provided a stronger overall balance of performance metrics.

The final model achieved an R² score of approximately **0.5464**, indicating that it explains around 54.64% of the variance in the test-set housing prices. This leaves room for future improvements through feature engineering and more advanced algorithms.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Model Serialization:** Joblib
* **Web Application:** Streamlit
* **Version Control:** Git and GitHub
* **Deployment:** Streamlit Community Cloud

## 📂 Project Structure

```text
HOUSE-PREDICTION/
│
├── app.py
├── main.py
├── Housing_Price_Dataset.csv
├── best_housing_price_model.pkl
├── model_info.pkl
├── regression_model_comparison.csv
├── requirements.txt
└── README.md
```

| File                              | Purpose                                    |
| --------------------------------- | ------------------------------------------ |
| `app.py`                          | Streamlit web application                  |
| `main.py`                         | Model training, evaluation, and comparison |
| `Housing_Price_Dataset.csv`       | Dataset used for training                  |
| `best_housing_price_model.pkl`    | Serialized final machine learning model    |
| `model_info.pkl`                  | Saved model metadata                       |
| `regression_model_comparison.csv` | Model evaluation results                   |
| `requirements.txt`                | Project dependencies                       |

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shauryagupta4/machine-learning-practice.git
```

### 2. Navigate to the project directory

```bash
cd machine-learning-practice/HOUSE-PREDICTION
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🔮 Future Improvements

* Experiment with Random Forest, Gradient Boosting, and other advanced regression algorithms.
* Improve predictive accuracy through feature engineering and hyperparameter tuning.
* Apply cross-validation for more robust model evaluation.
* Incorporate additional features such as location, property age, and furnishing status.
* Enhance the user interface with interactive visualizations.
* Explore model explainability techniques to understand feature contributions.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

* Understanding and implementing regression algorithms.
* Evaluating machine learning models using multiple metrics.
* Comparing model performance and selecting a suitable final model.
* Saving and loading trained models using Joblib.
* Building interactive ML applications with Streamlit.
* Deploying machine learning projects using GitHub and Streamlit Community Cloud.

## 👨‍💻 Author

**Shaurya Gupta**

CSE – Artificial Intelligence and Machine Learning Student

* GitHub: [Shauryagupta4](https://github.com/Shauryagupta4)
* Project Repository: [Machine Learning Practice](https://github.com/Shauryagupta4/machine-learning-practice)

---

⭐ If you find this project useful, consider giving the repository a star!
