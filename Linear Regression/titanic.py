import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV , cross_val_score
from sklearn.pipeline import Pipeline

df1 = pd.read_csv("train.csv")
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
df1["Embarked"] = df1["Embarked"].fillna(df1["Embarked"].mode()[0])
df1["Fare"] = df1["Fare"].fillna(df1["Fare"].mean())
df1["Title"] = df1["Name"].apply(
    lambda name: name.split(",")[1].split(".")[0].strip()
)
df1["Title"] = df1["Title"].replace( 
    ["Lady", "Countess", "Capt", "Col", "Don",
     "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona", "the Countess"],
    "Rare")
df1["Title"] = df1["Title"].replace("Mlle", "Miss")
df1["Title"] = df1["Title"].replace("Ms",   "Miss")
df1["Title"] = df1["Title"].replace("Mme",  "Mrs")
df1 = pd.get_dummies(df1, columns=["Sex", "Embarked"], drop_first=True)
df1 = pd.get_dummies(df1, columns=["Title"], drop_first=False)
df1["family_size"] = df1["SibSp"] + df1["Parch"] + 1
df1["is_alone"]    = (df1["family_size"] == 1).astype(int)
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S", "Sex_male", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Master", "Title_Rare", "family_size", "is_alone"]
X = df1[features]
y = df1["Survived"]
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size = 0.2 , random_state = 42)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(random_state=42))
])
param_grid = {
    "model__n_estimators": [10, 50, 100, 150, 200],
    "model__max_depth":    [3, 5, 7, 9, None]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train) 
print(grid.best_params_)
print(grid.best_score_)
predictions = grid.predict(X_val)
accuracy = accuracy_score(y_val, predictions) 
print(f"Validation Accuracy: {accuracy:.2f}") 
scores = cross_val_score(pipe, X, y, cv=5)
print(scores.mean())
print(scores.std())




df1 = pd.read_csv("train.csv")
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
df1["Embarked"] = df1["Embarked"].fillna(df1["Embarked"].mode()[0])
df1["Fare"] = df1["Fare"].fillna(df1["Fare"].mean())
df1["Title"] = df1["Name"].apply(
    lambda name: name.split(",")[1].split(".")[0].strip()
)
df1["Title"] = df1["Title"].replace( 
    ["Lady", "Countess", "Capt", "Col", "Don",
     "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona", "the Countess"],
    "Rare")
df1["Title"] = df1["Title"].replace("Mlle", "Miss")
df1["Title"] = df1["Title"].replace("Ms",   "Miss")
df1["Title"] = df1["Title"].replace("Mme",  "Mrs")
df1 = pd.get_dummies(df1, columns=["Sex", "Embarked"], drop_first=True)
df1 = pd.get_dummies(df1, columns=["Title"], drop_first=False)
df1["family_size"] = df1["SibSp"] + df1["Parch"] + 1
df1["is_alone"]    = (df1["family_size"] == 1).astype(int)
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S", "Sex_male", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Master", "Title_Rare", "family_size", "is_alone"]
X = df1[features]
y = df1["Survived"]
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size = 0.2 , random_state = 42)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression())
])
param_grid = {
    "model__C": [1, 0.1, 0.01]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train) 
print(grid.best_params_)
print(grid.best_score_)
predictions = grid.predict(X_val)
accuracy = accuracy_score(y_val, predictions) 
print(f"Validation Accuracy: {accuracy:.2f}") 
scores = cross_val_score(pipe, X, y, cv=5)
print(scores.mean())
print(scores.std())




