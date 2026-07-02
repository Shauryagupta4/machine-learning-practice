import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv("spam.csv", encoding = 'latin-1')
df = df[["v1" , "v2"]]
df.columns = ["label" , "message"]
df["label"]= df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

param_grid = {
    "tfidf__max_features": [1000, 3000, 5000],
    "tfidf__ngram_range":  [(1,1), (1,2)],
    "model__C":            [0.1, 1, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)
print(grid.best_params_)
predictions = grid.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions))


pipe_nb = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
    ("model", MultinomialNB())
])

pipe_nb.fit(X_train, y_train)
predictions_nb = pipe_nb.predict(X_test)
print(f"NB Accuracy: {accuracy_score(y_test, predictions_nb):.3f}")
print(classification_report(y_test, predictions_nb))