X = df1[features]
# y = df1["Survived"]
# X_train , X_val , y_train , y_val = train_test_split(X , y , test_size = 0.2 , random_state = 42)
# pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model",  RandomForestClassifier(random_state=42))
# ])
# param_grid = {
#     "model__n_estimators": [10, 50, 100],
#     "model__max_depth":    [3, 5, None]
# }
# grid = GridSearchCV(pipe, param_grid, cv=5)
# grid.fit(X_train, y_train) 
# print(grid.best_params_)
# print(grid.best_score_)
# predictions = grid.predict(X_val)
# accuracy = accuracy_score(y_val, predictions) 
# print(f"Validation Accuracy: {accuracy:.2f}") 
# scores = cross_val_score(pipe, X, y, cv=5)
# print(scores.mean())
# print(scores.std())




