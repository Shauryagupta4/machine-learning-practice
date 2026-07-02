import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df1 = pd.read_csv("customer_churn_dataset-training-master.csv")
df1 = df1.dropna()
df1 = df1.drop(columns=["CustomerID"], errors="ignore")
df1 = pd.get_dummies(df1 , columns = ["Gender" , "Subscription Type" , "Contract Length"] , drop_first = True)

X_train = df1[['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
       'Payment Delay', 'Total Spend', 'Last Interaction',
       'Gender_Male', 'Subscription Type_Premium',
       'Subscription Type_Standard', 'Contract Length_Monthly',
       'Contract Length_Quarterly']]
y_train = df1['Churn']

df2 = pd.read_csv("customer_churn_dataset-testing-master.csv")
df2 = df2.dropna()
df2 = df2.drop(columns=["CustomerID"], errors="ignore")
df2 = pd.get_dummies(df2 , columns = ["Gender" , "Subscription Type" , "Contract Length"] , drop_first = True)

X_test = df2[['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
       'Payment Delay', 'Total Spend', 'Last Interaction',
       'Gender_Male', 'Subscription Type_Premium',
       'Subscription Type_Standard', 'Contract Length_Monthly',
       'Contract Length_Quarterly']]

y_test = df2['Churn']

pipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators = 100, max_depth = 3 , random_state = 42))
])

pipe.fit(X_train , y_train)

predictions = pipe.predict(X_test)
print(f"Test Score:  {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions))
