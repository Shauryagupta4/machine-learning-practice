import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score 


data = {
    "area":         [1000, 1500, 1200, 1800, 2000,
                     2500, 900,  1100, 3000, 2200,
                     1600, 1400, 2800, 1900, 1300],
    "bedrooms":     [2, 3, 2, 4, 3, 5, 1, 2, 5, 4,
                     3, 2, 4, 3, 2],
    "bathrooms":    [1, 2, 1, 2, 2, 3, 1, 1, 3, 2,
                     2, 1, 3, 2, 1],
    "age":          [10, 5, 15, 3, 8, 1, 20, 12, 2,
                     6, 4, 18, 1, 7, 14],
    "parking":      [1, 2, 1, 2, 2, 3, 0, 1, 3, 2,
                     1, 1, 2, 2, 1],
    "location":     ["urban", "suburban", "urban",
                     "suburban", "urban", "suburban",
                     "rural", "urban", "suburban",
                     "urban", "rural", "urban",
                     "suburban", "urban", "rural"],
    "price":        [5000, 7500, 5500, 9000, 9500,
                     13000, 3500, 4800, 15000, 11000,
                     7000, 5200, 14000, 9200, 4500]
}

np.random.seed(42)
data["area"][3]      = np.nan
data["bedrooms"][7]  = np.nan
data["location"][11] = None

df = pd.DataFrame(data)
print("Length :", len(df))
print("Shape: " , df.shape)
print("Data type: " , df.dtypes)
print("Basic statistics: " , df.describe())
print("Missing Values: \n" , df.isnull().sum())
print(df["location"].value_counts())
print(df.isnull().sum() / len(df) * 100) 

df["area"] = df["area"].fillna(df["area"].mean())
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].mean())
df["location"] = df["location"].fillna(df["location"].mode()[0])
df = pd.get_dummies(df, columns=["location"] , drop_first = True)
print(df.head())

X = df[["area" , "bedrooms" , "bathrooms" , "age" , "parking" , "location_suburban" , "location_urban"]]
y = df["price"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled , y_train)

prediction = model.predict(X_test_scaled)
print("Prediction: " , prediction)

mae = mean_absolute_error(y_test , prediction)
mse = mean_squared_error(y_test , prediction)
rmse = mean_squared_error(y_test , prediction)**0.5
r2 = r2_score(y_test , prediction)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

for i in range (len(y_test)):
    print(f"ACTUAL: {y_test.iloc[i]} | PREDICTED: {prediction[i]}")