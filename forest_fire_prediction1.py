
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = '/content/forestfires.csv'
forest_fires = pd.read_csv(file_path)

print(forest_fires.head())

forest_fires_encoded = pd.get_dummies(forest_fires, columns=['month', 'day'])

X = forest_fires_encoded.drop(['area'], axis=1)
y = np.log1p(forest_fires_encoded['area'])  # Log-transform the target variable to handle skewness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)

model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training Mean Squared Error (MSE): {train_mse}")
print(f"Testing Mean Squared Error (MSE): {test_mse}")
print(f"Training R-squared (R²): {train_r2}")
print(f"Testing R-squared (R²): {test_r2}")

OUTPUT:
X  Y month  day  FFMC   DMC     DC  ISI  temp  RH  wind  rain  area
0  7  5   mar  fri  86.2  26.2   94.3  5.1   8.2  51   6.7   0.0   0.0
1  7  4   oct  tue  90.6  35.4  669.1  6.7  18.0  33   0.9   0.0   0.0
2  7  4   oct  sat  90.6  43.7  686.9  6.7  14.6  33   1.3   0.0   0.0
3  8  6   mar  fri  91.7  33.3   77.5  9.0   8.3  97   4.0   0.2   0.0
4  8  6   mar  sun  89.3  51.3  102.2  9.6  11.4  99   1.8   0.0   0.0
Training Mean Squared Error (MSE): 0.31281493697173235
Testing Mean Squared Error (MSE): 2.3452482259429184
Training R-squared (R²): 0.8344556781085567
Testing R-squared (R²): -0.06705911572387668
