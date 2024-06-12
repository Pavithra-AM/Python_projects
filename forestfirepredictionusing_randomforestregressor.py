import pandas as panda
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_squared_error as mse, r2_score as r2

file_path = '/content/forestfires.csv'
fire = panda.read_csv(file_path)

print(fire.head())

fire_encode = panda.get_dummies(fire, columns=['month', 'day'])

X = fire_encode.drop(['area'], axis=1)
y = np.log1p(fire_encode['area'])
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

scaler = ss()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rfr(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found: ", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)
train_r2 = r2(y_train, y_train_pred)
test_r2 = r2(y_test, y_test_pred)

print(f"Training Mean Squared Error (MSE): {train_mse}")
print(f"Testing Mean Squared Error (MSE): {test_mse}")
print(f"Training R-squared (R²): {train_r2}")
print(f"Testing R-squared (R²): {test_r2}")


OUTPU:
X  Y month  day  FFMC   DMC     DC  ISI  temp  RH  wind  rain  area
0  7  5   mar  fri  86.2  26.2   94.3  5.1   8.2  51   6.7   0.0   0.0
1  7  4   oct  tue  90.6  35.4  669.1  6.7  18.0  33   0.9   0.0   0.0
2  7  4   oct  sat  90.6  43.7  686.9  6.7  14.6  33   1.3   0.0   0.0
3  8  6   mar  fri  91.7  33.3   77.5  9.0   8.3  97   4.0   0.2   0.0
4  8  6   mar  sun  89.3  51.3  102.2  9.6  11.4  99   1.8   0.0   0.0
Best parameters found:  {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}
Training Mean Squared Error (MSE): 0.9607841858925448
Testing Mean Squared Error (MSE): 2.2356194111865624
Training R-squared (R²): 0.49154484732301473
Testing R-squared (R²): -0.017179352533887693
