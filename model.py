import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
train = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=columns)

max_cycles = train.groupby('engine_id')['cycle'].max()
train = train.merge(max_cycles.rename('max_cycle'), on='engine_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train = train.drop(columns=['max_cycle'])

std_values = train.std()
useless = std_values[std_values <= 0.001].index.tolist()
train = train.drop(columns=useless)

sensor_cols = [col for col in train.columns if 'sensor' in col]
scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])

X = train[sensor_cols]
y = train['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R2   : {r2:.2f}")

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
mae_rf = mean_absolute_error(y_test, rf_pred)
r2_rf = r2_score(y_test, rf_pred)

print("\nRandom Forest Results:")
print(f"RMSE : {rmse_rf:.2f}")
print(f"MAE  : {mae_rf:.2f}")
print(f"R2   : {r2_rf:.2f}") 

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
mae_xgb = mean_absolute_error(y_test, xgb_pred)
r2_xgb = r2_score(y_test, xgb_pred)

print("\nXGBoost Results:")
print(f"RMSE : {rmse_xgb:.2f}")
print(f"MAE  : {mae_xgb:.2f}")
print(f"R2   : {r2_xgb:.2f}")

print("\nFinal Comparison:")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R2':<10}")
print(f"{'Linear Regression':<20} {rmse:<10.2f} {mae:<10.2f} {r2:<10.2f}")
print(f"{'Random Forest':<20} {rmse_rf:<10.2f} {mae_rf:<10.2f} {r2_rf:<10.2f}")
print(f"{'XGBoost':<20} {rmse_xgb:<10.2f} {mae_xgb:<10.2f} {r2_xgb:<10.2f}")