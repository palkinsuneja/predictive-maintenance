import pandas as pd
import numpy as np

columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
train = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=columns)

# RUL calculate karo
max_cycles = train.groupby('engine_id')['cycle'].max()
train = train.merge(max_cycles.rename('max_cycle'), on='engine_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train = train.drop(columns=['max_cycle'])

# Step 1 - Useless sensors dekho
print("Sensor wise standard deviation:")
print(train.iloc[:, 5:26].std().sort_values())

# std = 0 wale sensors automatically hatao
std_values = train.std()
useless = std_values[std_values <= 0.001].index.tolist()
train = train.drop(columns=useless)

print("Removed sensors:", useless)
print("Shape after dropping:", train.shape)
print("Remaining columns:", train.columns.tolist())

from sklearn.preprocessing import MinMaxScaler
sensor_cols = [col for col in train.columns if 'sensor' in col]

scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])

print("\nAfter normalization:")
print(train[sensor_cols].describe().round(2))


from sklearn.model_selection import train_test_split

sensor_cols = [col for col in train.columns if 'sensor' in col]
X = train[sensor_cols]
y = train['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)