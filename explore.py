import pandas as pd
import numpy as np

columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]

train = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=columns)

print("Shape:", train.shape)
print("\nPehli 5 rows:")
print(train.head())

engine1 = train[train['engine_id'] == 1]
print("\nEngine 1 ka total cycles:", engine1['cycle'].max())
print("Matlab engine 1 ne itne cycles mein fail hua!")

max_cycles = train.groupby('engine_id')['cycle'].max()

# RUL column add karo
train = train.merge(max_cycles.rename('max_cycle'), on='engine_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train = train.drop(columns=['max_cycle'])

print("\nRUL column added!")
print(train[['engine_id', 'cycle', 'RUL']].head(10))
print("\nEngine 1 last 5 rows (RUL = 0 hona chahiye end mein):")
print(train[train['engine_id']==1][['engine_id','cycle','RUL']].tail())

max_cycles = train.groupby('engine_id')['cycle'].max()
train = train.merge(max_cycles.rename('max_cycle'), on='engine_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train = train.drop(columns=['max_cycle'])

print(train[['engine_id', 'cycle', 'RUL']].head(10))
print(train[train['engine_id']==1][['engine_id','cycle','RUL']].tail())
