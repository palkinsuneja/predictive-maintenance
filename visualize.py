import pandas as pd
import matplotlib.pyplot as plt

columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
train = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=columns)

max_cycles = train.groupby('engine_id')['cycle'].max()
train = train.merge(max_cycles.rename('max_cycle'), on='engine_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train = train.drop(columns=['max_cycle'])

engine1 = train[train['engine_id'] == 1]

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(engine1['cycle'], engine1['sensor2'])
axes[0].set_title('Sensor 2 over time')
axes[0].set_xlabel('Cycle')
axes[0].set_ylabel('Value')

axes[1].plot(engine1['cycle'], engine1['sensor11'], color='orange')
axes[1].set_title('Sensor 11 over time')
axes[1].set_xlabel('Cycle')
axes[1].set_ylabel('Value')

axes[2].plot(engine1['cycle'], engine1['RUL'], color='red')
axes[2].set_title('RUL over time')
axes[2].set_xlabel('Cycle')
axes[2].set_ylabel('RUL')

plt.tight_layout()
plt.savefig('sensor_plots.png')
print("Plot saved!")