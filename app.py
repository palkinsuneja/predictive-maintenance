import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predictive Maintenance", page_icon="✈️", layout="wide")
st.title("✈️ Jet Engine Predictive Maintenance")
st.markdown("### Predict Remaining Useful Life (RUL) of Jet Engines")

@st.cache_resource
def train_model():
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
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, sensor_cols

model, scaler, sensor_cols = train_model()

st.sidebar.header("🔧 Engine Sensor Inputs")
st.sidebar.markdown("Adjust sensor values to predict RUL")

sensor_values = {}
for sensor in sensor_cols:
    sensor_values[sensor] = st.sidebar.slider(sensor, 0.0, 1.0, 0.5, 0.01)

input_df = pd.DataFrame([sensor_values])
prediction = model.predict(input_df)[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted RUL", f"{prediction:.0f} cycles")

with col2:
    if prediction > 100:
        st.success("✅ Engine Healthy")
    elif prediction > 50:
        st.warning("⚠️ Monitor Closely")
    else:
        st.error("🔴 Maintenance Required!")

with col3:
    health = min(prediction / 200 * 100, 100)
    st.metric("Engine Health", f"{health:.1f}%")

st.markdown("---")
st.subheader("📊 Sensor Values Overview")
sensor_df = pd.DataFrame(list(sensor_values.items()), columns=['Sensor', 'Value'])
st.bar_chart(sensor_df.set_index('Sensor'))