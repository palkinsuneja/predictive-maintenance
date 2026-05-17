# ✈️ Jet Engine Predictive Maintenance

> Predict Remaining Useful Life (RUL) of jet engines using sensor data and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 About

This project builds a **Predictive Maintenance System** using NASA's CMAPSS Turbofan Engine dataset. The system predicts how many cycles a jet engine has before failure — enabling proactive maintenance decisions.

**Real-world impact:** Unplanned engine failures cost airlines millions. Early prediction saves lives and money.

---

## 🚀 Live Demo

Run the interactive Streamlit app:
```bash
streamlit run app.py
```
Adjust sensor sliders → Get real-time RUL prediction!

---

## 📊 Results

| Model | RMSE | MAE | R2 |
|-------|------|-----|----|
| Linear Regression | 44.32 | 34.04 | 0.57 |
| Random Forest | 41.38 | 29.52 | **0.63** ✅ |
| XGBoost | 41.50 | 29.65 | 0.62 |

**Random Forest** achieved best performance.

---

## 🧠 ML Pipeline

Raw Sensor Data → RUL Calculation → Feature Selection →
Normalization → Train/Test Split → Model Training → Evaluation → Streamlit App

---

## 📁 Project Structure

predictive-maintenance/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── explore.py        # Data exploration
├── visualize.py      # Sensor visualization
├── preprocess.py     # Data preprocessing
├── model.py          # Model training & comparison
├── app.py            # Streamlit web app
└── requirements.txt


---

## 🔬 Key Concepts

- **RUL (Remaining Useful Life)** — cycles left before engine failure
- **Feature Selection** — removed sensors with std ≈ 0
- **MinMax Normalization** — all sensors scaled to [0,1]
- **Model Comparison** — Linear Regression vs Random Forest vs XGBoost
- **Interactive UI** — real-time predictions via Streamlit

---

## 📜 Dataset

NASA CMAPSS Turbofan Engine Degradation Dataset
- 100 engines, 21 sensors, 20,631 readings
- Verified at: [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmapss)

---

## 👩‍💻 Author

**Palkin Suneja** — [GitHub](https://github.com/palkinsuneja) · [LinkedIn](https://linkedin.com/in/palkinsuneja)

