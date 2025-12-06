# 📘 Student Exam Score Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-NDArray-013243?logo=numpy&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-Charts-FF3A5A)

A fast, friendly Streamlit application that predicts a student's exam score using demographics, study habits, sleep patterns, and environment factors. Supports single predictions, batch CSV uploads, session history with export, and basic model insights.

---

## ✨ Features
- Single prediction form with real‑time result
- Batch CSV upload with automatic predictions and downloadable results
- Session history table with CSV export
- Basic model insights: feature importance or coefficients when available
- Clean, responsive UI with smart defaults

---

## 🧠 How It Works
- A trained `scikit-learn` model is loaded from `best_model.pkl`.
- Categorical inputs are encoded using predefined mappings in `app.py`.
- Numerical inputs are fed directly to the model.
- The result is displayed instantly and recorded in session history.

---

## 🚀 Quick Start

### Prerequisites
- Python installed

### Setup
```bash
python -m pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```
- Streamlit prints a local URL like `http://localhost:8502` — open it in your browser.

### Model File
- Place a trained model at `best_model.pkl` in the project root.
- If missing or incompatible, the app shows: `Could not load model 'best_model.pkl'`.

---

## 🧩 Inputs
- `gender`: `male`, `other`, `female`
- `course`: `diploma`, `bca`, `b.tech`, `b.sc`, `bba`, `ba`, `b.com`
- `internet_access`: `yes`, `no`
- `sleep_quality`: `poor`, `average`, `good`
- `study_method`: `coaching`, `online videos`, `mixed`, `self-study`, `group study`
- `facility_rating`: `low`, `medium`, `high`
- `exam_difficulty`: `hard`, `moderate`, `easy`
- `age`: integer
- `study_hours`: float
- `class_attendance`: percentage integer (0–100)
- `sleep_hours`: float

---

## 📦 Batch CSV Format
The uploaded CSV must include all columns below (order doesn’t matter):
```
gender,course,internet_access,sleep_quality,study_method,facility_rating,exam_difficulty,age,study_hours,class_attendance,sleep_hours
```
Example:
```csv
gender,course,internet_access,sleep_quality,study_method,facility_rating,exam_difficulty,age,study_hours,class_attendance,sleep_hours
male,b.tech,yes,good,self-study,high,moderate,20,2.5,80,7.0
female,b.sc,no,average,coaching,medium,easy,22,3.0,90,8.0
```
After processing, the app appends a `predicted_score` column and lets you download results.

---

## 📊 Model Insights
- If the model exposes `feature_importances_`, a bar chart is shown.
- If the model exposes `coef_`, a coefficient chart is shown.
- Otherwise, the app informs you that insights are not available for the model type.

---

## 🗂 Project Structure
```
.
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── best_model.pkl                 # Trained model (provided by you)
├── Exam_Score_Prediction.csv      # Example data
└── final_version_exam_score_analysis.ipynb  # Notebook used in development
```

---

## 🛠 Troubleshooting
- "Could not load model 'best_model.pkl'": ensure the file exists and matches your environment’s `scikit-learn` version.
- Port already in use: run `streamlit run app.py --server.port 8503`.
- Batch upload errors: verify columns match exactly and categorical values use the allowed options above.

---

## 🤝 Contributing
- Fork and create a feature branch.
- Keep UI clean and inputs consistent with model training.
- Open a PR with a short description and screenshots.

---

## 📜 License
- This project’s license is not specified. Add a `LICENSE` file if you plan to share or open‑source.

---

## 🌟 Credits
- Built with Streamlit, scikit‑learn, Pandas, NumPy, and Altair.
