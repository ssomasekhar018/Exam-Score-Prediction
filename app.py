import streamlit as st
import pickle
import numpy as np
import pandas as pd
import io
import altair as alt
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -----------------------------
# Config & model load
# -----------------------------
st.set_page_config(page_title="Student Exam Score Prediction", page_icon="📘", layout="wide")
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = pickle.load(open("best_model.pkl", "rb"))
except Exception as e:
    st.error("Could not load model 'best_model.pkl'. Make sure the file exists.")
    raise


# -----------------------------
# Mapping (must match training)
# -----------------------------
mapping = {
    "gender": {"male": 1, "other": 2, "female": 3},
    "course": {
        "diploma": 1, "bca": 2, "b.tech": 3,
        "b.sc": 4, "bba": 5, "ba": 6, "b.com": 7
    },
    "internet_access": {"yes": 1, "no": 2},
    "sleep_quality": {"poor": 1, "average": 2, "good": 3},
    "study_method": {
        "coaching": 1, "online videos": 2, "mixed": 3,
        "self-study": 4, "group study": 5
    },
    "facility_rating": {"low": 1, "medium": 2, "high": 3},
    "exam_difficulty": {"hard": 1, "moderate": 2, "easy": 3},
}


def encode(column, value):
    return mapping[column].get(value, np.nan)


FEATURE_NAMES = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
    "age",
    "study_hours",
    "class_attendance",
    "sleep_hours",
]


def predict_row(data_row):
    arr = np.array([[
        encode("gender", data_row["gender"]),
        encode("course", data_row["course"]),
        encode("internet_access", data_row["internet_access"]),
        encode("sleep_quality", data_row["sleep_quality"]),
        encode("study_method", data_row["study_method"]),
        encode("facility_rating", data_row["facility_rating"]),
        encode("exam_difficulty", data_row["exam_difficulty"]),
        float(data_row["age"]),
        float(data_row["study_hours"]),
        float(data_row["class_attendance"]),
        float(data_row["sleep_hours"]),
    ]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return float(model.predict(arr)[0])


# -----------------------------
# Session state for history
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Sidebar - presets and batch
# -----------------------------
with st.sidebar:
    st.header("Options")
    preset = st.selectbox("Example profiles / presets", [
        "-- none --",
        "Typical low performer",
        "Typical high performer",
        "Average student",
    ])
    st.markdown("---")
    st.write("Batch predictions")
    upload = st.file_uploader("Upload CSV (columns must match input names)", type=["csv"]) 
    st.write("CSV must contain: `gender,course,internet_access,sleep_quality,study_method,facility_rating,exam_difficulty,age,study_hours,class_attendance,sleep_hours`")
    st.markdown("---")
    if st.button("Clear history"):
        st.session_state.history = []


# -----------------------------
# Main UI
# -----------------------------
title_col1, title_col2 = st.columns([8, 2])
with title_col1:
    st.markdown("<div style='display:flex;align-items:center'>\n<h1 style='margin:0;padding:0 12px 0 0'>📘 Student Exam Score Prediction</h1>\n</div>", unsafe_allow_html=True)
    st.write("Fill the details below to predict an exam score. Use the sidebar to upload CSVs or pick presets.")


left, right = st.columns([2, 1])
with left:
    with st.form("predict_form"):
        gender = st.selectbox("Gender", ["male", "other", "female"])
        course = st.selectbox("Course",
                              ["diploma", "bca", "b.tech", "b.sc", "bba", "ba", "b.com"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
        sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
        study_method = st.selectbox("Study Method",
                                    ["coaching", "online videos", "mixed",
                                     "self-study", "group study"]) 
        facility = st.selectbox("Facility Rating", ["low", "medium", "high"])
        difficulty = st.selectbox("Exam Difficulty", ["hard", "moderate", "easy"])

        age = st.slider("Age", 10, 40, 20)
        study_hours = st.slider("Study Hours per Day", 0.0, 15.0, 2.0)
        class_attendance = st.slider("Class Attendance %", 0, 100, 75)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)

        submitted = st.form_submit_button("🔮 Predict Score")

    if submitted:
        data = {
            "gender": gender,
            "course": course,
            "internet_access": internet,
            "sleep_quality": sleep_quality,
            "study_method": study_method,
            "facility_rating": facility,
            "exam_difficulty": difficulty,
            "age": age,
            "study_hours": study_hours,
            "class_attendance": class_attendance,
            "sleep_hours": sleep_hours,
        }
        try:
            pred = predict_row(data)
            st.success(f"📘 Predicted Exam Score: **{pred:.2f}**")
            # record
            rec = data.copy()
            rec["predicted_score"] = round(pred, 2)
            st.session_state.history.append(rec)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Session history")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download history CSV", data=csv, file_name="predictions_history.csv", mime="text/csv")
    else:
        st.info("No predictions yet — make one to see history.")

with right:
    st.subheader("Model insights")
    # try feature importance
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            src = pd.DataFrame({"feature": FEATURE_NAMES, "importance": fi})
            chart = alt.Chart(src).mark_bar().encode(x="importance:Q", y=alt.Y("feature:N", sort="-x"))
            st.altair_chart(chart, use_container_width=True)
        elif hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            src = pd.DataFrame({"feature": FEATURE_NAMES, "coefficient": coefs})
            chart = alt.Chart(src).mark_bar().encode(x="coefficient:Q", y=alt.Y("feature:N", sort="-x"))
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Feature importance not available for this model type.")
    except Exception:
        st.write("Could not extract model insights.")


# -----------------------------
# Batch CSV processing
# -----------------------------
if upload is not None:
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

    if df is not None:
        required = set(FEATURE_NAMES)
        if not required.issubset(set(df.columns)):
            st.error("Uploaded CSV is missing required columns. See sidebar for expected names.")
        else:
            st.info("Running batch predictions...")
            preds = []
            for _, row in df.iterrows():
                try:
                    r = {k: row[k] for k in FEATURE_NAMES}
                    preds.append(predict_row(r))
                except Exception as e:
                    preds.append(np.nan)
            df["predicted_score"] = np.round(preds, 2)
            st.write("Batch results")
            st.dataframe(df)
            out_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download batch results", data=out_csv, file_name="batch_predictions.csv", mime="text/csv")

    
