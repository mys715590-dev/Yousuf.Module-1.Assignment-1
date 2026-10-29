import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "salary_model.pkl")
FEATURE_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")

if not os.path.isfile(MODEL_PATH) or not os.path.isfile(FEATURE_PATH):
    st.error("Trained model files not found.")
    st.write("App directory:", BASE_DIR)
    st.write("Artifacts directory:", ARTIFACT_DIR)
    st.write("Files available:", os.listdir(BASE_DIR))
    if os.path.isdir(ARTIFACT_DIR):
        st.write("Artifacts contents:", os.listdir(ARTIFACT_DIR))
    else:
        st.write("The artifacts folder is missing.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH) 