# app_streamlit.py

import streamlit as st
import pandas as pd
from src import preprocessing, train_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Upload student dataset and predict new student performance with improvement suggestions.")

# -----------------------------
# 1️⃣ Upload Dataset & Train Model
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload student dataset (CSV)", type=["csv"])

model = None
df_clean = None

if uploaded_file is not None:
    # Load and preprocess
    df_clean = preprocessing.run_preprocessing(uploaded_file)

    # Train the model
    model, accuracy, X_test, y_test = train_model.train(df_clean)

    st.success(f"✅ Data loaded and cleaned successfully! Model trained with accuracy: **{accuracy*100:.2f}%**")

# -----------------------------
# 2️⃣ New Student Prediction Form
# -----------------------------
st.subheader("📋 Enter New Student Details for Prediction")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["female", "male"])
    parental_education = st.selectbox(
        "Parental Level of Education",
        ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    )
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
    math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    submit_button = st.form_submit_button("Predict Performance")

# -----------------------------
# 3️⃣ Run Prediction
# -----------------------------
if submit_button:
    if model is None:
        st.error("⚠ Please upload and train the model with a dataset first.")
    else:
        # Create dataframe for new student
        new_data = pd.DataFrame([{
            "gender": gender,
            "parental level of education": parental_education,
            "lunch": lunch,
            "test preparation course": test_prep,
            "math score": math_score,
            "reading score": reading_score,
            "writing score": writing_score
        }])

        # Encode like training data
        label_cols = ["gender", "parental level of education", "lunch", "test preparation course"]
        le = LabelEncoder()
        for col in label_cols:
            # Fit encoder on training data values
            le.fit(df_clean[col])
            new_data[col] = le.transform(new_data[col])

        # Add average score
        new_data["average_score"] = new_data[["math score", "reading score", "writing score"]].mean(axis=1)

        # Predict
        prediction = model.predict(new_data.drop(columns=["average_score"]))[0]
        confidence = max(model.predict_proba(new_data.drop(columns=["average_score"]))[0])

        # Show result
        result_label = "PASS ✅" if prediction == 1 else "FAIL ❌"
        st.markdown(f"### 🎯 Prediction: **{result_label}**")
        st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")

        # -----------------------------
        # 4️⃣ Improvement Suggestions
        # -----------------------------
        suggestions = []
        if math_score < 60:
            suggestions.append("📌 Focus on improving math fundamentals.")
        if reading_score < 60:
            suggestions.append("📌 Work on reading comprehension skills.")
        if writing_score < 60:
            suggestions.append("📌 Practice essay writing and grammar.")
        if test_prep == "none":
            suggestions.append("📌 Consider completing a test preparation course.")

        if suggestions:
            st.subheader("💡 Suggestions for Improvement:")
            for s in suggestions:
                st.write(s)
        else:
            st.success("🎉 Great! The student is performing well in all areas.")
