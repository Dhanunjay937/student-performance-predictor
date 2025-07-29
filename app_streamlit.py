# app_streamlit.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessing import run_preprocessing


st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Upload a CSV file to analyze student scores and predict performance.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and preprocess
        df_clean = run_preprocessing(uploaded_file)

        st.success("✅ Data loaded and cleaned successfully.")

        # Preview data
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df_clean.head())

        # Dataset statistics
        st.subheader("📈 Dataset Summary")
        st.write(df_clean.describe())

        # Score distribution
        st.subheader("🎯 Average Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_clean['average_score'], kde=True, ax=ax)
        st.pyplot(fig)

        # Try your own scores
        st.subheader("📝 Try Your Own Scores")
        math = st.slider("Math score", 0, 100, 70)
        reading = st.slider("Reading score", 0, 100, 70)
        writing = st.slider("Writing score", 0, 100, 70)

        avg_score = (math + reading + writing) / 3
        st.success(f"📘 Predicted Average Score: **{avg_score:.2f}**")

        # Download cleaned data
        st.download_button(
            label="💾 Download Cleaned Data",
            data=df_clean.to_csv(index=False),
            file_name="cleaned_student_data.csv",
            mime="text/csv"
        )

        st.info("🔍 Tip: Use the sliders above to simulate new student scores.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.warning("📂 Please upload a CSV file to get started.")
