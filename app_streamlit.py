import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="wide")

# ------------------------
# 📌 File Upload
# ------------------------
st.title("🎓 Student Performance Predictor")
uploaded_file = st.file_uploader("📂 Upload Student Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data Loaded Successfully!")
    st.dataframe(df.head())

    # ------------------------
    # 📌 Preprocessing
    # ------------------------
    st.subheader("🔄 Data Preprocessing")

    df = df.dropna()

    # Encode categorical variables
    label_cols = df.select_dtypes(include='object').columns
    le_dict = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # Save encoder for later predictions

    # Add average score column
    if all(col in df.columns for col in ['math score', 'reading score', 'writing score']):
        df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    else:
        st.error("❌ Dataset must contain 'math score', 'reading score', and 'writing score' columns.")
        st.stop()

    st.write("✅ Data Cleaned & Encoded Successfully!")
    st.dataframe(df.head())

    # ------------------------
    # 📌 Model Training
    # ------------------------
    st.subheader("🤖 Model Training")
    X = df.drop(columns=['average_score'])
    y = df['average_score'].apply(lambda x: 1 if x >= 60 else 0)  # 1 = Pass, 0 = Fail

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model trained successfully! Accuracy: **{accuracy:.2f}**")
    joblib.dump(model, "model.joblib")

    # ------------------------
    # 📊 Pie Chart
    # ------------------------
    st.subheader("📊 Student Performance Distribution")
    performance_counts = y.value_counts()
    labels = ['Good Performance', 'Needs Improvement']
    sizes = [
        performance_counts.get(1, 0),
        performance_counts.get(0, 0)
    ]
    colors = ['#4CAF50', '#FF6347']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # ------------------------
    # 📌 Prediction for New Student
    # ------------------------
    st.subheader("📝 Predict New Student Performance")

    new_student = {}
    for col in X.columns:
        if col in le_dict:  # categorical input
            options = list(le_dict[col].classes_)
            choice = st.selectbox(f"{col}", options)
            new_student[col] = le_dict[col].transform([choice])[0]
        else:  # numerical input
            new_student[col] = st.number_input(f"{col}", min_value=0.0, max_value=100.0, value=50.0)

    if st.button("🔍 Predict Performance"):
        new_df = pd.DataFrame([new_student])
        prediction = model.predict(new_df)[0]
        if prediction == 1:
            st.success("🎉 Predicted: **Good Performance** ✅")
        else:
            st.error("⚠️ Predicted: **Needs Improvement** ❌")
