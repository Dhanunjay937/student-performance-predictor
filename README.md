🎓 Student Performance Predictor
An AI-powered web application built with Python, Streamlit, and Scikit-learn to predict student performance and provide subject-wise improvement suggestions.

📌 Features

1. CSV Upload & Preprocessing
Upload your student dataset in CSV format.

Cleans data automatically (handles missing values, encodes categorical data).

Calculates Average Score.

2. Performance Prediction
Predicts Pass / Fail based on average score.

Uses Random Forest Classifier for high accuracy.

Displays model accuracy after training.

3. Subject-Wise Analysis
Shows where students need improvement in:

📘 Math

📖 Reading

✍️ Writing

Gives actionable suggestions for improvement.

4. Visualization
Pie chart showing pass vs fail distribution.

Bar chart showing subject-wise average performance.

5. Live Input Prediction
Enter details for a new student:

Gender

Parental education level

Lunch type

Test preparation course

Math score

Reading score

Writing score

Attendance %

Study time per week

Extra classes (yes/no)

Parental involvement score

Instantly get prediction & improvement tips.

🛠️ Tech Stack
Frontend: Streamlit

Backend: Python

ML Model: Scikit-learn (Random Forest)

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Model Persistence: Joblib

📂 Project Structure
bash
Copy
Edit
student-performance-predictor/
│
├── app_streamlit.py       # Main Streamlit app
├── app.py                 # CLI version
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
│
├── src/                   # Source code
│   ├── preprocessing.py   # Data cleaning & encoding
│   ├── train_model.py     # Model training & saving
│   ├── predict.py         # Prediction logic
│   ├── visualize.py       # Visualization functions
│
├── data/                  # Sample datasets
│   └── StudentsPerformance.csv
│
└── model.joblib           # Saved ML model
🚀 Installation & Usage
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
2. Create a virtual environment
bash
Copy
Edit
python -m venv venv
3. Activate the environment
Windows (PowerShell)

bash
Copy
Edit
venv\Scripts\activate
Mac/Linux

bash
Copy
Edit
source venv/bin/activate
4. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
5. Run the Streamlit app
bash
Copy
Edit
streamlit run app_streamlit.py
📊 Example Output
Pie Chart – Pass vs Fail

csharp
Copy
Edit
[Pie chart showing distribution]
Bar Chart – Subject-Wise Performance

csharp
Copy
Edit
[Bar chart showing scores for Math, Reading, Writing]
Sample Prediction Output

vbnet
Copy
Edit
Prediction: Pass ✅
Accuracy: 98%
Improvement Areas: Reading, Writing
Suggestions: Increase daily reading practice, join writing workshops.
📈 Future Improvements
🔹 Support for more ML algorithms.

🔹 Export improvement report as PDF.

🔹 Integration with Google Sheets for live data updates.

🔹 User login system for personalized analysis.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss your idea.