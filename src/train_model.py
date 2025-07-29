# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model(df):
    # Features and label
    X = df.drop(columns=['average_score'])
    y = df['average_score'].apply(lambda x: 1 if x >= 60 else 0)  # 1 = pass, 0 = fail

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(model, 'model.joblib')

    return model, accuracy, X_test, y_test
