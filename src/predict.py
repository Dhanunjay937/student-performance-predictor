# src/predict.py

import pandas as pd

# src/predict.py

def run_prediction(model, df, new_data_dict):
    # Convert to DataFrame
    new_data = pd.DataFrame([new_data_dict])

    # Drop target column to get feature columns only
    feature_columns = df.drop(columns=['average_score'])

    # Combine to ensure matching one-hot encoding
    combined = pd.concat([feature_columns, new_data], axis=0)

    # One-hot encode
    combined_encoded = pd.get_dummies(combined)

    # Align with model's expected input
    combined_encoded = combined_encoded.reindex(columns=feature_columns.columns, fill_value=0)

    # Get only the last row (new data)
    new_data_encoded = combined_encoded.tail(1)

    # Predict
    prediction = model.predict(new_data_encoded)[0]
    result = 'Pass' if prediction == 1 else 'Fail'
    return result
