import pandas as pd
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(file):
    df = pd.read_csv(file)

    # Drop missing values
    df = df.dropna()

    # Encode categorical variables
    label_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # Add a new column for average score
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

    return df

# Debug print
if __name__ == "__main__":
    print("✅ Preprocessing module loaded successfully.")
    print("Available functions:", dir())
