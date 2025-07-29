# app.py

from src import preprocessing, train_model, predict, visualize

# Step 1: Preprocess the data
data_file = 'data/StudentsPerformance.csv'
df_clean = preprocessing.run_preprocessing(data_file)

# Step 2: Train model and get accuracy
model, accuracy, X_test, y_test = train_model.train_and_save_model(df_clean)
print(f"\nModel trained. Accuracy: {accuracy:.2f}")

# Step 3: Make prediction for a new student (change input values as needed)
new_data = {
    'gender': 'female',
    'race/ethnicity': 'group B',
    'parental level of education': "bachelor's degree",
    'lunch': 'standard',
    'test preparation course': 'completed',
    'math score': 65,
    'reading score': 70,
    'writing score': 72
}
predict.run_prediction(model, df_clean, new_data)

# Step 4: Visualize the dataset
visualize.plot_visualizations(df_clean)
