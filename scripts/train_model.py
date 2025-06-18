# Import necessary libraries
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Create models directory if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Load dataset with error handling
try:
    data_path = Path("../data/health_data.csv")
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path.absolute()}")
        print("Current directory:", os.getcwd())
        print("Available files in ../data:", os.listdir("../data"))
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded Successfully")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    print("Current directory:", os.getcwd())
    sys.exit(1)

# Initialize LabelEncoders for categorical columns
encoders = {
    col: LabelEncoder().fit(df[col])
    for col in ['Gender', 'Smoking', 'FamilyHistory', 'Disease']
}

# Encode categorical columns in the dataset
encoded_df = df.copy()
for col, encoder in encoders.items():
    encoded_df[col] = encoder.transform(df[col])

# Split data into features and target
X = encoded_df.drop(columns=['Disease'])
y = encoded_df['Disease']

# Split into train and test for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train/Test Split Complete")

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Decode the predicted and actual labels
decoded_y_test = encoders['Disease'].inverse_transform(y_test)
decoded_y_pred = encoders['Disease'].inverse_transform(y_pred)

# Calculate and print the accuracy
accuracy = accuracy_score(decoded_y_test, decoded_y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Print the confusion matrix
print(confusion_matrix(decoded_y_test, decoded_y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(decoded_y_test, decoded_y_pred))

# Function to predict disease based on input patient data
def predict_disease(input_data: dict) -> str:
    input_df = pd.DataFrame([input_data])
    for col in ['Gender', 'Smoking', 'FamilyHistory']:
        input_df[col] = encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)[0]
    return encoders['Disease'].inverse_transform([prediction])[0]

# Save model and encoders with error handling
try:
    with open("../models/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("../models/label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print("Model and encoders saved successfully to 'models' folder")
except Exception as e:
    print(f"Error saving model and encoders: {str(e)}")
    sys.exit(1)