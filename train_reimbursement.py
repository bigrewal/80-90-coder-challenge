import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json
import sys
import pickle

# Load and prepare data from public_cases.json
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    X = []
    y = []
    for case in data:
        inputs = case['input']
        days = inputs['trip_duration_days']
        miles = inputs['miles_traveled']
        receipts = inputs['total_receipts_amount']
        
        X.append([days, miles, receipts])
        y.append(case['expected_output'])
    return np.array(X), np.array(y)


# Train the model and evaluate
def train_model(X, y):
    # Split data: 80% train, 20% test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=800, random_state=42, max_features=2)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error on Test Set: ${mae:.2f}")
    
    # Feature importance
    features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    importance = model.feature_importances_
    for feat, imp in zip(features, importance):
        print(f"Feature: {feat}, Importance: {imp:.4f}")
    
    # Save the trained model
    with open('reimbursement_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'reimbursement_model.pkl'")
    
    return model

# Load the trained model
def load_model():
    try:
        with open('reimbursement_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Error: 'reimbursement_model.pkl' not found. Run in training mode first.")
        sys.exit(1)

# Predict reimbursement for new inputs
def predict_reimbursement(model, days, miles, receipts):
    input_data = np.array([[float(days), float(miles), float(receipts)]])
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Main execution
if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) == 1:
        # Training mode: Load data, train, and save model
        try:
            X, y = load_data('public_cases.json')
            model = train_model(X, y)
            print("Model trained and saved successfully!")
        except FileNotFoundError:
            print("Error: 'public_cases.json' not found in the current directory.")
            sys.exit(1)
    elif len(sys.argv) == 4:
        # Prediction mode: Load model and predict
        try:
            model = load_model()
            days, miles, receipts = sys.argv[1], sys.argv[2], sys.argv[3]
            result = predict_reimbursement(model, days, miles, receipts)
            print(result)
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)
    else:
        print("Usage:")
        print("  Train: python3 train_and_predict_reimbursement.py")
        print("  Predict: python3 train_and_predict_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)