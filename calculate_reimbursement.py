#!/usr/bin/env python3
import sys
import pickle
import numpy as np

def load_model():
    try:
        with open('reimbursement_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Error: 'reimbursement_model.pkl' not found. Run in training mode first.")
        sys.exit(1)



def calculate_reimbursement(days, miles, receipts):
    model = load_model()
    input_data = np.array([[float(days), float(miles), float(receipts)]])
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)
    
if __name__ == "__main__":
    # Check if exactly 3 arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>", file=sys.stderr)
        sys.exit(1)
    
    # Get inputs from command line
    days, miles, receipts = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Calculate and print reimbursement
    result = calculate_reimbursement(days, miles, receipts)
    print(result)