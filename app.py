from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app) # Allows your frontend to communicate with this backend

# Load assets
model = joblib.load('churn_model.pkl')
scaler = joblib.load('churn_scaler.pkl')
expected_columns = joblib.load('model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json # Receive data from your HTML frontend

    # 1. Create default dictionary
    input_dict = {col: 0 for col in expected_columns}

    # 2. Update numerical values
    input_dict['tenure'] = float(data['tenure'])
    input_dict['MonthlyCharges'] = float(data['monthly_charges'])
    input_dict['TotalCharges'] = input_dict['tenure'] * input_dict['MonthlyCharges']

    # 3. Update categorical values
    contract = data['contract']
    internet = data['internet']
    if f"Contract_{contract}" in input_dict:
        input_dict[f"Contract_{contract}"] = 1
    if f"InternetService_{internet}" in input_dict:
        input_dict[f"InternetService_{internet}"] = 1

    # 4. Format, Scale, and Predict
    input_df = pd.DataFrame([input_dict])
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    probability = model.predict_proba(input_df)[0][1]

    # Send result back to the HTML frontend
    return jsonify({
        'probability': probability,
        'is_high_risk': bool(probability >= 0.35)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
