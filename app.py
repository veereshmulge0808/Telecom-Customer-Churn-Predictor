import streamlit as st
import pandas as pd
import joblib

st.title("Telecom Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood of churn.")

# 1. Load the saved assets
model = joblib.load('churn_model.pkl')
scaler = joblib.load('churn_scaler.pkl')
expected_columns = joblib.load('model_columns.pkl')

# 2. Get user inputs from the sidebar
st.sidebar.header("Customer Attributes")
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = tenure * monthly_charges  # Simple estimation for the app
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

if st.button("Predict Risk"):
    # 3. Create a dictionary with all expected columns set to 0 (default state)
    input_dict = {col: 0 for col in expected_columns}
    
    # 4. Update the numerical values
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    
    # 5. Update the categorical values (mimicking One-Hot Encoding)
    if f"Contract_{contract}" in input_dict:
        input_dict[f"Contract_{contract}"] = 1
    if f"InternetService_{internet}" in input_dict:
        input_dict[f"InternetService_{internet}"] = 1
        
    # 6. Convert to a single-row DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 7. Scale numerical features just like we did in training
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    # 8. Make the prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # 9. Display the results
    st.subheader("Prediction Results")
    if probability >= 0.35: # Using the optimized threshold we established
        st.error(f"High Risk of Churn! (Probability: {probability:.1%})")
        st.write("Recommendation: Offer a discount or migrate to a longer-term contract.")
    else:
        st.success(f"Customer is likely to stay. (Probability: {probability:.1%})")
