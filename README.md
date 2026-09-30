# Telecom Customer Churn Prediction 📉

## Business Overview
Customer attrition (churn) is one of the highest cost-drivers in the telecommunications industry. This project implements an end-to-end Machine Learning pipeline to identify at-risk customers before they cancel their service. By optimizing a predictive algorithm for **Recall**, this solution allows business stakeholders to proactively deploy targeted retention strategies.

**Live Web Application:** https://telecom-customer-churn-predictor-jansx52tee8lbmciprygiw.streamlit.app/

---

## Key Business Insights
Through Exploratory Data Analysis (EDA) and Random Forest feature importance ranking, three critical risk factors were identified:
*   **The First-Year Cliff:** Customers in their first 12 months have a staggering **47.6% churn rate**. Retention stabilizes significantly, dropping to just 6.6% by year five.
*   **Fiber Optic Unreliability:** Despite being a premium tier, Fiber Optic users churn at **41.8%** (compared to 18.9% for standard DSL). This indicates a severe mismatch in pricing-to-value or infrastructure reliability.
*   **Manual Payment Friction:** Customers paying via electronic checks churn at **45.2%**, significantly higher than automated methods like credit cards (15-16%). Manual payments serve as a recurring friction point.

## Technical Pipeline
This project was structured into four distinct engineering phases:
1.  **Data Integration & Cleaning:** Merged disjointed datasets, handled hidden missing values within numerical columns, and standardized data types.
2.  **Feature Engineering:** Applied One-Hot Encoding to categorical variables (using `drop_first=True` to prevent multicollinearity) and normalized numerical features using `MinMaxScaler`.
3.  **Predictive Modeling:** Trained a **Random Forest Classifier**. Applied `class_weight='balanced'` to mathematically penalize the algorithm for missing the minority class (churners).
4.  **Strategic Optimization:** Lowered the decision threshold from 0.50 to 0.35 to prioritize **Recall** over raw precision, aligning the algorithm's mathematical output with real-world financial priorities.

## Model Performance
In churn prediction, missing a churner costs lifetime value; falsely predicting a churn merely costs a retention discount. The model was aggressively tuned to reflect this.
*   **Baseline Recall (Default 0.50 Threshold):** 49%
*   **Optimized Recall (Custom 0.35 Threshold):** **68%**
*   **Overall Accuracy:** 77%
*   **Result:** The optimized model successfully captures the vast majority of departing customers while maintaining strong overall accuracy across the user base.
