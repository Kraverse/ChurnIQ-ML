from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ChurnIQ-ML", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.title("ChurnIQ-ML")
st.caption("Customer churn prediction and retention intelligence")

if not MODEL_PATH.exists():
    st.warning("Model is not available. Run the training pipeline first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Customer profile")
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    monthly_charges = st.number_input("Monthly charges", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total charges", 0.0, 10000.0, 840.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"], key="paperless_2")
    submitted = st.button("Predict churn", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }])
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= 0.5)
    st.subheader("Prediction")
    if prediction:
        st.error(f"Higher churn risk — probability: {probability:.1%}")
        st.write("Consider retention outreach and reviewing contract, billing, and service experience.")
    else:
        st.success(f"Lower churn risk — probability: {probability:.1%}")

st.divider()

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.subheader("Model evaluation")
    st.write(f"Selected model: **{metrics.get('selected_model', 'N/A')}**")
    rows = []
    for name, values in metrics.get("models", {}).items():
        rows.append({"Model": name, **values})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Evaluation metrics will appear after the training pipeline generates reports/metrics.json.")
