from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import streamlit as st

from churniq.pipeline import build_models, prepare_frame

st.set_page_config(page_title="ChurnIQ-ML", page_icon="📊", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
MODEL_NAME = "logistic_regression"


@st.cache_resource
def load_model():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    X, y = prepare_frame(df)
    model = build_models(X)[MODEL_NAME]
    model.fit(X, y)
    return model


st.title("ChurnIQ-ML")
st.caption("Customer churn prediction and retention intelligence")

with st.spinner("Preparing the churn prediction model..."):
    model = load_model()

st.success("Model ready")

st.subheader("Customer prediction")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col2:
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    senior_citizen = st.selectbox("Senior citizen", [0, 1])
with col3:
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

if st.button("Predict churn", type="primary"):
    row = {
        "gender": "Male",
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No internet service" if internet_service == "No" else "No",
        "OnlineBackup": "No internet service" if internet_service == "No" else "No",
        "DeviceProtection": "No internet service" if internet_service == "No" else "No",
        "TechSupport": "No internet service" if internet_service == "No" else "No",
        "StreamingTV": "No internet service" if internet_service == "No" else "No",
        "StreamingMovies": "No internet service" if internet_service == "No" else "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure,
    }

    frame = pd.DataFrame([row])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])

    st.divider()
    st.subheader("Prediction")
    st.metric("Churn probability", f"{probability:.1%}")
    if prediction:
        st.warning("Higher churn risk detected")
    else:
        st.success("Lower churn risk detected")

st.divider()
st.caption("Model: Logistic Regression | Training data: IBM Telco Customer Churn dataset")
