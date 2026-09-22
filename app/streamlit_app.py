import streamlit as st

st.set_page_config(page_title="ChurnIQ-ML", page_icon="📊", layout="wide")
st.title("ChurnIQ-ML")
st.caption("Customer churn prediction and retention intelligence")

st.info("The interactive prediction form will be connected to the trained inference pipeline after model training is executed.")

st.subheader("Project")
st.write("End-to-end supervised machine learning workflow covering preprocessing, model comparison, evaluation, and API inference.")

st.subheader("Model metrics")
st.write("Metrics are intentionally loaded from generated experiment results rather than hard-coded, so the dashboard never presents fabricated performance numbers.")
