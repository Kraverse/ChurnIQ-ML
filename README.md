# ChurnIQ-ML

End-to-end customer churn prediction and retention intelligence project built with Python and scikit-learn.

## What it does

- Downloads and validates the Telco Customer Churn dataset
- Cleans numeric and categorical features with a reproducible preprocessing pipeline
- Compares Logistic Regression and Random Forest models
- Uses stratified 5-fold cross-validation with ROC-AUC for model selection
- Evaluates accuracy, precision, recall, F1 and ROC-AUC on a held-out test set
- Persists the complete preprocessing + model pipeline with Joblib
- Exposes predictions through FastAPI
- Provides an interactive Streamlit prediction dashboard
- Runs automated linting, tests, training and evaluation through GitHub Actions

## Current model result

The current training run selected **Logistic Regression** using mean 5-fold cross-validation ROC-AUC. On the held-out test set, the recorded ROC-AUC is approximately **0.841**. These values are experiment results, not guarantees of future production performance.

## Architecture

```text
Telco dataset
    -> validation
    -> preprocessing
    -> train/test split
    -> stratified 5-fold CV
    -> Logistic Regression + Random Forest
    -> model selection
    -> evaluation
    -> Joblib model
    -> FastAPI /predict
    -> Streamlit dashboard
```

## Run locally

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
python -m pytest -q
```

Run the API:

```bash
uvicorn app.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## API

`GET /health` reports service/model availability.

`POST /predict` accepts customer feature fields and returns a binary churn prediction plus churn probability.

## Testing

The repository includes unit/API tests and GitHub Actions CI. The latest verified CI run passed linting, tests and the ML training/evaluation workflow.

## Project structure

```text
app/                 FastAPI and Streamlit applications
src/churniq/         reusable ML/data pipeline code
scripts/              dataset and training entry points
tests/                automated tests
reports/              generated evaluation metrics
models/               generated model artifacts (ignored by Git)
```

## Limitations

This is a supervised learning project using the public Telco Customer Churn dataset. It is intended for demonstration and learning; it should not be treated as a validated production decision system without additional monitoring, calibration, fairness analysis, data validation and retraining processes.
