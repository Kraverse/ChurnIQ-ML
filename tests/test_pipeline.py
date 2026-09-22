import pandas as pd

from churniq.pipeline import prepare_frame, build_models


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["a", "b", "c", "d"],
        "tenure": [1, 20, 5, 40],
        "MonthlyCharges": [70.0, 50.0, 80.0, 40.0],
        "TotalCharges": ["70", "1000", "400", "1600"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "Churn": ["Yes", "No", "Yes", "No"],
    })


def test_prepare_frame_encodes_target_and_drops_identifier():
    X, y = prepare_frame(sample_frame())
    assert "customerID" not in X.columns
    assert y.tolist() == [1, 0, 1, 0]
    assert X["TotalCharges"].dtype.kind in "fi"


def test_models_have_expected_classifiers():
    models = build_models(sample_frame().drop(columns="Churn"))
    assert set(models) == {"logistic_regression", "random_forest"}
