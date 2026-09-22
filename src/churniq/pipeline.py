from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]


def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    y = frame.pop(TARGET).map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Unexpected values in Churn target")
    frame = frame.drop(columns=DROP_COLUMNS, errors="ignore")
    return frame, y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(X)
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(X)),
            ("model", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def save_pipeline(model: Pipeline, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
