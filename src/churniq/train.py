from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]


def build_pipeline(model, X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_models(df: pd.DataFrame) -> dict:
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])
    y = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_name, best_cv = None, -1
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator, X_train)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "cv_roc_auc_mean": float(scores.mean()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        }
        results[name] = metrics
        if metrics["cv_roc_auc_mean"] > best_cv:
            best_name, best_cv = name, metrics["cv_roc_auc_mean"]
    return {"results": results, "best_model": best_name, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "candidates": candidates}


def save_best(result: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = result["best_model"]
    pipe = build_pipeline(result["candidates"][name], result["X_train"])
    pipe.fit(result["X_train"], result["y_train"])
    path = output / "churn_model.joblib"
    joblib.dump(pipe, path)
    return path
