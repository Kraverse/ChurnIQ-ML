from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from churniq.data import validate_target
from churniq.pipeline import prepare_frame, build_models, save_pipeline

DATASET = Path("data/raw/Telco-Customer-Churn.csv")
METRICS = Path("reports/metrics.json")
MODEL_DIR = Path("models")


def main() -> None:
    df = pd.read_csv(DATASET)
    validate_target(df)
    X, y = prepare_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict] = {}
    best_name = None
    best_score = -1.0

    for name, model in build_models(X_train).items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        results[name] = metrics
        if metrics["cv_roc_auc_mean"] > best_score:
            best_score = metrics["cv_roc_auc_mean"]
            best_name = name
            save_pipeline(model, MODEL_DIR / "churn_model.joblib")

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({"dataset_rows": len(df), "models": results, "selected_model": best_name}, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Selected model: {best_name}")


if __name__ == "__main__":
    main()
