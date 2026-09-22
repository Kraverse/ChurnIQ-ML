from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Churn"


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and fail early when it is empty or unreadable."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("Dataset is empty")
    return frame


def validate_target(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> None:
    """Validate that the expected binary target exists."""
    if target not in frame.columns:
        raise ValueError(f"Missing target column: {target}")
    if frame[target].isna().any():
        raise ValueError(f"Target column contains missing values: {target}")
    if frame[target].nunique() != 2:
        raise ValueError("This classifier expects exactly two target classes")
