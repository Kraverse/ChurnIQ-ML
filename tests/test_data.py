import pandas as pd
import pytest

from churniq.data import validate_target


def test_validate_target_accepts_binary_target():
    validate_target(pd.DataFrame({"Churn": [0, 1, 0, 1]}))


def test_validate_target_rejects_missing_target():
    with pytest.raises(ValueError, match="Missing target"):
        validate_target(pd.DataFrame({"x": [1, 2]}))
