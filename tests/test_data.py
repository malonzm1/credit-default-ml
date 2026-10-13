
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

def test_prepared_data_has_expected_columns():
    train = pd.read_csv(
        ROOT / "data" / "processed" / "train.csv"
    )
    test = pd.read_csv(
        ROOT / "data" / "processed" / "test.csv"
    )

    expected = FEATURES + ["default_next_month"]

    assert set(train.columns) == set(expected)
    assert set(test.columns) == set(expected)
    assert len(train) > len(test) > 0

def test_target_is_binary():
    train = pd.read_csv(
        ROOT / "data" / "processed" / "train.csv"
    )
    assert set(train["default_next_month"].unique()) == {0, 1}
