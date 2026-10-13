
from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "credit_default.db"
OUT = ROOT / "data" / "processed"

TARGET = "default_next_month"

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT * FROM credit_records", con
        )

    # The row identifier is not a predictive feature.
    df = df.drop(columns=["record_id"])

    # Verify the target before splitting.
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    train_df.to_csv(OUT / "train.csv", index=False)
    test_df.to_csv(OUT / "test.csv", index=False)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("\nTraining target distribution:")
    print(train_df[TARGET].value_counts(normalize=True))
    print("\nTest target distribution:")
    print(test_df[TARGET].value_counts(normalize=True))

if __name__ == "__main__":
    main()
