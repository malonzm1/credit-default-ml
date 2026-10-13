
from pathlib import Path
import sqlite3

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "credit_default.db"

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

def main():
    ROOT.joinpath("data", "raw").mkdir(
        parents=True, exist_ok=True
    )
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Normalize names from the source client.
    X.columns = [str(c).strip().upper() for c in X.columns]

    # Handle a possible generic-column representation.
    generic_names = {
        f"X{i}": name
        for i, name in enumerate(FEATURES, start=1)
    }
    X = X.rename(columns=generic_names)

    missing = sorted(set(FEATURES) - set(X.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Received: {X.columns.tolist()}"
        )

    X = X[FEATURES].copy()
    y = pd.to_numeric(y, errors="raise")

    df = X.copy()
    df["default_next_month"] = y.to_numpy()
    df.insert(0, "record_id", range(1, len(df) + 1))

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs")

    if df["default_next_month"].isna().any():
        raise ValueError("Missing target values")

    if not set(df["default_next_month"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected target values")

    # Keep a local source snapshot for reproducibility.
    df.to_csv(
        ROOT / "data" / "raw" / "credit_records.csv",
        index=False,
    )

    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(
            "credit_records",
            con,
            if_exists="replace",
            index=False,
        )

        con.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                ingestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        con.execute(
            """INSERT INTO ingestion_runs(source, row_count)
               VALUES (?, ?)""",
            ("UCI dataset 350", len(df)),
        )

    print(f"Rows ingested: {len(df):,}")
    print(f"Database: {DB_PATH}")
    print(df.head())
    print(df["default_next_month"].value_counts())

if __name__ == "__main__":
    main()
