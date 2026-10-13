
from pathlib import Path
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "credit_default.db"
REPORTS = ROOT / "reports"

def main():
    files = [
        ("ml", REPORTS / "ml_metrics.json"),
        ("deep_learning", REPORTS / "dl_metrics.json"),
    ]

    with sqlite3.connect(DB) as con:
        for family, path in files:
            if not path.exists():
                continue

            results = json.loads(path.read_text())

            for name, metrics in results.items():
                con.execute(
                    """
                    INSERT INTO model_runs (
                        model_name, model_version,
                        test_roc_auc, test_pr_auc,
                        test_precision, test_recall, notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        "v1",
                        metrics["roc_auc"],
                        metrics["pr_auc"],
                        metrics["precision"],
                        metrics["recall"],
                        f"Model family: {family}; "
                        "random stratified holdout; threshold=0.5",
                    ),
                )

    print("Model metrics recorded in SQLite.")

if __name__ == "__main__":
    main()
