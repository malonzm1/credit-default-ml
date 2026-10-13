
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    training_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    test_roc_auc REAL,
    test_pr_auc REAL,
    test_precision REAL,
    test_recall REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    default_probability REAL NOT NULL
        CHECK(default_probability >= 0
          AND default_probability <= 1),
    predicted_default INTEGER NOT NULL
        CHECK(predicted_default IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prediction_created_at
ON prediction_log(created_at);

CREATE INDEX IF NOT EXISTS idx_prediction_model
ON prediction_log(model_name, model_version);


