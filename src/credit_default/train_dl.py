
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.credit_default.train_ml import (
    ROOT, DATA, MODELS, REPORTS,
    TARGET, NUMERIC, CATEGORICAL,
    make_preprocessor,
)

def main():
    tf.keras.utils.set_random_seed(42)

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    features = NUMERIC + CATEGORICAL

    X_train = train[features]
    y_train = train[TARGET].astype(int).to_numpy()

    X_test = test[features]
    y_test = test[TARGET].astype(int).to_numpy()

    # Fit preprocessing on training data only.
    preprocessor = make_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    X_train_t = np.asarray(X_train_t, dtype="float32")
    X_test_t = np.asarray(X_test_t, dtype="float32")

    joblib.dump(
        preprocessor,
        MODELS / "dl_preprocessor.joblib",
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(X_train_t.shape[1],)
        ),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                name="pr_auc", curve="PR"
            ),
        ],
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = dict(zip(classes, weights))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train_t,
        y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2,
    )

    probabilities = model.predict(
        X_test_t, batch_size=256, verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    model.save(MODELS / "neural_network.keras")

    with open(REPORTS / "dl_metrics.json", "w") as f:
        json.dump({"neural_network": results}, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
