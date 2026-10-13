
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

API_URL = "http://127.0.0.1:8000"

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

st.set_page_config(
    page_title="Credit Default ML",
    page_icon="📊",
    layout="wide",
)

st.title("Credit Default Prediction")
st.caption(
    "Research portfolio demonstration — not a lending decision tool."
)

st.header("Model evaluation")

metrics = {}
for filename in ["ml_metrics.json", "dl_metrics.json"]:
    path = REPORTS / filename
    if path.exists():
        metrics.update(pd.read_json(path, typ="series").to_dict())

if metrics:
    st.json(metrics)
else:
    st.info("Train the models to display evaluation metrics.")

st.header("Score a sample customer")

test_path = DATA / "test.csv"
if not test_path.exists():
    st.error("Run the data preparation script first.")
    st.stop()

test_df = pd.read_csv(test_path)
sample = test_df[FEATURES].iloc[0].to_dict()

model_name = st.selectbox(
    "Model",
    ["random_forest", "logistic_regression", "neural_network"],
)

with st.form("prediction_form"):
    values = {}

    for feature in FEATURES:
        default = float(sample[feature])
        values[feature] = st.number_input(
            feature,
            value=default,
            format="%.2f",
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            params={"model_name": model_name},
            json=values,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        st.metric(
            "Estimated default probability",
            f"{result['default_probability']:.1%}",
        )
        st.write(
            "Predicted class at threshold 0.5:",
            result["predicted_default_at_0_5"],
        )
        st.caption(result["notice"])

    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
