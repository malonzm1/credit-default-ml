
# Credit Default Prediction: End-to-End ML System

## Overview

An end-to-end binary classification project that estimates
the probability of next-month credit-card default using a
public research dataset from the UCI Machine Learning
Repository.

The project includes API-based data retrieval, SQLite,
exploratory data analysis, classical machine learning,
TensorFlow/Keras, FastAPI, Streamlit, and automated tests.

This is an educational research project, not a production
credit-scoring system or a lending recommendation tool.

## Problem Statement

Can borrower characteristics, repayment status, historical
bill amounts, and payment amounts help predict next-month
default?

## Dataset

Source: UCI Machine Learning Repository
Dataset: Default of Credit Card Clients
https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

The dataset contains 30,000 historical records.

## Architecture

UCI data source -> Python ingestion -> SQLite
-> preprocessing -> ML and deep learning
-> FastAPI -> Streamlit

## Models

- Logistic regression
- Random forest
- TensorFlow/Keras neural network

## Evaluation

Report ROC-AUC, average precision (PR-AUC),
precision, recall, F1, confusion matrix, and calibration.

Include a comparison table with the actual test results.
Do not invent or assume model performance.

## Leakage Prevention

- Reserve a stratified test set before training.
- Fit preprocessing on training data only.
- Use validation data for neural-network early stopping.
- Do not tune against the final test set.

## Run Locally

1. Create a Python virtual environment.
2. Install requirements.txt.
3. Run the ingestion script.
4. Run the preparation script.
5. Train the classical ML models.
6. Train the neural network.
7. Start the FastAPI service.
8. Start the Streamlit dashboard.
9. Run pytest.

## Limitations

The dataset is historical, geographically specific,
and does not represent all borrowers or current lending
conditions. Predictions should not be interpreted as
validated real-world default probabilities.

## Reproducibility

Document Python and package versions, random seeds,
data source, model settings, and evaluation procedure.

## License and Attribution

Credit the UCI dataset authors and follow the dataset's
CC BY 4.0 license. Include appropriate attribution.
