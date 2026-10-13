
from fastapi.testclient import TestClient
from credit_default.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    assert "random_forest" in response.json()["available_models"]

def test_missing_features_are_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422
