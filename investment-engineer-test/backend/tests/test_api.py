import pytest
from fastapi.testclient import TestClient
from backend.app import app
import json

client = TestClient(app)

def test_health_check():
    """health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["data_loaded"] is True

def test_get_assets():
    """test fetching asset data."""
    response = client.get("/assets")
    assert response.status_code == 200
    data = response.json()
    assert "assets" in data
    assert "categories" in data
    assert len(data["assets"]) > 0

def test_get_asset_metrics():
    """per-asset metrics endpoint."""
    response = client.get("/assets/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert isinstance(data["metrics"], list)
    assert len(data["metrics"]) > 0
    # Check a sample metric
    assert "annualised_return" in data["metrics"][0]
    assert "annualised_volatility" in data["metrics"][0]

def test_get_asset_correlation():
    """test the correlation matrix endpoint."""
    response = client.get("/assets/correlation")
    assert response.status_code == 200
    data = response.json()
    assert "correlation" in data
    assert isinstance(data["correlation"], dict)
    assert len(data["correlation"]) > 0
    # Check for NaN/inf values which would cause a 500 error
    assert "null" not in json.dumps(data)
    assert "NaN" not in json.dumps(data)

@pytest.mark.parametrize("endpoint, payload, expected_key", [
    ("/forecast_statistic/annualised_return", {"weights": {"US Equity": 1.0}}, "annualised_return"),
    ("/forecast_statistic/annualised_volatility", {"weights": {"US Equity": 1.0}}, "annualised_volatility"),
    ("/forecast_statistic/sharpe_ratio", {"weights": {"US Equity": 1.0}, "risk_free": 0.02}, "sharpe_ratio"),
    ("/forecast_statistic/downside_deviation", {"weights": {"US Equity": 1.0}, "mar": 0.0}, "downside_deviation"),
    ("/forecast_statistic/value_at_risk", {"weights": {"US Equity": 1.0}, "alpha": 0.05}, "value_at_risk"),
    ("/forecast_statistic/conditional_value_at_risk", {"weights": {"US Equity": 1.0}, "alpha": 0.05}, "conditional_value_at_risk"),
    ("/forecast_statistic/maximum_drawdown", {"weights": {"US Equity": 1.0}}, "maximum_drawdown"),
])
def test_post_statistics(endpoint, payload, expected_key):
    """Test various POST endpoints with valid data."""
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert expected_key in data
    assert isinstance(data[expected_key], float)

def test_post_projection():
    """Test the projection endpoint."""
    payload = {"weights": {"US Equity": 1.0}, "start_value": 10000}
    response = client.post("/projection/percentiles", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "years" in data
    assert "percentiles" in data
    assert isinstance(data["years"], list)
    assert isinstance(data["percentiles"], dict)
    assert "0.5" in data["percentiles"]

## Edge Cases and Error Handling Tests

def test_post_with_invalid_weights():
    """Test endpoint with a non-existent asset."""
    payload = {"weights": {"Invalid Asset Name": 1.0}}
    response = client.post("/forecast_statistic/annualised_return", json=payload)
    assert response.status_code == 400
    assert "Unknown assets in weights" in response.json()["detail"]

def test_post_with_zero_sum_weights():
    """Test endpoint with weights that sum to zero."""
    payload = {"weights": {"Cash": 0, "US Equity": 0}}
    response = client.post("/forecast_statistic/annualised_return", json=payload)
    assert response.status_code == 400
    assert "Sum of provided weights must be > 0" in response.json()["detail"]

def test_post_without_normalise_and_invalid_sum():
    """Test normalise=False with weights not summing to 1."""
    payload = {"weights": {"US Equity": 0.5, "Cash": 0.6}, "normalise": False}
    response = client.post("/forecast_statistic/annualised_return", json=payload)
    assert response.status_code == 400
    assert "Sum of weights must be 1.0" in response.json()["detail"]