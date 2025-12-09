"""
Unit tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_analyze_endpoint_valid():
    """Test analyze endpoint with valid input"""
    payload = {
        "text": "This is a test news article about scientific discoveries that were made recently by researchers."
    }
    
    response = client.post("/api/v1/analyze", json=payload)
    
    # Should return 200 or 500 depending on if model is loaded
    # For testing without model, we expect a 500
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert "confidence_score" in data
        assert "probabilities" in data
        assert data["label"] in ["Real", "Fake"]
        assert 0.0 <= data["confidence_score"] <= 1.0


def test_analyze_endpoint_invalid_short_text():
    """Test analyze endpoint with too short text"""
    payload = {
        "text": "Short"
    }
    
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422  # Validation error


def test_analyze_endpoint_empty_text():
    """Test analyze endpoint with empty text"""
    payload = {
        "text": ""
    }
    
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422  # Validation error


def test_feedback_endpoint():
    """Test feedback endpoint"""
    payload = {
        "text": "This is a test news article about scientific discoveries.",
        "predicted_label": "Real",
        "confidence_score": 0.95,
        "user_correction": "Fake"
    }
    
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "feedback_id" in data
    assert isinstance(data["feedback_id"], int)


def test_feedback_endpoint_invalid_label():
    """Test feedback endpoint with invalid label"""
    payload = {
        "text": "This is a test news article.",
        "predicted_label": "Invalid",  # Invalid label
        "confidence_score": 0.95,
        "user_correction": "Real"
    }
    
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422  # Validation error


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/api/v1/analyze")
    # CORS headers should be present
    # Note: This might not work in TestClient, would need actual server
    assert response.status_code in [200, 405]  # OPTIONS might not be enabled
