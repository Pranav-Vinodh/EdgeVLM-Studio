import pytest
from starlette.testclient import TestClient
from edgevlm.remote.colab_server import create_colab_server, TrainingPayload, InferencePayload


def test_colab_server_health():
    server = create_colab_server(auth_token="secret123")
    client = TestClient(server.app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "gpu" in data


def test_colab_server_auth_rejection():
    server = create_colab_server(auth_token="secret123")
    client = TestClient(server.app)

    # Missing token
    resp_missing = client.post("/train/start", json=TrainingPayload().model_dump())
    assert resp_missing.status_code == 401

    # Invalid token
    headers = {"Authorization": "Bearer wrong_token"}
    resp_invalid = client.post(
        "/train/start", json=TrainingPayload().model_dump(), headers=headers
    )
    assert resp_invalid.status_code == 403


def test_colab_server_training_lifecycle():
    server = create_colab_server(auth_token="secret123")
    client = TestClient(server.app)
    headers = {"Authorization": "Bearer secret123"}

    # Start training
    payload = TrainingPayload(epochs=1)
    resp = client.post("/train/start", json=payload.model_dump(), headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert "job_id" in data

    # Check status
    status_resp = client.get("/train/status", headers=headers)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "is_running" in status_data

    # Stop training
    stop_resp = client.post("/train/stop", headers=headers)
    assert stop_resp.status_code == 200


def test_colab_server_remote_inference():
    server = create_colab_server(auth_token="secret123")
    client = TestClient(server.app)
    headers = {"Authorization": "Bearer secret123"}

    payload = InferencePayload(prompt="Test prompt")
    resp = client.post("/inference", json=payload.model_dump(), headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["detections"]) > 0
