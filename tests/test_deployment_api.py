"""Tests for FastAPI Deployment REST API."""

import pytest
from fastapi.testclient import TestClient
from autoopt.models import get_adapter
from autoopt.deployment import create_inference_app
import examples.sklearn_random_forest as ex_sklearn


def test_deployment_api_endpoints():
    model = ex_sklearn.get_model(n_estimators=10)
    sample = ex_sklearn.get_sample_input(1)
    adapter = get_adapter(model, sample_input=sample)

    config = {
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
        "workers": 1,
        "compile_graph": False
    }

    app = create_inference_app(adapter, config, enable_dynamic_batching=False)

    with TestClient(app) as client:
        # Health check
        res_h = client.get("/health")
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "healthy"

        # Config check
        res_c = client.get("/config")
        assert res_c.status_code == 200
        assert "deployment_config" in res_c.json()

        # Predict single
        res_p = client.post("/predict", json={"inputs": sample.tolist()})
        assert res_p.status_code == 200
        assert "predictions" in res_p.json()

        # Metrics
        res_m = client.get("/metrics")
        assert res_m.status_code == 200
        assert res_m.json()["service"]["total_requests_processed"] >= 1
