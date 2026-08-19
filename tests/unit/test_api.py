"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from physics_reasoning.api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestAPI:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_list_equations(self, client):
        resp = client.get("/equations")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 20
        eq_ids = [eq["id"] for eq in data]
        assert "newton2" in eq_ids
