"""HTTP layer: routing, validation, error shapes and the served frontend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import config
from tests.conftest import AI_LIKE, HUMAN_LIKE, SHORT_TEXT


@pytest.fixture(scope="module")
def client():
    import os

    os.environ["ASD_PRELOAD_MODELS"] = "false"
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# analyse
# --------------------------------------------------------------------------


def test_analyze_returns_the_full_payload(client):
    response = client.post("/api/analyze", json={
        "text": HUMAN_LIKE, "category": "auto", "analysis_mode": "fast"})
    assert response.status_code == 200
    payload = response.json()
    for key in ("result", "slop", "category", "detectors", "statistics",
                "explanation", "warnings", "meta", "processing_ms",
                "disclaimer"):
        assert key in payload, f"missing {key}"


def test_analyze_validates_the_schema(client):
    from app.schemas.response import AnalyzeResponse

    response = client.post("/api/analyze", json={
        "text": AI_LIKE, "analysis_mode": "fast"})
    AnalyzeResponse.model_validate(response.json())


def test_empty_text_returns_400_with_a_code(client):
    response = client.post("/api/analyze", json={"text": "   "})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "empty_input"
    assert detail["error"]


def test_short_text_returns_200_and_abstains(client):
    response = client.post("/api/analyze", json={"text": SHORT_TEXT})
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["ai_origin_score"] is None
    assert payload["result"]["classification"] == "Insufficient evidence"


def test_missing_text_field_is_a_422(client):
    assert client.post("/api/analyze", json={}).status_code == 422


def test_invalid_mode_is_a_422(client):
    response = client.post("/api/analyze", json={
        "text": HUMAN_LIKE, "analysis_mode": "ludicrous"})
    assert response.status_code == 422


def test_invalid_category_is_a_422(client):
    response = client.post("/api/analyze", json={
        "text": HUMAN_LIKE, "category": "sonnet"})
    assert response.status_code == 422


@pytest.mark.parametrize("category", ["auto", *config.CATEGORIES])
def test_every_declared_category_is_accepted(client, category):
    response = client.post("/api/analyze", json={
        "text": HUMAN_LIKE, "category": category, "analysis_mode": "fast"})
    assert response.status_code == 200


@pytest.mark.parametrize("mode", ["fast", "standard", "deep"])
def test_every_mode_is_accepted(client, mode):
    response = client.post("/api/analyze", json={
        "text": HUMAN_LIKE, "analysis_mode": mode})
    assert response.status_code == 200
    assert response.json()["meta"]["analysis_mode"] == mode


def test_batch_endpoint(client):
    response = client.post("/api/analyze/batch", json={
        "texts": [HUMAN_LIKE, SHORT_TEXT, "   "], "analysis_mode": "fast"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert payload["results"][0]["ok"] is True
    assert payload["results"][2]["ok"] is False
    assert payload["results"][2]["error"]["code"] == "empty_input"


# --------------------------------------------------------------------------
# introspection endpoints
# --------------------------------------------------------------------------


def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["version"] == config.APP_VERSION
    assert isinstance(payload["models_loaded"], dict)
    assert isinstance(payload["warnings"], list)


def test_config_endpoint_exposes_thresholds(client):
    payload = client.get("/api/config").json()
    assert payload["length"]["min_words"] == config.MIN_WORDS
    assert payload["classes"] == list(config.CLASSES)
    assert payload["slop_weights"] == config.SLOP_WEIGHTS
    assert len(payload["display_bands"]) == len(config.DISPLAY_BANDS)
    assert "normalisation" in payload and "calibration" in payload


def test_capabilities_is_an_honest_inventory(client):
    payload = client.get("/api/capabilities").json()
    assert set(payload["engines"]) == {
        "transformer", "probability", "curvature", "binoculars", "stylometry",
        "semantic", "humanization", "slop", "ood"}
    assert payload["engines"]["curvature"]["variant"] == \
        "Fast-DetectGPT analytic white-box"
    assert payload["engines"]["curvature"]["not_implemented"]
    ensemble = payload["ensemble"]
    assert payload["output_is_a_probability"] == (
        ensemble["trained_meta_classifier_installed"]
        and ensemble["calibrator_installed"])
    assert payload["how_to_train"]


def test_startup_warnings_endpoint(client):
    assert isinstance(client.get("/api/startup-warnings").json()["warnings"], list)


# --------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------


def test_frontend_is_served(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "AI Slop Detector" in root.text
    for asset in ("/static/style.css", "/static/script.js"):
        assert client.get(asset).status_code == 200


def test_openapi_document_builds(client):
    schema = client.get("/openapi.json").json()
    assert "/api/analyze" in schema["paths"]
    assert schema["info"]["version"] == config.APP_VERSION
