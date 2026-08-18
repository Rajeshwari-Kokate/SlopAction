"""
Regression tests for defects found by an adversarial audit of this codebase.

Each test here corresponds to a real bug that shipped in an earlier revision.
They are collected in one file so the reasoning behind them survives; every one
of them would pass trivially on a naive implementation and fails loudly if the
fix is ever reverted.
"""

from __future__ import annotations

import json
import math

import pytest

from app import pipeline
from app.core import config
from app.core.types import DetectorResult
from app.detectors import normalisation
from app.detectors.ood_detector import OODDetector
from app.ensemble import aggregation, meta_classifier
from training.common import build_context
from tests.conftest import AI_LIKE, CODE_TEXT, HUMAN_LIKE, MIXED_TEXT


def _result(name, score=0.5, **kwargs):
    return DetectorResult(name=name, available=True, score=score,
                          features={"probe": 1.0}, **kwargs)


# --------------------------------------------------------------------------
# slop and OOD must not reach the authorship estimate
# --------------------------------------------------------------------------


def test_slop_and_ood_are_absent_from_the_ensemble_feature_vector():
    """Excluding them from the untrained pooling is not sufficient.

    A trained meta-classifier consumes the raw vector, so if slop and OOD
    features are present it will learn to predict authorship from writing
    quality and from "this looks like source code" - reintroducing exactly the
    conflation the three-axis design exists to prevent. They must be filtered
    where the vector is built.
    """
    context = build_context(AI_LIKE)
    results = {name: _result(name) for name in
               ("transformer", "stylometry", "semantic", "slop", "ood")}
    vector = meta_classifier.build_ensemble_vector(context, results)

    assert not [k for k in vector if k.startswith("slop__")]
    assert not [k for k in vector if k.startswith("ood__")]
    # the authorship engines are still there
    assert [k for k in vector if k.startswith("transformer__")]
    assert [k for k in vector if k.startswith("stylometry__")]


def test_end_to_end_response_shows_no_slop_or_ood_contribution():
    result = pipeline.analyse(CODE_TEXT + "\n\n" + AI_LIKE, mode="standard")
    contributions = result["ensemble"]["detail"].get("contributions", [])
    names = {c["detector"] for c in contributions}
    assert not names & set(meta_classifier.EXCLUDED_FROM_AUTHORSHIP)


# --------------------------------------------------------------------------
# statistical OOD must be an upper-tail measure, not a raw percentile
# --------------------------------------------------------------------------


@pytest.mark.parametrize("distance,expected", [
    (0.0, 0.0),      # far below the fitting distribution
    (5.0, 0.0),      # exactly the median of it - must NOT read as 50% OOD
    (7.5, 0.5),      # p75
    (10.0, 1.0),     # the far tail
])
def test_statistical_ood_is_an_upper_tail_measure(distance, expected):
    """A raw training-set percentile gives the median of the fitting corpus an
    OOD score of 0.5 and pushes its top 18% over the hard threshold, making a
    correctly fitted reference worse than having none at all.

    The artefact is written to the real path so this also exercises the
    load-after-startup behaviour.
    """
    import joblib
    import numpy as np

    from app.utils import model_loader

    path = config.ArtifactPaths().ood_reference
    assert not path.exists(), "test would clobber a real artefact"

    # a two-feature reference centred on the origin with identity precision, so
    # the Mahalanobis distance of a point is just its Euclidean norm
    joblib.dump({
        "mean": np.zeros(2),
        "precision": np.eye(2),
        "feature_names": ["probe_a", "probe_b"],
        "distance_quantiles": np.linspace(0.0, 10.0, 101),
        "n_samples": 101,
        "fitted_on": "regression-test",
    }, _ensure_parent(path))

    model_loader.reset_cache()
    try:
        context = build_context(HUMAN_LIKE)
        # feed the vector directly so the distance is exactly what we intend
        context._feature_cache["stylometry_vector"] = {
            "probe_a": distance, "probe_b": 0.0}
        score, detail = OODDetector()._statistical(context)
        assert detail["available"] is True, detail
        assert abs(detail["mahalanobis_distance"] - distance) < 1e-6
        assert abs(score - expected) < 0.02, detail
    finally:
        path.unlink(missing_ok=True)
        model_loader.reset_cache()


def _ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_ood_stays_quiet_on_ordinary_prose():
    for text in (HUMAN_LIKE, AI_LIKE):
        assert OODDetector().analyse(build_context(text)).score < \
            config.OOD_ALERT_THRESHOLD


# --------------------------------------------------------------------------
# artefacts written while the service is running must be picked up
# --------------------------------------------------------------------------


def test_normalisation_artefact_is_picked_up_without_a_restart():
    """A negative-result cache meant a model trained while the server ran was
    never loaded, while /api/capabilities (which stats the filesystem) reported
    it as installed - the two endpoints disagreed about the same file."""
    path = config.MODELS_DIR / "normalisation.json"
    assert not path.exists(), "test would clobber a real artefact"
    assert normalisation.is_fitted() is False

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps({
            "fitted_on": "regression-test",
            "references": {"perplexity": {"midpoint": 25.0, "scale": 10.0,
                                          "direction": -1}},
        }))
        assert normalisation.is_fitted() is True
        assert normalisation.references()["perplexity"].midpoint == 25.0
    finally:
        path.unlink(missing_ok=True)
        normalisation._load.cache_clear()
    assert normalisation.is_fitted() is False


# --------------------------------------------------------------------------
# unit scores must be distinguishable from the document verdict
# --------------------------------------------------------------------------


def test_unit_labels_use_the_coarse_vocabulary():
    """A paragraph must never be labelled 'Strong AI Indicators'. It carries
    nowhere near enough evidence for a six-band verdict, and reusing the
    document vocabulary invites a reader to treat it as one."""
    allowed = {"likely_ai", "uncertain", "likely_human", "unscored"}
    document_bands = {b.label for b in config.DISPLAY_BANDS}

    result = pipeline.analyse(MIXED_TEXT, mode="standard")
    for key in ("chunks", "paragraphs", "sentences"):
        labels = {u["label"] for u in result[key]}
        assert labels <= allowed, f"{key} used {labels - allowed}"
        assert not labels & document_bands


def test_scored_units_declare_their_score_type():
    result = pipeline.analyse(MIXED_TEXT, mode="standard")
    for key in ("chunks", "paragraphs", "sentences"):
        for unit in result[key]:
            if unit["ai_score"] is not None:
                detail = unit.get("detail") or {}
                assert detail.get("score_type") == aggregation.UNIT_SCORE_TYPE


def test_unit_pooling_returns_none_rather_than_a_neutral_placeholder():
    assert aggregation._pool_signals({}) is None
    assert aggregation._pool_signals({"unknown_engine": 0.9}) is None
    pooled = aggregation._pool_signals({"transformer": 1.0, "probability": 0.0})
    assert pooled is not None and 0.0 <= pooled <= 1.0


# --------------------------------------------------------------------------
# the chunk aggregate must not let its source engines vote twice
# --------------------------------------------------------------------------


def test_chunk_aggregate_supersedes_its_sources():
    context = build_context(AI_LIKE)
    results = {
        "transformer": _result("transformer", 1.0, trained=True),
        "curvature": _result("curvature", 1.0),
        "stylometry": _result("stylometry", 0.0),
    }
    outcome = meta_classifier.predict(
        context, results, chunk_signal=1.0,
        chunk_signal_sources=["transformer", "curvature"])
    contributions = {c["detector"]: c for c in outcome.detail["contributions"]}
    assert "chunk_aggregate" in contributions
    assert set(contributions["chunk_aggregate"]["supersedes"]) == \
        {"transformer", "curvature"}
    # the superseded engines must not appear again as their own terms
    assert "transformer" not in contributions
    assert "curvature" not in contributions
    assert "stylometry" in contributions


def test_superseded_engines_still_count_towards_abstention():
    """Folding an engine into the aggregate must not make the system think it
    has less evidence than it does."""
    context = build_context(AI_LIKE)
    results = {
        "transformer": _result("transformer", 0.9, trained=True),
        "curvature": _result("curvature", 0.9),
    }
    outcome = meta_classifier.predict(
        context, results, chunk_signal=0.9,
        chunk_signal_sources=["transformer", "curvature"])
    assert outcome.abstained is False


# --------------------------------------------------------------------------
# numerical safety
# --------------------------------------------------------------------------


def test_squash_survives_pathological_values():
    """A degenerate model can emit a perplexity of 1e6; that must not raise
    OverflowError inside a detector."""
    for value in (-1e30, -1e6, 0.0, 1e6, 1e30):
        squashed = normalisation.squash("perplexity", value)
        assert squashed is not None and 0.0 <= squashed <= 1.0


def test_hard_wrapped_prose_is_not_mistaken_for_verse():
    """Short unterminated lines are necessary but not sufficient evidence of
    verse - plain-text email and anything pasted from a terminal looks the same
    until you check terminal-punctuation density."""
    wrapped = "\n".join(
        AI_LIKE.replace("\n", " ")[i:i + 72]
        for i in range(0, len(AI_LIKE.replace("\n", " ")), 72))
    result = OODDetector().analyse(build_context(wrapped))
    reasons = " ".join(result.raw["reasons"]).lower()
    assert "verse" not in reasons


# --------------------------------------------------------------------------
# response shape parity
# --------------------------------------------------------------------------


def test_abstain_response_has_the_same_top_level_keys():
    """A client that reads response["normalisation"]["fitted"] must not
    KeyError just because the user pasted a short text."""
    long_result = pipeline.analyse(HUMAN_LIKE, mode="fast")
    short_result = pipeline.analyse("Hello how are you?", mode="fast")
    missing = set(long_result) - set(short_result)
    assert not missing, f"abstain response is missing {sorted(missing)}"
    assert set(long_result["features"]) == set(short_result["features"])


def test_batch_survives_a_failing_item():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.post("/api/analyze/batch", json={
            "texts": [HUMAN_LIKE, "  ", AI_LIKE], "analysis_mode": "fast"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["succeeded"] == 2
    assert payload["results"][1]["ok"] is False
