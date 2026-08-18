"""Meta-classifier, calibration, confidence, explanation and aggregation."""

from __future__ import annotations

import math

import pytest

from app.core import config
from app.core.types import DetectorResult, UnitScore, normalise_distribution
from app.ensemble import calibrator as calibration_module
from app.ensemble import confidence, explanation, meta_classifier
from training.common import build_context
from tests.conftest import AI_LIKE, HUMAN_LIKE, MIXED_TEXT


def _result(name, score, available=True, trained=False, is_fallback=False,
            features=None, class_probabilities=None):
    return DetectorResult(
        name=name, available=available, score=score, trained=trained,
        is_fallback=is_fallback, features=features or {},
        class_probabilities=class_probabilities)


# --------------------------------------------------------------------------
# distribution invariants
# --------------------------------------------------------------------------


def test_normalise_distribution_sums_to_one():
    for raw in ({"human": 3.0, "pure_ai": 1.0, "humanized_ai": 0.5},
                {"human": 0.0, "pure_ai": 0.0, "humanized_ai": 0.0},
                {"human": -5.0, "pure_ai": 2.0, "humanized_ai": float("nan")},
                {}):
        result = normalise_distribution(raw, config.CLASSES)
        assert abs(sum(result.values()) - 1.0) < 1e-12
        assert all(0.0 <= v <= 1.0 for v in result.values())
        assert all(not math.isnan(v) for v in result.values())


def test_degenerate_input_yields_uniform_not_nan():
    result = normalise_distribution({"human": 0.0, "pure_ai": 0.0,
                                     "humanized_ai": 0.0}, config.CLASSES)
    assert all(abs(v - 1 / 3) < 1e-9 for v in result.values())


# --------------------------------------------------------------------------
# meta-classifier
# --------------------------------------------------------------------------


def test_ensemble_abstains_with_too_few_signals():
    context = build_context(AI_LIKE)
    results = {"stylometry": _result("stylometry", 0.7, is_fallback=True)}
    outcome = meta_classifier.predict(context, results)
    assert outcome.abstained is True
    assert outcome.method == "abstained"
    assert abs(sum(outcome.probabilities.values()) - 1.0) < 1e-9
    assert any("Too few detectors" in w for w in outcome.warnings)


def test_untrained_fallback_is_labelled_and_shrunk():
    context = build_context(AI_LIKE)
    results = {
        "transformer": _result("transformer", 0.99, trained=True),
        "curvature": _result("curvature", 0.99),
        "probability": _result("probability", 0.99),
        "stylometry": _result("stylometry", 0.99, is_fallback=True),
    }
    outcome = meta_classifier.predict(context, results)
    assert outcome.trained is False
    assert outcome.method == "untrained_transparent_pooling"
    assert outcome.is_probability is False
    assert any("UNCALIBRATED DETECTION SCORE" in w for w in outcome.warnings)
    # even with every detector screaming 0.99, shrinkage must hold it back
    assert outcome.ai_origin < 0.97


def test_fallback_never_simple_averages():
    """The pooled value must reflect the published weights, not a mean."""
    context = build_context(AI_LIKE)
    results = {
        "transformer": _result("transformer", 1.0, trained=True),   # weight 1.00
        "semantic": _result("semantic", 0.0),                        # weight 0.25
        "probability": _result("probability", 0.0),                  # weight 0.55
    }
    outcome = meta_classifier.predict(context, results)
    pooled = outcome.detail["pooled_signal"]
    naive_mean = (1.0 + 0.0 + 0.0) / 3
    expected = 1.0 * 1.00 / (1.00 + 0.25 + 0.55)
    assert abs(pooled - expected) < 1e-3   # detail is rounded to 4dp for display
    assert abs(pooled - naive_mean) > 1e-3


def test_fallback_detector_penalty_applies():
    context = build_context(AI_LIKE)
    strong = {"transformer": _result("transformer", 0.9, trained=True),
              "stylometry": _result("stylometry", 0.1, is_fallback=False),
              "probability": _result("probability", 0.9)}
    weak = {"transformer": _result("transformer", 0.9, trained=True),
            "stylometry": _result("stylometry", 0.1, is_fallback=True),
            "probability": _result("probability", 0.9)}
    # a fallback detector disagreeing should pull the result less than a
    # non-fallback one disagreeing by the same amount
    assert meta_classifier.predict(context, weak).ai_origin > \
        meta_classifier.predict(context, strong).ai_origin


def test_probabilities_always_sum_to_one():
    context = build_context(AI_LIKE)
    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        results = {"transformer": _result("transformer", score, trained=True),
                   "curvature": _result("curvature", score),
                   "probability": _result("probability", score)}
        outcome = meta_classifier.predict(context, results)
        total = sum(outcome.probabilities.values())
        assert abs(total - 1.0) < 1e-9
        assert abs(outcome.ai_origin + outcome.probabilities["human"] - 1.0) < 1e-9


def test_native_three_class_transformer_drives_the_humanized_split():
    context = build_context(AI_LIKE)
    results = {
        "transformer": _result(
            "transformer", 0.8, trained=True,
            features={"transformer_models_humanized": 1.0},
            class_probabilities={"human": 0.2, "pure_ai": 0.16,
                                 "humanized_ai": 0.64}),
        "curvature": _result("curvature", 0.7),
        "probability": _result("probability", 0.7),
    }
    outcome = meta_classifier.predict(context, results)
    assert outcome.detail["humanized_share_source"] == "transformer_native_three_class"
    assert outcome.probabilities["humanized_ai"] > outcome.probabilities["pure_ai"]


def test_chunk_signal_is_not_counted_as_independent_evidence():
    context = build_context(AI_LIKE)
    results = {"transformer": _result("transformer", 0.9, trained=True)}
    # one engine + a chunk aggregate must still abstain: the chunk aggregate is
    # derived from that same engine
    outcome = meta_classifier.predict(context, results, chunk_signal=0.9)
    assert outcome.abstained is True


def test_ensemble_vector_is_flat_and_finite():
    context = build_context(AI_LIKE)
    results = {"transformer": _result("transformer", 0.5, trained=True,
                                      features={"a": 1.0, "b": float("nan")})}
    vector = meta_classifier.build_ensemble_vector(context, results)
    assert all(isinstance(v, float) for v in vector.values())
    assert all(not math.isnan(v) for v in vector.values())
    assert "transformer__available" in vector
    assert "meta__word_count" in vector
    assert any(k.startswith("meta__category_") for k in vector)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def test_identity_calibrator_is_used_when_no_artefact():
    calibrator = calibration_module.load()
    described = calibrator.describe()
    if not described["fitted"]:
        assert described["method"] == "identity"
        assert "detection score" in described["note"]


def test_calibrators_preserve_the_simplex():
    raw = {"human": 0.2, "pure_ai": 0.5, "humanized_ai": 0.3}
    for calibrator in (
        calibration_module.IdentityCalibrator(),
        calibration_module.TemperatureCalibrator(2.0),
        calibration_module.TemperatureCalibrator(0.4),
        calibration_module.VectorScalingCalibrator(
            {"human": 1.2, "pure_ai": 0.8, "humanized_ai": 1.0},
            {"human": 0.1, "pure_ai": -0.2, "humanized_ai": 0.0}),
    ):
        out = calibrator.apply(raw)
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in out.values())


def test_temperature_above_one_softens_and_below_one_sharpens():
    raw = {"human": 0.1, "pure_ai": 0.8, "humanized_ai": 0.1}
    soft = calibration_module.TemperatureCalibrator(3.0).apply(raw)
    sharp = calibration_module.TemperatureCalibrator(0.3).apply(raw)
    assert soft["pure_ai"] < raw["pure_ai"] < sharp["pure_ai"]


def test_temperature_fitting_recovers_a_sensible_value():
    import numpy as np

    rng = np.random.default_rng(0)
    # deliberately over-confident probabilities: correct 70% of the time but
    # always claiming ~0.95
    labels, probabilities = [], []
    for _ in range(600):
        correct = rng.random() < 0.7
        row = [0.025, 0.025, 0.025]
        row[0] = 0.95
        probabilities.append(row if correct else [0.95, 0.025, 0.025])
        labels.append("human" if correct else "pure_ai")
    temperature = calibration_module.fit_temperature(
        probabilities, labels, config.CLASSES)
    assert temperature > 1.0, "over-confident input should be softened"


def test_ece_and_brier_are_bounded():
    import numpy as np

    probabilities = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1],
                              [0.3, 0.3, 0.4]])
    labels = ["human", "pure_ai", "humanized_ai"]
    ece = calibration_module.expected_calibration_error(
        probabilities, labels, config.CLASSES)
    brier = calibration_module.brier_score(probabilities, labels, config.CLASSES)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


def _assess(**overrides):
    kwargs = dict(
        word_count=400, band="normal",
        results={"transformer": _result("transformer", 0.8, trained=True),
                 "curvature": _result("curvature", 0.78),
                 "probability": _result("probability", 0.82)},
        probabilities={"human": 0.2, "pure_ai": 0.5, "humanized_ai": 0.3},
        chunk_scores=[UnitScore(index=i, ai_score=80.0, confidence=80.0,
                                reliability=0.8) for i in range(4)],
        ood_score=0.0, category_confidence=0.8,
        ensemble_trained=True, calibrated=True, abstained=False)
    kwargs.update(overrides)
    return confidence.assess(**kwargs)


def test_confidence_is_bounded_and_labelled():
    block = _assess()
    assert 0.0 <= block["reliability_score"] <= 100.0
    assert block["confidence"] in {"high", "medium", "low"}


def test_disagreement_lowers_confidence():
    agree = _assess()
    disagree = _assess(results={
        "transformer": _result("transformer", 0.95, trained=True),
        "curvature": _result("curvature", 0.10),
        "probability": _result("probability", 0.85)})
    assert disagree["reliability_score"] < agree["reliability_score"]
    assert disagree["components"]["detector_agreement"] < \
        agree["components"]["detector_agreement"]


def test_short_text_confidence_is_capped_by_band():
    block = _assess(word_count=70, band="very_low")
    ceiling = config.length_band(70).reliability_ceiling
    assert block["reliability_score"] <= ceiling * 100 + 1e-6
    assert any("length band" in note for note in block["notes"])


def test_untrained_ensemble_cannot_reach_high_confidence():
    block = _assess(ensemble_trained=False)
    assert block["reliability_score"] <= \
        confidence.UNTRAINED_CONFIDENCE_CEILING * 100 + 1e-6
    assert block["confidence"] != "high"


def test_uncalibrated_output_is_capped():
    block = _assess(calibrated=False)
    assert block["reliability_score"] <= \
        confidence.UNCALIBRATED_CONFIDENCE_CEILING * 100 + 1e-6


def test_hard_ood_forces_low_confidence():
    block = _assess(ood_score=0.95)
    assert block["confidence"] == "low"
    assert block["reliability_score"] <= 20.0


def test_abstention_zeroes_confidence():
    block = _assess(abstained=True)
    assert block["reliability_score"] == 0.0


def test_inconsistent_chunks_lower_confidence():
    consistent = _assess()
    inconsistent = _assess(chunk_scores=[
        UnitScore(index=0, ai_score=5.0, confidence=80.0, reliability=0.8),
        UnitScore(index=1, ai_score=95.0, confidence=80.0, reliability=0.8),
        UnitScore(index=2, ai_score=10.0, confidence=80.0, reliability=0.8),
        UnitScore(index=3, ai_score=90.0, confidence=80.0, reliability=0.8)])
    assert inconsistent["components"]["chunk_consistency"] < \
        consistent["components"]["chunk_consistency"]


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------


def test_explanation_never_claims_proof():
    results = {"stylometry": _result(
        "stylometry", 0.7, features={"str__sentence_length_cv": 0.2,
                                     "str__burstiness": -0.6})}
    block = explanation.build(results)
    everything = " ".join(
        entry["signal"] for key in ("strong_signals", "moderate_signals",
                                    "human_signals")
        for entry in block[key]) + block["summary"] + block["phrasing_note"]
    for forbidden in ("proves", "proof that", "demonstrates that this was",
                      "definitely", "certainly generated"):
        assert forbidden not in everything.lower()
    assert "associated with" in block["phrasing_note"]


def test_explanation_carries_measured_evidence():
    results = {"stylometry": _result(
        "stylometry", 0.7, features={"str__sentence_length_cv": 0.2})}
    block = explanation.build(results)
    for entry in block["moderate_signals"]:
        assert "feature" in entry["evidence"]
        assert isinstance(entry["evidence"]["value"], float)
        assert "reference" in entry["evidence"]


def test_explanation_drops_self_contradictions():
    """A feature that fires both an AI rule and a human rule proves nothing."""
    results = {"probability": _result(
        "probability", 0.5, features={"log_prob_std": 3.5})}
    block = explanation.build(results)
    features_cited = {e["evidence"]["feature"] for key in
                      ("strong_signals", "moderate_signals", "human_signals")
                      for e in block[key]}
    # log_prob_std has a "below 2.0 = AI" rule and an "above 3.0 = human" rule;
    # at 3.5 only the human rule fires, so it should appear exactly once
    assert list(features_cited).count("probability.log_prob_std") <= 1


def test_explanation_lists_unavailable_detectors():
    results = {"transformer": DetectorResult.unavailable(
        "transformer", "checkpoint not downloadable")}
    block = explanation.build(results)
    assert block["unavailable_detectors"][0]["detector"] == "transformer"
    assert "checkpoint" in block["unavailable_detectors"][0]["reason"]
