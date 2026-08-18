"""
End-to-end pipeline behaviour and the response invariants that matter most.

The most important tests in the whole suite are in this file: the ones that
assert the system refuses to answer when it should, and that no number it emits
is ever NaN, negative, above 100, or a probability it has not earned.
"""

from __future__ import annotations

import math

import pytest

from app import pipeline
from app.core import config
from tests.conftest import (AI_LIKE, CODE_TEXT, EMOJI_TEXT, HUMAN_LIKE,
                            LONG_TEXT, MATH_TEXT, MIXED_TEXT, NON_ENGLISH,
                            ONE_SENTENCE, POETRY_TEXT, SHORT_TEXT,
                            SPECIAL_CHARS, URL_TEXT)

EVERY_TEXT = [HUMAN_LIKE, AI_LIKE, MIXED_TEXT, POETRY_TEXT, CODE_TEXT,
              MATH_TEXT, EMOJI_TEXT, URL_TEXT, SPECIAL_CHARS, NON_ENGLISH,
              ONE_SENTENCE]


def _numbers(payload, path="root"):
    """Yield every numeric leaf with a breadcrumb path."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from _numbers(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from _numbers(value, f"{path}[{index}]")
    elif isinstance(payload, bool):
        return
    elif isinstance(payload, (int, float)):
        yield path, float(payload)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad,code", [
    ("", "empty_input"),
    ("    \n\t ", "empty_input"),
    (None, "empty_input"),
    (42, "invalid_type"),
])
def test_invalid_input_raises_analysis_error(bad, code):
    with pytest.raises(pipeline.AnalysisError) as exc:
        pipeline.analyse(bad)
    assert exc.value.code == code


def test_invalid_mode_rejected():
    with pytest.raises(pipeline.AnalysisError) as exc:
        pipeline.analyse(HUMAN_LIKE, mode="turbo")
    assert exc.value.code == "invalid_mode"


def test_invalid_category_rejected():
    with pytest.raises(pipeline.AnalysisError) as exc:
        pipeline.analyse(HUMAN_LIKE, category="limerick")
    assert exc.value.code == "invalid_category"


# --------------------------------------------------------------------------
# short-text protection - the headline requirement
# --------------------------------------------------------------------------


def test_hello_how_are_you_gets_no_percentage():
    """The exact failure mode this project exists to avoid."""
    result = pipeline.analyse(SHORT_TEXT)
    assert result["result"]["classification"] == "Insufficient evidence"
    assert result["result"]["ai_origin_score"] is None
    assert result["result"]["human_score"] is None
    assert result["authorship"] is None
    assert result["result"]["abstained"] is True
    assert result["result"]["reliability_score"] == 0.0
    assert any("Insufficient evidence" in w for w in result["warnings"])


@pytest.mark.parametrize("word_count", [1, 5, 20, 49])
def test_everything_below_the_minimum_abstains(word_count):
    text = " ".join(["word"] * word_count)
    result = pipeline.analyse(text)
    assert result["result"]["ai_origin_score"] is None
    assert result["result"]["classification"] == "Insufficient evidence"


def test_just_above_minimum_is_scored_but_flagged_unreliable():
    text = " ".join(HUMAN_LIKE.split()[:config.MIN_WORDS + 8])
    result = pipeline.analyse(text)
    assert result["reliability"]["band"] in ("very_low", "moderate")
    if result["result"]["ai_origin_score"] is not None:
        assert result["result"]["confidence"] != "high"
    assert any("reliability" in w.lower() for w in result["warnings"])


def test_short_text_does_not_run_the_engines():
    """The length gate sits before inference so a tweet costs nothing."""
    result = pipeline.analyse(SHORT_TEXT)
    assert result["detectors"] == {}
    assert result["meta"]["detector_timings_ms"] == {}


# --------------------------------------------------------------------------
# numeric invariants across every kind of input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", EVERY_TEXT, ids=lambda t: t[:24].replace("\n", " "))
def test_no_nan_no_infinity_anywhere(text):
    result = pipeline.analyse(text, mode="fast")
    for path, value in _numbers(result):
        assert not math.isnan(value), f"NaN at {path}"
        assert not math.isinf(value), f"infinity at {path}"


@pytest.mark.parametrize("text", EVERY_TEXT, ids=lambda t: t[:24].replace("\n", " "))
def test_percentages_are_in_range(text):
    result = pipeline.analyse(text, mode="fast")
    for key in ("ai_origin_score", "human_score", "reliability_score"):
        value = result["result"].get(key)
        if value is not None:
            assert 0.0 <= value <= 100.0, f"{key} = {value}"
    for unit_key in ("chunks", "paragraphs", "sentences"):
        for unit in result[unit_key]:
            if unit.get("ai_score") is not None:
                assert 0.0 <= unit["ai_score"] <= 100.0
            assert 0.0 <= unit.get("confidence", 0.0) <= 100.0


@pytest.mark.parametrize("text", [HUMAN_LIKE, AI_LIKE, MIXED_TEXT])
def test_authorship_percentages_sum_to_one_hundred(text):
    result = pipeline.analyse(text, mode="standard")
    authorship = result["authorship"]
    if authorship is None:
        pytest.skip("system abstained")
    total = authorship["human"] + authorship["pure_ai"] + authorship["humanized_ai"]
    assert abs(total - 100.0) < 0.15, f"authorship sums to {total}"


@pytest.mark.parametrize("text", [HUMAN_LIKE, AI_LIKE, MIXED_TEXT])
def test_ai_origin_equals_pure_plus_humanized(text):
    result = pipeline.analyse(text, mode="standard")
    authorship = result["authorship"]
    if authorship is None:
        pytest.skip("system abstained")
    expected = authorship["pure_ai"] + authorship["humanized_ai"]
    assert abs(result["result"]["ai_origin_score"] - expected) < 0.15


@pytest.mark.parametrize("text", [HUMAN_LIKE, AI_LIKE])
def test_ai_and_human_sum_to_one_hundred(text):
    result = pipeline.analyse(text, mode="standard")
    if result["result"]["ai_origin_score"] is None:
        pytest.skip("system abstained")
    total = result["result"]["ai_origin_score"] + result["result"]["human_score"]
    assert abs(total - 100.0) < 0.15


# --------------------------------------------------------------------------
# the honesty contract
# --------------------------------------------------------------------------


def test_score_type_matches_what_is_actually_installed():
    result = pipeline.analyse(AI_LIKE, mode="standard")
    trained = result["ensemble"]["trained"]
    calibrated = result["calibration"]["fitted"]
    expected = "calibrated_probability" if (trained and calibrated) \
        else "uncalibrated_detection_score"
    if not result["result"]["abstained"]:
        assert result["result"]["score_type"] == expected
        assert result["result"]["is_probability"] is (trained and calibrated)


def test_untrained_system_warns_loudly():
    """Whichever untrained state we are in, the user must be told about it.

    Two distinct outcomes are legitimate here: with enough detector signals the
    system emits an explicitly uncalibrated *detection score*, and with too few
    it abstains outright. Both must be stated in the warnings; neither may pass
    silently as a probability.
    """
    result = pipeline.analyse(AI_LIKE, mode="standard")
    if result["ensemble"]["trained"]:
        pytest.skip("a trained meta-classifier is installed")
    joined = " ".join(result["warnings"]).lower()
    if result["result"]["abstained"]:
        assert "too few detectors" in joined
        assert "no authorship classification was produced" in joined
    else:
        assert "detection score" in joined
        assert "not a statistical probability" in joined or \
            "not a calibrated probability" in joined


def test_disclaimer_always_present():
    for text in (SHORT_TEXT, HUMAN_LIKE, CODE_TEXT):
        result = pipeline.analyse(text, mode="fast")
        assert "proof" in result["disclaimer"].lower()
        assert "misconduct" in result["disclaimer"].lower()


def test_unavailable_detectors_carry_a_reason_and_no_score():
    result = pipeline.analyse(AI_LIKE, mode="standard")
    for name, block in result["detectors"].items():
        if not block["available"]:
            assert block.get("reason"), f"{name} has no reason"
            assert "score" not in block or block.get("score") is None
        assert result["detector_signals"][name] is None or block["available"]


def test_slop_never_feeds_authorship():
    """Slop is a separate axis; the ensemble must not consume it."""
    result = pipeline.analyse(AI_LIKE, mode="fast")
    contributions = result["ensemble"]["detail"].get("contributions", [])
    assert all(c["detector"] != "slop" for c in contributions)
    assert result["slop"].get("independent_of_authorship") is True


def test_ood_never_feeds_authorship():
    result = pipeline.analyse(CODE_TEXT, mode="fast")
    contributions = result["ensemble"]["detail"].get("contributions", [])
    assert all(c["detector"] != "ood" for c in contributions)


# --------------------------------------------------------------------------
# out-of-distribution handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [CODE_TEXT, MATH_TEXT, POETRY_TEXT, NON_ENGLISH])
def test_out_of_distribution_input_lowers_reliability(text):
    result = pipeline.analyse(text, mode="fast")
    assert result["reliability"]["ood_score"] is not None
    if result["reliability"]["ood_score"] >= config.OOD_HARD_THRESHOLD:
        assert result["result"]["confidence"] == "low"
        assert any("outside the detector" in w for w in result["warnings"])
        assert result["reliability"]["ood_reasons"]


def test_non_english_is_flagged():
    result = pipeline.analyse(NON_ENGLISH, mode="fast")
    assert result["language"]["is_supported"] is False
    assert any("outside the supported set" in w for w in result["warnings"])


# --------------------------------------------------------------------------
# structure of the response
# --------------------------------------------------------------------------


def test_response_matches_the_declared_schema():
    from app.schemas.response import AnalyzeResponse

    for text in (HUMAN_LIKE, SHORT_TEXT, CODE_TEXT):
        payload = pipeline.analyse(text, mode="fast")
        AnalyzeResponse.model_validate(payload)   # raises on mismatch


def test_units_are_present_and_indexed():
    result = pipeline.analyse(MIXED_TEXT, mode="standard")
    assert len(result["paragraphs"]) == result["statistics"]["paragraphs"]
    assert [p["index"] for p in result["paragraphs"]] == \
        list(range(len(result["paragraphs"])))
    assert [s["index"] for s in result["sentences"]] == \
        list(range(len(result["sentences"])))


def test_sentence_offsets_point_at_the_real_text():
    from app.preprocessing import cleaner

    normalised = cleaner.normalise(MIXED_TEXT)
    result = pipeline.analyse(MIXED_TEXT, mode="standard")
    for sentence in result["sentences"]:
        start, end = sentence["start"], sentence["end"]
        assert normalised.text[start:end] == sentence["text"]


def test_sentence_labels_are_from_the_allowed_set():
    result = pipeline.analyse(MIXED_TEXT, mode="standard")
    allowed = {"likely_ai", "likely_human", "uncertain", "unscored"}
    assert {s["label"] for s in result["sentences"]} <= allowed


def test_classification_is_a_declared_band():
    result = pipeline.analyse(AI_LIKE, mode="fast")
    allowed = {b.label for b in config.DISPLAY_BANDS} | \
        {"Insufficient evidence", "Insufficient detector evidence"}
    assert result["result"]["classification"] in allowed


def test_processing_time_reported():
    result = pipeline.analyse(HUMAN_LIKE, mode="fast")
    assert result["processing_ms"] > 0
    assert result["meta"]["processing_ms"] == result["processing_ms"]


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fast", "standard", "deep"])
def test_all_modes_run(mode):
    result = pipeline.analyse(HUMAN_LIKE, mode=mode)
    assert result["meta"]["analysis_mode"] == mode


def test_fast_mode_skips_the_expensive_engines():
    result = pipeline.analyse(HUMAN_LIKE, mode="fast")
    for name in ("probability", "curvature", "binoculars", "semantic",
                 "humanization"):
        block = result["detectors"][name]
        assert block["available"] is False
        assert "fast" in block["reason"]


def test_manual_category_overrides_detection():
    result = pipeline.analyse(AI_LIKE, category="technical", mode="fast")
    assert result["category"]["name"] == "technical"
    assert result["category"]["manual"] is True


# --------------------------------------------------------------------------
# long input
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_long_document_is_chunked_and_bounded():
    result = pipeline.analyse(LONG_TEXT, mode="fast")
    assert result["statistics"]["words"] > 1500
    assert len(result["chunks"]) > 1
    assert len(result["chunks"]) <= config.MAX_CHUNKS
    assert len(result["sentences"]) <= config.MAX_SENTENCES_SCORED
    for path, value in _numbers(result):
        assert not math.isnan(value) and not math.isinf(value)
