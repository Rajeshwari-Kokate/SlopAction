"""
Detection engines.

The recurring theme: an engine that cannot run must say so and return no score.
There is no such thing as a placeholder detector output in this system.
"""

from __future__ import annotations

import math

import pytest

from app.core import config
from app.core.types import DetectorResult
from app.detectors import normalisation
from app.detectors.binocular_detector import BinocularDetector
from app.detectors.curvature_detector import CurvatureDetector
from app.detectors.humanization_detector import HumanizationDetector
from app.detectors.ood_detector import OODDetector
from app.detectors.probability_detector import ProbabilityDetector
from app.detectors.semantic_detector import SemanticDetector
from app.detectors.slop_detector import SlopDetector
from app.detectors.stylometry_detector import StylometryDetector
from app.detectors.transformer_detector import TransformerDetector, _map_label
from training.common import build_context
from tests.conftest import (AI_LIKE, CODE_TEXT, EMOJI_TEXT, HUMAN_LIKE,
                            MATH_TEXT, NON_ENGLISH, POETRY_TEXT)

ALL_ENGINES = [
    TransformerDetector(), ProbabilityDetector(), CurvatureDetector(),
    BinocularDetector(), StylometryDetector(), SemanticDetector(),
    HumanizationDetector(), SlopDetector(), OODDetector(),
]


# --------------------------------------------------------------------------
# the core honesty contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ALL_ENGINES, ids=lambda e: e.name)
def test_unavailable_engines_return_no_score(engine):
    context = build_context(AI_LIKE)
    result = engine.analyse(context)
    if not result.available:
        assert result.score is None, (
            f"{engine.name} returned a score while reporting unavailable")
        assert result.reason, f"{engine.name} gave no reason for being unavailable"
    else:
        if result.score is not None:
            assert 0.0 <= result.score <= 1.0


@pytest.mark.parametrize("engine", ALL_ENGINES, ids=lambda e: e.name)
def test_engines_never_raise(engine):
    """A broken engine degrades to unavailable; it never takes down a request."""
    for text in (HUMAN_LIKE, AI_LIKE, POETRY_TEXT, CODE_TEXT, MATH_TEXT,
                 EMOJI_TEXT, NON_ENGLISH):
        context = build_context(text)
        result = engine.analyse(context)
        assert isinstance(result, DetectorResult)
        assert result.elapsed_ms >= 0.0


def test_detector_result_forces_score_to_none_when_unavailable():
    result = DetectorResult(name="x", available=False, score=0.9)
    assert result.score is None


def test_detector_result_clamps_scores():
    assert DetectorResult(name="x", available=True, score=1.7).score == 1.0
    assert DetectorResult(name="x", available=True, score=-3.0).score == 0.0


def test_detector_result_sanitises_nan_features():
    result = DetectorResult(name="x", available=True, score=0.5,
                            features={"a": float("nan"), "b": float("inf")})
    assert result.features["a"] == 0.0
    assert result.features["b"] == 0.0


# --------------------------------------------------------------------------
# stylometry
# --------------------------------------------------------------------------


def test_stylometry_features_always_available():
    result = StylometryDetector().analyse(build_context(HUMAN_LIKE))
    assert result.available is True
    assert len(result.features) > 120


def test_untrained_stylometry_is_labelled_as_a_fallback():
    result = StylometryDetector().analyse(build_context(HUMAN_LIKE))
    if not result.trained:
        assert result.is_fallback is True
        assert result.method == "experimental_rule_based_fallback"
        assert any("EXPERIMENTAL" in w for w in result.warnings)
        assert any("not a machine-learning prediction" in w.lower()
                   or "not a machine-learning" in w.lower()
                   for w in result.warnings)
        assert result.raw["status"] == "untrained"


def test_untrained_stylometry_is_shrunk_towards_neutral():
    """An unlearned rule set is not entitled to a strong opinion."""
    result = StylometryDetector().analyse(build_context(AI_LIKE))
    if not result.trained and result.score is not None:
        assert 0.2 <= result.score <= 0.8


# --------------------------------------------------------------------------
# semantic
# --------------------------------------------------------------------------


def test_semantic_detects_paraphrased_repetition():
    repetitive = """
Artificial intelligence is transforming education across the world today.
Machine learning is changing how schools teach their students everywhere.
Education is being revolutionised by artificial intelligence in many places.
AI technology is reshaping the way education works in classrooms globally.
Schools everywhere are being changed by the arrival of machine intelligence.
The world of teaching is being transformed by artificial intelligence systems.
"""
    varied = """
The boiler failed on Saturday morning without any warning at all.
My neighbour lent me a wrench, which turned out to be the useful part.
Replacement parts cost forty-eight pounds and arrive on Tuesday afternoon.
The children find the kettle arrangement enormously entertaining.
Rain is forecast for the weekend, which complicates the loft insulation plan.
I have not yet told my wife about the second invoice from the plumber.
"""
    r = SemanticDetector().analyse(build_context(repetitive))
    v = SemanticDetector().analyse(build_context(varied))
    if r.available and v.available:
        assert r.features["mean_similarity"] > v.features["mean_similarity"]
        assert r.features["semantic_redundancy_score"] > \
            v.features["semantic_redundancy_score"]


def test_semantic_requires_enough_sentences():
    result = SemanticDetector().analyse(build_context(
        "One sentence only, which is nowhere near enough for this engine."))
    assert result.available is False
    assert "sentence" in (result.reason or "").lower()


def test_semantic_tfidf_fallback_is_flagged():
    result = SemanticDetector().analyse(build_context(AI_LIKE))
    if result.available and "tfidf" in result.method:
        assert result.is_fallback is True
        assert any("TF-IDF" in w for w in result.warnings)


# --------------------------------------------------------------------------
# slop
# --------------------------------------------------------------------------


def test_slop_score_in_range_and_weights_published():
    result = SlopDetector().analyse(build_context(AI_LIKE))
    assert result.available is True
    assert 0.0 <= result.raw["score"] <= 100.0
    assert result.raw["level"] in {"very_low", "low", "moderate", "high", "very_high"}
    assert result.raw["weights"] == {
        k: config.SLOP_WEIGHTS.get(k, 0.0) for k in result.raw["components"]}
    assert result.raw["independent_of_authorship"] is True


def test_slop_separates_padded_from_dense_writing():
    padded = ("""It is important to note that in today's fast-paced world,
technology plays a crucial role. Furthermore, technology is important in the
modern era. Moreover, in today's digital age, technology plays a vital role.
Additionally, it goes without saying that technology matters a great deal.
Overall, technology is generally quite important in many different ways.
In conclusion, technology plays a crucial role in today's world. To summarise,
technology is important. Ultimately, the key to success is technology.\n\n""" * 3)
    dense = ("""Ofsted inspected 1,842 schools in England during 2023. Of those,
14% were rated "requires improvement", down from 19% in 2019. The inspectorate
attributed the change to revised safeguarding guidance issued in March 2022.
Amanda Spielman told the Education Select Committee that inspector headcount
fell by 240 over the same period. Three local authorities - Knowsley, Blackpool
and Middlesbrough - accounted for a fifth of the inadequate ratings.\n\n""" * 3)
    p = SlopDetector().analyse(build_context(padded))
    d = SlopDetector().analyse(build_context(dense))
    assert p.raw["score"] > d.raw["score"]


def test_slop_reports_unmeasured_components_rather_than_zeroing_them():
    result = SlopDetector().analyse(build_context(AI_LIKE))
    unmeasured = result.raw["components_unmeasured"]
    for key in unmeasured:
        assert result.raw["components"][key] is None


# --------------------------------------------------------------------------
# humanization
# --------------------------------------------------------------------------


def test_humanization_requires_minimum_length():
    result = HumanizationDetector().analyse(build_context(
        "Short text that is nowhere near sixty words in length at all."))
    assert result.available is False


def test_humanization_returns_three_views():
    result = HumanizationDetector().analyse(build_context(AI_LIKE))
    assert result.available is True
    prefixes = {key.split("__")[0] for key in result.features}
    assert {"orig", "cont", "expr", "div"} <= prefixes


def test_humanization_untrained_is_labelled():
    result = HumanizationDetector().analyse(build_context(AI_LIKE))
    if not result.trained:
        assert result.is_fallback is True
        assert any("EXPERIMENTAL" in w for w in result.warnings)
        assert result.raw["status"] == "untrained"


def test_convention_mixing_signal_responds_to_mixed_spelling():
    consistent = ("The organisation analysed the behaviour of the colour "
                  "sensors in the centre of the theatre. " * 12)
    mixed = ("The organization analysed the behavior of the colour sensors in "
             "the center of the theatre. " * 12)
    c = HumanizationDetector().analyse(build_context(consistent))
    m = HumanizationDetector().analyse(build_context(mixed))
    assert m.features["expr__spelling_convention_mixing"] > \
        c.features["expr__spelling_convention_mixing"]


def test_homoglyph_detection():
    clean = HUMAN_LIKE
    obfuscated = HUMAN_LIKE.replace("e", "е", 4)   # Cyrillic small ie
    c = HumanizationDetector().analyse(build_context(clean))
    o = HumanizationDetector().analyse(build_context(obfuscated))
    assert o.features["expr__homoglyph_count"] > c.features["expr__homoglyph_count"]


# --------------------------------------------------------------------------
# OOD
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_reason", [
    (CODE_TEXT, "source-code"),
    (MATH_TEXT, "mathematical"),
    (POETRY_TEXT, "verse"),
    (NON_ENGLISH, "supported"),
])
def test_ood_flags_out_of_distribution_inputs(text, expected_reason):
    result = OODDetector().analyse(build_context(text))
    assert result.available is True
    assert result.score > config.OOD_ALERT_THRESHOLD, \
        f"expected elevated OOD for this input, got {result.score}"
    joined = " ".join(result.raw["reasons"]).lower()
    assert expected_reason in joined


def test_ood_is_quiet_on_ordinary_prose():
    for text in (HUMAN_LIKE, AI_LIKE):
        result = OODDetector().analyse(build_context(text))
        assert result.score < config.OOD_ALERT_THRESHOLD


def test_ood_never_changes_authorship_direction():
    """OOD is a reliability discount, so it must not be an authorship signal."""
    result = OODDetector().analyse(build_context(CODE_TEXT))
    assert result.class_probabilities is None
    assert "never changes the AI/human balance" in result.raw["effect"]


# --------------------------------------------------------------------------
# transformer label mapping (no model required)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [
    ("Human", "human"), ("human", "human"), ("LABEL_0", "human"),
    ("ChatGPT", "pure_ai"), ("fake", "pure_ai"), ("AI-generated", "pure_ai"),
    ("humanized_ai", "humanized_ai"), ("paraphrased", "humanized_ai"),
    ("Humanised AI", "humanized_ai"),
    ("something_unrelated", None),
])
def test_label_mapping(label, expected):
    assert _map_label(label) == expected


def test_two_class_checkpoint_is_reported_as_such():
    mapping, unmapped = TransformerDetector._label_mapping(
        {0: "Human", 1: "ChatGPT"})
    assert set(mapping.values()) == {"human", "pure_ai"}
    assert "humanized_ai" not in mapping.values()
    assert unmapped == []


def test_three_class_checkpoint_is_recognised():
    mapping, _ = TransformerDetector._label_mapping(
        {0: "human", 1: "pure_ai", 2: "humanized_ai"})
    assert set(mapping.values()) == {"human", "pure_ai", "humanized_ai"}


def test_unmappable_labels_are_reported_not_guessed():
    mapping, unmapped = TransformerDetector._label_mapping(
        {0: "positive", 1: "negative"})
    assert mapping == {}
    assert set(unmapped) == {"positive", "negative"}


def test_pooling_normalises_to_one():
    pooled = TransformerDetector._pool(
        [{"human": 0.8, "pure_ai": 0.2, "humanized_ai": 0.0},
         {"human": 0.1, "pure_ai": 0.9, "humanized_ai": 0.0}],
        [100.0, 300.0])
    assert abs(sum(pooled.values()) - 1.0) < 1e-9
    # the longer segment must dominate
    assert pooled["pure_ai"] > pooled["human"]


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_squash_is_monotone_and_direction_aware():
    # perplexity has direction -1: lower perplexity is more AI-associated
    low = normalisation.squash("perplexity", 8.0)
    high = normalisation.squash("perplexity", 90.0)
    assert low > high
    # sampling discrepancy has direction +1
    assert normalisation.squash("sampling_discrepancy", 3.0) > \
        normalisation.squash("sampling_discrepancy", -1.0)


def test_squash_bounds_and_unknown_features():
    for value in (-1e6, 0.0, 1e6):
        squashed = normalisation.squash("perplexity", value)
        assert 0.0 <= squashed <= 1.0
    assert normalisation.squash("no_such_feature", 1.0) is None
    assert normalisation.squash("perplexity", float("nan")) is None


def test_normalisation_reports_whether_it_was_fitted():
    described = normalisation.describe()
    assert isinstance(described["fitted"], bool)
    assert "not a probability" in described["note"].lower()
