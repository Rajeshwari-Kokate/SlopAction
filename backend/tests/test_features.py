"""Feature extractors: every value must be finite, in range, and length-robust
where it claims to be."""

from __future__ import annotations

import math

import pytest

from app.core.types import safe_float
from app.features import (discourse, lexical, punctuation, repetition, syntax,
                          vectorizer)
from app.preprocessing import cleaner
from app.preprocessing import tokenizer as text_tokenizer
from tests.conftest import (AI_LIKE, CODE_TEXT, EMOJI_TEXT, HUMAN_LIKE,
                            ONE_SENTENCE, POETRY_TEXT, SPECIAL_CHARS)

TEXTS = [HUMAN_LIKE, AI_LIKE, POETRY_TEXT, CODE_TEXT, EMOJI_TEXT,
         SPECIAL_CHARS, ONE_SENTENCE]


def _prepare(text: str):
    normalised = cleaner.normalise(text)
    paragraphs, sentences = text_tokenizer.segment(normalised.text)
    return normalised, paragraphs, sentences, text_tokenizer.words(normalised.text)


# --------------------------------------------------------------------------
# universal invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", TEXTS)
def test_all_features_finite(text):
    normalised, paragraphs, sentences, words = _prepare(text)
    blocks = [
        lexical.extract(words, normalised.text),
        syntax.structure_features(sentences, paragraphs),
        syntax.pos_features(normalised.text)["features"],
        punctuation.extract(normalised.text, sentences, len(words)),
        discourse.extract(normalised.text, sentences, len(words)),
        repetition.extract(normalised.text, sentences, paragraphs),
    ]
    for block in blocks:
        for key, value in block.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"
            assert not math.isnan(float(value)), f"{key} is NaN"
            assert not math.isinf(float(value)), f"{key} is infinite"


@pytest.mark.parametrize("text", TEXTS)
def test_ratio_features_are_bounded(text):
    normalised, paragraphs, sentences, words = _prepare(text)
    blocks = {
        "lexical": lexical.extract(words, normalised.text),
        "structure": syntax.structure_features(sentences, paragraphs),
        "repetition": repetition.extract(normalised.text, sentences, paragraphs),
    }
    for name, block in blocks.items():
        for key, value in block.items():
            if key.endswith("_ratio") or key.endswith("_rate") \
                    or key.startswith("distinct_"):
                assert -1e-9 <= value <= 1.0 + 1e-9, f"{name}.{key} = {value}"


def test_empty_word_list_is_safe():
    features = lexical.extract([], "")
    assert all(value == 0.0 for value in features.values())


def test_single_sentence_features_do_not_crash():
    normalised, paragraphs, sentences, words = _prepare(ONE_SENTENCE)
    structure = syntax.structure_features(sentences, paragraphs)
    assert structure["sentence_count"] == 1
    assert structure["burstiness"] == 0.0          # undefined for n<2, not NaN
    assert structure["sentence_length_std"] == 0.0


# --------------------------------------------------------------------------
# lexical
# --------------------------------------------------------------------------


def test_mattr_is_length_robust_but_ttr_is_not():
    """Raw TTR falls with length; MATTR must not.

    This is the whole reason MATTR is in the feature set: a detector using raw
    TTR learns 'long text = AI'.
    """
    short = " ".join(HUMAN_LIKE.split()[:60])
    long = HUMAN_LIKE + " " + AI_LIKE
    short_features = lexical.extract(text_tokenizer.words(short), short)
    long_features = lexical.extract(text_tokenizer.words(long), long)

    assert short_features["type_token_ratio"] > long_features["type_token_ratio"]
    drift = abs(short_features["mattr_50"] - long_features["mattr_50"])
    assert drift < 0.25, "MATTR should be far more stable across lengths than TTR"


def test_repetition_raises_repeated_word_ratio():
    varied = HUMAN_LIKE
    repeated = "the model the model the model " * 40
    v = lexical.extract(text_tokenizer.words(varied), varied)
    r = lexical.extract(text_tokenizer.words(repeated), repeated)
    assert r["repeated_word_ratio"] > v["repeated_word_ratio"]
    assert r["type_token_ratio"] < v["type_token_ratio"]
    assert r["compression_ratio"] < v["compression_ratio"]


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_uniform_sentences_have_low_burstiness():
    uniform = " ".join(["The system processes the data and returns a value."] * 20)
    varied = ("Yes. The system, which had been running unattended since the "
              "previous Thursday afternoon without any supervision whatsoever, "
              "finally returned a value. Good. It worked. Eventually, after a "
              "great deal of prodding and several restarts, everything settled "
              "down again. Fine.")
    _, p_u, s_u, _ = _prepare(uniform)
    _, p_v, s_v, _ = _prepare(varied)
    assert syntax.structure_features(s_u, p_u)["burstiness"] < \
        syntax.structure_features(s_v, p_v)["burstiness"]


def test_repeated_openings_detected():
    text = " ".join(["Furthermore, the model performs well on this task."] * 8)
    _, paragraphs, sentences, _ = _prepare(text)
    features = syntax.structure_features(sentences, paragraphs)
    assert features["sentence_opening_similarity"] > 0.8
    assert features["template_sentence_ratio"] > 0.8


def test_pos_source_is_reported():
    result = syntax.pos_features(HUMAN_LIKE)
    assert result["source"] in ("spacy", "lexicon_fallback")
    assert result["features"]


def test_lexicon_fallback_omits_what_it_cannot_measure():
    """The fallback must not invent adjective density or passive voice."""
    features = syntax._lexicon_pos(HUMAN_LIKE)
    assert "adjective_ratio" not in features
    assert "past_tense_ratio" not in features
    assert "pronoun_ratio" in features


# --------------------------------------------------------------------------
# punctuation & discourse
# --------------------------------------------------------------------------


def test_punctuation_shares_sum_to_one():
    _, _, sentences, words = _prepare(HUMAN_LIKE)
    features = punctuation.extract(HUMAN_LIKE, sentences, len(words))
    shares = [v for k, v in features.items() if k.endswith("_share")]
    assert abs(sum(shares) - 1.0) < 1e-6


def test_terminal_ratios_sum_to_one():
    _, _, sentences, words = _prepare(AI_LIKE)
    features = punctuation.extract(AI_LIKE, sentences, len(words))
    total = (features["terminal_period_ratio"] + features["terminal_question_ratio"]
             + features["terminal_exclamation_ratio"] + features["terminal_other_ratio"])
    assert abs(total - 1.0) < 1e-6


def test_transition_density_measured_not_scored():
    _, _, sentences, words = _prepare(AI_LIKE)
    features = discourse.extract(AI_LIKE, sentences, len(words))
    assert features["transition_total_per_100_words"] > 0
    # the module must expose densities, never anything named like a verdict
    assert not any("score" in k and "framing" not in k for k in features)


def test_marker_inventory_quotes_real_evidence():
    inventory = discourse.marker_inventory(AI_LIKE)
    assert "transition_additive" in inventory
    assert all(marker.lower() in AI_LIKE.lower()
               for markers in inventory.values() for marker in markers)


def test_specificity_index_rewards_concrete_detail():
    vague = ("Many organisations are exploring various approaches to improve "
             "outcomes in a range of different contexts. " * 10)
    concrete = ("In 2023 Ofsted inspected 1,842 schools in England and found "
                "that 14% required improvement, according to Amanda Spielman. " * 10)
    _, _, s_v, w_v = _prepare(vague)
    _, _, s_c, w_c = _prepare(concrete)
    v = discourse.extract(vague, s_v, len(w_v))["specificity_index"]
    c = discourse.extract(concrete, s_c, len(w_c))["specificity_index"]
    assert c > v


# --------------------------------------------------------------------------
# repetition
# --------------------------------------------------------------------------


def test_ngram_repetition_detects_duplication():
    unique = HUMAN_LIKE
    duplicated = (AI_LIKE.split("\n\n")[0] + "\n\n") * 5
    _, p_u, s_u, _ = _prepare(unique)
    _, p_d, s_d, _ = _prepare(duplicated)
    u = repetition.extract(unique, s_u, p_u)
    d = repetition.extract(duplicated, s_d, p_d)
    assert d["fourgram_repetition_rate"] > u["fourgram_repetition_rate"]
    assert d["longest_repeated_span"] > u["longest_repeated_span"]
    assert d["repeated_span_coverage"] > u["repeated_span_coverage"]


# --------------------------------------------------------------------------
# vectorizer contract
# --------------------------------------------------------------------------


def test_vector_is_namespaced_and_flat():
    vector, metadata = vectorizer.vector_from_text(HUMAN_LIKE)
    assert len(vector) > 120
    assert all("__" in key for key in vector)
    assert all(isinstance(v, float) for v in vector.values())
    assert metadata["pos_source"] in ("spacy", "lexicon_fallback")


def test_to_array_zero_fills_and_reports_coverage():
    vector, _ = vectorizer.vector_from_text(HUMAN_LIKE)
    names = list(vector.keys())[:10] + ["nonexistent__feature_a",
                                        "nonexistent__feature_b"]
    array, coverage = vectorizer.to_array(vector, names)
    assert array.shape == (1, 12)
    assert abs(coverage - (10 / 12)) < 1e-9
    assert array[0][-1] == 0.0


def test_stack_orders_consistently():
    v1, _ = vectorizer.vector_from_text(HUMAN_LIKE)
    v2, _ = vectorizer.vector_from_text(AI_LIKE)
    names = vectorizer.union_feature_names([v1, v2])
    matrix = vectorizer.stack([v1, v2], names)
    assert matrix.shape == (2, len(names))
    assert not any(math.isnan(x) for row in matrix for x in row)
