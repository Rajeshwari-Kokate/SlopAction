"""Preprocessing: validation, normalisation, segmentation, chunking, language,
category."""

from __future__ import annotations

import pytest

from app.core import config
from app.preprocessing import category as category_module
from app.preprocessing import chunker, cleaner
from app.preprocessing import language as language_module
from app.preprocessing import tokenizer as text_tokenizer
from tests.conftest import (AI_LIKE, CODE_TEXT, EMOJI_TEXT, HUMAN_LIKE,
                            NON_ENGLISH, ONE_SENTENCE, SPECIAL_CHARS, URL_TEXT)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\n\t  \n"])
def test_empty_input_rejected(bad):
    with pytest.raises(cleaner.ValidationError) as exc:
        cleaner.validate(bad)
    assert exc.value.code == "empty_input"


def test_none_input_rejected():
    with pytest.raises(cleaner.ValidationError) as exc:
        cleaner.validate(None)
    assert exc.value.code == "empty_input"


def test_non_string_rejected():
    with pytest.raises(cleaner.ValidationError) as exc:
        cleaner.validate(12345)
    assert exc.value.code == "invalid_type"


def test_oversized_input_rejected():
    with pytest.raises(cleaner.ValidationError) as exc:
        cleaner.validate("word " * (config.MAX_INPUT_WORDS + 50))
    assert exc.value.code == "too_long"


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_normalisation_preserves_style_before_cleaning():
    text = 'She said “hello” — then left…  Twice.  Really.'
    result = cleaner.normalise(text)
    assert result.style_signals["smart_double_quote_count"] == 2
    assert result.style_signals["em_dash_count"] == 1
    assert result.style_signals["unicode_ellipsis_count"] == 1
    # double space after a full stop is a real authorship habit and must be
    # measured, not silently normalised away without record
    assert result.style_signals["double_space_after_period"] >= 1


def test_zero_width_characters_removed_and_reported():
    text = "This is a​ test of invisible‍ characters in a sentence." * 3
    result = cleaner.normalise(text)
    assert "​" not in result.text
    assert result.style_signals["zero_width_count"] == 6
    assert any("zero-width" in note for note in result.notes)


def test_crlf_normalised():
    result = cleaner.normalise("one\r\ntwo\r\n\r\nthree")
    assert "\r" not in result.text
    assert result.style_signals["crlf_count"] == 3


def test_statistics_are_consistent():
    stats = cleaner.normalise(HUMAN_LIKE).statistics
    assert stats["words"] > 0
    assert stats["sentences"] > 0
    assert stats["paragraphs"] > 0
    assert stats["unique_words"] <= stats["words"]
    assert 0.0 <= stats["whitespace_ratio"] <= 1.0
    assert stats["characters_no_spaces"] <= stats["characters"]


def test_url_and_emoji_counted():
    assert cleaner.normalise(URL_TEXT).statistics["url_count"] >= 8
    assert cleaner.normalise(URL_TEXT).statistics["email_count"] >= 8
    assert cleaner.normalise(EMOJI_TEXT).statistics["emoji_count"] > 10


def test_special_characters_do_not_break_statistics():
    stats = cleaner.normalise(SPECIAL_CHARS).statistics
    assert stats["words"] > 0
    assert stats["non_ascii_ratio"] > 0.0


def test_code_fences_counted():
    assert cleaner.normalise(CODE_TEXT).statistics["code_fence_count"] >= 1


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------


def test_segmentation_offsets_are_valid():
    normalised = cleaner.normalise(HUMAN_LIKE)
    paragraphs, sentences = text_tokenizer.segment(normalised.text)
    for sentence in sentences:
        assert normalised.text[sentence.start:sentence.end] == sentence.text
    for paragraph in paragraphs:
        assert normalised.text[paragraph.start:paragraph.end] == paragraph.text


def test_every_sentence_belongs_to_a_paragraph():
    normalised = cleaner.normalise(AI_LIKE)
    paragraphs, sentences = text_tokenizer.segment(normalised.text)
    indices = {p.index for p in paragraphs}
    assert all(s.paragraph_index in indices for s in sentences)
    linked = sum(len(p.sentence_indices) for p in paragraphs)
    assert linked == len(sentences)


def test_abbreviations_do_not_split_sentences():
    text = ("Dr. Smith met Mr. Jones at 3 p.m. on Tuesday. They discussed the "
            "e.g. clause and the i.e. clause at length. It went well.")
    sentences = text_tokenizer.split_sentences(text, use_spacy=False)
    assert len(sentences) == 3


def test_single_sentence_input():
    sentences = text_tokenizer.split_sentences(ONE_SENTENCE)
    assert len(sentences) == 1
    assert sentences[0].word_count > 15


def test_list_block_split_per_item():
    text = "- first item here\n- second item here\n- third item here"
    sentences = text_tokenizer.split_sentences(text, use_spacy=False)
    assert len(sentences) == 3


def test_many_paragraphs():
    text = "\n\n".join(f"Paragraph number {i} contains a complete sentence "
                       f"about topic {i}." for i in range(40))
    paragraphs, sentences = text_tokenizer.segment(text)
    assert len(paragraphs) == 40
    assert len(sentences) == 40


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_chunks_cover_all_sentences():
    normalised = cleaner.normalise(AI_LIKE * 3)
    _, sentences = text_tokenizer.segment(normalised.text)
    chunks, _ = chunker.build_chunks(normalised.text, sentences, chunk_size=80,
                                     overlap=20)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.sentence_indices)
    assert covered == set(range(len(sentences)))


def test_chunks_overlap():
    normalised = cleaner.normalise(AI_LIKE * 3)
    _, sentences = text_tokenizer.segment(normalised.text)
    chunks, _ = chunker.build_chunks(normalised.text, sentences, chunk_size=80,
                                     overlap=30)
    if len(chunks) > 1:
        shared = set(chunks[0].sentence_indices) & set(chunks[1].sentence_indices)
        assert shared, "consecutive chunks should share context"


def test_short_text_is_one_chunk():
    normalised = cleaner.normalise(ONE_SENTENCE)
    _, sentences = text_tokenizer.segment(normalised.text)
    chunks, _ = chunker.build_chunks(normalised.text, sentences)
    assert len(chunks) == 1


def test_chunk_text_matches_offsets():
    normalised = cleaner.normalise(AI_LIKE * 2)
    _, sentences = text_tokenizer.segment(normalised.text)
    chunks, _ = chunker.build_chunks(normalised.text, sentences, chunk_size=60,
                                     overlap=15)
    for chunk in chunks:
        assert normalised.text[chunk.start:chunk.end] == chunk.text


def test_no_sentences_yields_no_chunks():
    chunks, _ = chunker.build_chunks("", [])
    assert chunks == []


def test_chunk_cap_respected_without_dropping_text():
    normalised = cleaner.normalise(AI_LIKE * 8)
    _, sentences = text_tokenizer.segment(normalised.text)
    chunks, _ = chunker.build_chunks(normalised.text, sentences, chunk_size=40,
                                     overlap=10, max_chunks=4)
    assert len(chunks) <= 4
    covered = set()
    for chunk in chunks:
        covered.update(chunk.sentence_indices)
    assert covered == set(range(len(sentences)))


# --------------------------------------------------------------------------
# language
# --------------------------------------------------------------------------


def test_english_detected():
    result = language_module.detect_language(HUMAN_LIKE)
    assert result["language"] == "en"
    assert result["is_supported"] is True
    assert 0.0 < result["confidence"] <= 1.0


def test_german_detected_and_unsupported():
    result = language_module.detect_language(NON_ENGLISH)
    assert result["language"] == "de"
    assert result["is_supported"] is False


def test_very_short_input_returns_unknown_not_a_guess():
    result = language_module.detect_language("ok sure")
    assert result["language"] == "unknown"
    assert result["confidence"] == 0.0


def test_code_has_a_weak_language_signal():
    """Source code contains English keywords, so the stop-word profile will
    lean English. That is fine and expected - catching code is the OOD
    detector's job (see test_detectors.py), not the language detector's. What
    matters here is that the confidence stays modest rather than asserting
    fluent English prose."""
    result = language_module.detect_language(CODE_TEXT)
    assert result["confidence"] < 0.95


def test_cyrillic_script_detected():
    text = ("Искусственный интеллект стремительно развивается в последние годы "
            "и меняет многие отрасли экономики и образования.")
    result = language_module.detect_language(text)
    assert result["script"] == "cyrillic"
    assert result["language"] == "ru"


# --------------------------------------------------------------------------
# category
# --------------------------------------------------------------------------


def test_manual_category_is_respected():
    stats = cleaner.normalise(AI_LIKE).statistics
    result = category_module.detect_category(AI_LIKE, stats, manual="academic")
    assert result["name"] == "academic"
    assert result["manual"] is True
    assert result["confidence"] == 1.0


def test_invalid_manual_category_rejected():
    stats = cleaner.normalise(AI_LIKE).statistics
    with pytest.raises(ValueError):
        category_module.detect_category(AI_LIKE, stats, manual="haiku")


def test_auto_category_returns_valid_name_and_distribution():
    stats = cleaner.normalise(AI_LIKE).statistics
    result = category_module.detect_category(AI_LIKE, stats, manual="auto")
    assert result["name"] in config.CATEGORIES
    assert 0.0 <= result["confidence"] <= 1.0
    total = sum(result["distribution"].values())
    assert abs(total - 1.0) < 1e-6


def test_untrained_lexical_prior_confidence_is_capped():
    """An unlearned genre prior must not claim strong confidence."""
    stats = cleaner.normalise(AI_LIKE).statistics
    result = category_module.detect_category(AI_LIKE, stats, manual="auto")
    if result["method"] == "lexical_prior":
        assert result["confidence"] <= 0.55
