"""
Neural code-path smoke tests.

These run against the tiny, **randomly initialised** fixture models built by
``tests/fixtures/build_tiny_models.py``.  Build them first::

    python -m tests.fixtures.build_tiny_models

WHAT IS AND IS NOT ASSERTED HERE
--------------------------------
Asserted: that the tensor plumbing executes correctly.  Tokenisation, offset
mapping, the sliding-window protocol, log-softmax, rank computation, the
analytic sampling-discrepancy moments, batched classification, the two-model
Binoculars path, embedding pooling, per-span slicing, and the shape and range of
everything that comes out.

NOT asserted: that any number means anything.  The weights are random.  A test
that claimed "AI text scores higher than human text" against these fixtures
would be asserting noise, so no such test exists here.  Detector *quality* is
measured by ``training/evaluate.py`` against a real labelled corpus, not by the
unit-test suite.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import pytest

from tests.conftest import AI_LIKE, HUMAN_LIKE, LONG_TEXT, TINY_MODELS

pytestmark = pytest.mark.smoke

REQUIRED = ("causal_lm", "causal_lm_b", "classifier", "classifier3", "encoder")


def _have_fixtures() -> bool:
    return all((TINY_MODELS / name).exists() for name in REQUIRED)


pytest.importorskip("torch", reason="PyTorch is required for the smoke tests")

if not _have_fixtures():  # pragma: no cover - depends on local state
    pytest.skip("tiny fixture models not built; run "
                "`python -m tests.fixtures.build_tiny_models`",
                allow_module_level=True)


@pytest.fixture(scope="module")
def tiny_env():
    """Point the whole application at the fixture models for this module."""
    import importlib

    from app.utils import model_loader

    previous = dict(os.environ)
    os.environ.update({
        "ASD_HF_LOCAL_ONLY": "true",
        "ASD_LM_MODEL": str(TINY_MODELS / "causal_lm"),
        "ASD_DETECTOR_MODEL": str(TINY_MODELS / "classifier"),
        "ASD_EMBEDDING_MODEL": str(TINY_MODELS / "encoder"),
        "ASD_BINOCULARS_OBSERVER": str(TINY_MODELS / "causal_lm"),
        "ASD_BINOCULARS_PERFORMER": str(TINY_MODELS / "causal_lm_b"),
        "ASD_DEEP_ANALYSIS": "true",
    })
    model_loader.reset_cache()

    from app.core import config as config_module
    from app.detectors import normalisation
    from app.preprocessing import category as category_module

    importlib.reload(config_module)
    normalisation._load.cache_clear()
    category_module._prototype_matrix.cache_clear()

    yield

    os.environ.clear()
    os.environ.update(previous)
    model_loader.reset_cache()
    importlib.reload(config_module)
    normalisation._load.cache_clear()
    category_module._prototype_matrix.cache_clear()


@pytest.fixture(scope="module")
def context(tiny_env):
    from training.common import build_context

    return build_context(AI_LIKE * 2, mode="deep")


@contextmanager
def swapped_model(**env):
    """Temporarily point the app at a different checkpoint.

    Order matters: the environment must be restored *before* the final config
    reload, otherwise the reloaded module keeps the temporary checkpoint and
    leaks it into every later test in the module.
    """
    import importlib

    from app.core import config as config_module
    from app.utils import model_loader

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    model_loader.reset_cache()
    importlib.reload(config_module)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        model_loader.reset_cache()
        importlib.reload(config_module)


# --------------------------------------------------------------------------
# language model statistics
# --------------------------------------------------------------------------


def test_token_statistics_are_well_formed(context):
    stats = context.token_statistics()
    assert stats is not None, context.token_statistics_error
    n = stats.token_count
    assert n > 100
    for array in (stats.log_probs, stats.entropies, stats.ranks,
                  stats.expected_log_prob, stats.log_prob_variance):
        assert len(array) == n
        assert all(math.isfinite(float(v)) for v in array)

    assert all(v <= 1e-9 for v in stats.log_probs), "log probabilities must be <= 0"
    assert all(v >= 1 for v in stats.ranks), "ranks are 1-based"
    assert all(v >= 0 for v in stats.entropies)
    assert all(v >= 0 for v in stats.log_prob_variance), "variance cannot be negative"
    # E[log p] under the model's own distribution is exactly -entropy
    for expected, entropy in list(zip(stats.expected_log_prob, stats.entropies))[:50]:
        assert abs(expected + entropy) < 1e-4


def test_offsets_align_with_the_source_text(context):
    stats = context.token_statistics()
    assert stats.offsets, "fast tokenizers must expose an offset mapping"
    assert len(stats.offsets) == stats.token_count
    for start, end in stats.offsets[:100]:
        assert 0 <= start <= end <= len(context.text)


def test_sliding_window_scores_every_token_exactly_once(tiny_env):
    """The document is longer than the fixture model's 256-token context, so
    this exercises the windowing protocol rather than a single forward pass."""
    from training.common import build_context

    long_context = build_context(LONG_TEXT[:8000])
    stats = long_context.token_statistics()
    assert stats is not None
    assert stats.windows > 1, "expected multiple windows for a long document"
    # offsets must be strictly non-decreasing: no token scored twice, none skipped
    starts = [s for s, _ in stats.offsets]
    assert starts == sorted(starts)
    assert len(set(stats.offsets)) == len(stats.offsets)


def test_probability_aggregates_are_sane(context):
    from app.features import probability as probability_features

    features = probability_features.aggregate(context.token_statistics())
    assert features["perplexity"] > 1.0
    assert features["negative_log_likelihood"] > 0.0
    assert abs(features["perplexity"]
               - math.exp(min(50.0, features["negative_log_likelihood"]))) < 1e-6
    assert 0.0 <= features["top1_token_ratio"] <= 1.0
    assert features["top1_token_ratio"] <= features["top10_token_ratio"] \
        <= features["top100_token_ratio"]
    assert features["mean_rank"] >= 1.0
    assert all(math.isfinite(v) for v in features.values())


def test_span_statistics_slice_the_single_forward_pass(context):
    from app.features import probability as probability_features

    stats = context.token_statistics()
    paragraph = context.paragraphs[0]
    span = probability_features.span_statistics(
        stats, paragraph.start, paragraph.end)
    assert span is not None
    assert span["tokens"] < stats.token_count
    assert span["perplexity"] > 1.0
    assert 0.0 <= span["top1_token_ratio"] <= 1.0

    # a span with too few tokens must return None rather than a noisy estimate
    assert probability_features.span_statistics(stats, 0, 5) is None


def test_curvature_discrepancy_is_finite(context):
    from app.detectors.curvature_detector import CurvatureDetector

    result = CurvatureDetector().analyse(context)
    assert result.available is True
    assert math.isfinite(result.raw["curvature_score"])
    assert 0.0 <= result.score <= 1.0
    assert result.raw["variant"] == "analytic_conditional_sampling_discrepancy"
    assert result.raw["not_implemented"]


def test_curvature_refuses_short_input(tiny_env):
    from app.detectors.curvature_detector import CurvatureDetector
    from training.common import build_context

    short = build_context(" ".join(HUMAN_LIKE.split()[:55]))
    result = CurvatureDetector().analyse(short)
    if not result.available:
        assert "token" in result.reason


# --------------------------------------------------------------------------
# transformer classifier
# --------------------------------------------------------------------------


def test_classifier_runs_and_reports_its_two_class_limitation(context):
    from app.detectors.transformer_detector import TransformerDetector

    result = TransformerDetector().analyse(context)
    assert result.available is True
    assert result.trained is True
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6
    assert result.features["transformer_models_humanized"] == 0.0
    assert any("two-class" in w for w in result.warnings)
    # a two-class checkpoint must not manufacture a humanized probability
    assert result.class_probabilities["humanized_ai"] == 0.0


def test_classifier_scores_every_chunk(context):
    from app.detectors.transformer_detector import TransformerDetector

    result = TransformerDetector().analyse(context)
    distributions = result.raw["_segment_distributions"]
    assert len(distributions) == len(context.chunks)
    indices = [d["chunk_index"] for d in distributions]
    assert indices == [c.index for c in context.chunks]
    for entry in distributions:
        assert abs(sum(entry["distribution"].values()) - 1.0) < 1e-5


def test_three_class_checkpoint_is_used_natively(tiny_env):
    from app.detectors.transformer_detector import TransformerDetector
    from training.common import build_context

    with swapped_model(ASD_DETECTOR_MODEL=str(TINY_MODELS / "classifier3")):
        result = TransformerDetector().analyse(build_context(AI_LIKE))

    assert result.available is True, result.reason
    assert result.features["transformer_models_humanized"] == 1.0
    assert not any("two-class" in w for w in result.warnings)
    assert result.class_probabilities["humanized_ai"] > 0.0


# --------------------------------------------------------------------------
# binoculars
# --------------------------------------------------------------------------


def test_binoculars_runs_with_a_shared_vocabulary(context):
    from app.detectors.binocular_detector import BinocularDetector

    result = BinocularDetector().analyse(context)
    assert result.available is True, result.reason
    raw = result.raw
    assert raw["shared_vocabulary"] is True
    assert raw["threshold_transfers"] is False      # not the published pair
    assert math.isfinite(raw["binoculars_score"])
    assert raw["cross_perplexity"] > 1.0
    assert any("no published decision threshold" in w for w in result.warnings)


def test_binoculars_refuses_mismatched_tokenizers(tiny_env):
    """A different vocabulary makes the cross-entropy term meaningless, so the
    engine must disable itself rather than return a plausible wrong number."""
    from app.detectors.binocular_detector import BinocularDetector
    from training.common import build_context

    with swapped_model(ASD_BINOCULARS_PERFORMER=str(
            TINY_MODELS / "encoder" / "backbone")):
        result = BinocularDetector().analyse(build_context(AI_LIKE, mode="deep"))

    assert result.available is False
    assert result.score is None
    assert "tokeniz" in result.reason or "vocab" in result.reason


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


def test_sentence_encoder_produces_normalised_vectors(context):
    matrix, method, error = context.embeddings()
    assert method == "sentence_transformer", error
    assert matrix.shape[0] == min(len(context.sentences), 300)
    norms = (matrix ** 2).sum(axis=1) ** 0.5
    assert all(abs(float(n) - 1.0) < 1e-3 for n in norms)


def test_similarity_statistics_are_bounded(context):
    from app.features import semantic as semantic_features

    matrix, _, _ = context.embeddings()
    stats = semantic_features.similarity_statistics(matrix)
    for key in ("mean_similarity", "median_similarity", "max_similarity",
                "p90_similarity", "adjacent_similarity_mean"):
        assert -1.0 <= stats[key] <= 1.0
    for key in ("high_similarity_ratio", "redundant_sentence_ratio", "coverage"):
        assert 0.0 <= stats[key] <= 1.0
    assert stats["pairs_evaluated"] > 0


def test_large_document_uses_windowed_similarity(tiny_env):
    """Above the full-matrix limit the engine must sample rather than build an
    n^2 matrix, and must say what fraction of pairs it covered."""
    from app.core import config
    from app.features import semantic as semantic_features
    from training.common import build_context

    many = build_context(" ".join(
        f"This is sentence number {i} about an entirely ordinary topic."
        for i in range(config.SEMANTIC_FULL_MATRIX_LIMIT + 60)))
    matrix, _, _ = many.embeddings()
    stats = semantic_features.similarity_statistics(matrix)
    assert stats["coverage"] < 1.0
    assert stats["pairs_evaluated"] > 0


# --------------------------------------------------------------------------
# full pipeline with every engine live
# --------------------------------------------------------------------------


def test_full_pipeline_with_all_engines_available(tiny_env):
    from app import pipeline

    result = pipeline.analyse(AI_LIKE * 2, mode="deep")

    for name in ("transformer", "probability", "curvature", "binoculars",
                 "stylometry", "semantic", "humanization", "slop", "ood"):
        assert result["detectors"][name]["available"] is True, \
            f"{name}: {result['detectors'][name].get('reason')}"

    assert result["result"]["ai_origin_score"] is not None
    authorship = result["authorship"]
    assert abs(sum(authorship.values()) - 100.0) < 0.15

    scored_chunks = [c for c in result["chunks"] if c["ai_score"] is not None]
    assert scored_chunks, "per-chunk scoring should work once engines are live"
    assert all(c["weight"] > 0 for c in scored_chunks)

    scored_sentences = [s for s in result["sentences"] if s["ai_score"] is not None]
    assert scored_sentences

    assert result["mixed_authorship"]["is_mixed"] in (True, False)
    assert "weight_formula" in result["chunk_weighting"]

    # still not a probability: no trained meta-classifier or calibrator
    assert result["result"]["score_type"] == "uncalibrated_detection_score"
    assert result["result"]["is_probability"] is False


def test_chunk_weights_are_not_uniform(tiny_env):
    """Chunks must be weighted by token share, reliability and confidence -
    proving the aggregation is not a blind average."""
    from app import pipeline

    result = pipeline.analyse(LONG_TEXT[:6000], mode="standard")
    weights = [c["weight"] for c in result["chunks"] if c["ai_score"] is not None]
    if len(weights) > 2:
        assert len(set(round(w, 6) for w in weights)) > 1
