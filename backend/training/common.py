"""
Shared training utilities.

Everything the trainers agree on lives here: the dataset record schema, JSONL
I/O, the artefact format that the runtime loaders expect, and the metric suite.

The artefact contract
---------------------
Every trained model is saved as a dict with **at minimum**::

    {
        "model":         <fitted estimator with predict_proba>,
        "feature_names": [ordered list of feature keys],
        "algorithm":     "logistic_regression" | "random_forest" | ...,
        "classes":       ["human", "pure_ai", "humanized_ai"],
        "trained_on":    {"dataset": ..., "n_samples": ..., "date": ...},
        "metrics":       {validation metrics dict},
    }

``feature_names`` is not optional.  Feature order is the single most common
source of silent corruption when a model trained offline meets a service whose
feature set has since changed; storing the names lets
``features.vectorizer.to_array`` project safely and report coverage.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# make `app` importable when the training scripts are run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402

CLASSES: Tuple[str, ...] = config.CLASSES

#: every field a dataset row may carry
RECORD_FIELDS = (
    "text", "label", "source", "generator_model", "category", "humanizer",
    "language", "attack_type", "topic", "id", "split", "length_bucket",
)


@dataclass
class Record:
    """One dataset row.

    ``label`` must be one of ``human``, ``pure_ai``, ``humanized_ai``.

    The metadata fields are not decoration: ``generator_model``, ``humanizer``
    and ``category`` are what make model-held-out, humanizer-held-out and
    domain-held-out evaluation possible, and those are the only evaluations
    that tell you whether the detector generalises.
    """

    text: str
    label: str
    source: str = "unknown"
    generator_model: Optional[str] = None
    category: str = "general"
    humanizer: Optional[str] = None
    language: str = "en"
    attack_type: Optional[str] = None
    topic: Optional[str] = None
    id: Optional[str] = None
    split: Optional[str] = None
    length_bucket: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label not in CLASSES:
            raise ValueError(
                f"label must be one of {CLASSES}, got {self.label!r}")
        if self.length_bucket is None:
            self.length_bucket = length_bucket(len(self.text.split()))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def length_bucket(word_count: int) -> str:
    if word_count < 120:
        return "short"
    if word_count < 350:
        return "medium"
    return "long"


# --------------------------------------------------------------------------
# JSONL I/O
# --------------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[Record | Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if isinstance(record, Record) else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Record]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            payload = {k: v for k, v in payload.items() if k in RECORD_FIELDS}
            try:
                records.append(Record(**payload))
            except TypeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def dataset_summary(records: Sequence[Record]) -> Dict[str, Any]:
    from collections import Counter

    return {
        "n_samples": len(records),
        "labels": dict(Counter(r.label for r in records)),
        "categories": dict(Counter(r.category for r in records)),
        "generators": dict(Counter(r.generator_model or "-" for r in records)),
        "humanizers": dict(Counter(r.humanizer or "-" for r in records)),
        "attack_types": dict(Counter(r.attack_type or "-" for r in records)),
        "length_buckets": dict(Counter(r.length_bucket or "-" for r in records)),
        "languages": dict(Counter(r.language for r in records)),
        "topics": len({r.topic for r in records if r.topic}),
    }


# --------------------------------------------------------------------------
# artefact I/O
# --------------------------------------------------------------------------


def save_artefact(path: Path, payload: Dict[str, Any]) -> Path:
    import joblib

    required = {"model", "feature_names", "algorithm"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"artefact is missing required fields: {sorted(missing)}")
    payload.setdefault("classes", list(CLASSES))
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def describe_artefact(path: Path) -> Dict[str, Any]:
    import joblib

    payload = joblib.load(Path(path))
    return {
        "algorithm": payload.get("algorithm"),
        "classes": payload.get("classes"),
        "feature_count": len(payload.get("feature_names", [])),
        "trained_on": payload.get("trained_on"),
        "metrics": payload.get("metrics"),
        "created_at": payload.get("created_at"),
    }


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def build_estimator(algorithm: str, random_state: int = 0):
    """Instantiate one of the supported estimators.

    Logistic regression is the default everywhere because it is interpretable,
    it calibrates well and it does not pretend to have learned interactions that
    a small dataset cannot support.  Tree ensembles are available for when the
    dataset is large enough to justify them.
    """
    algorithm = algorithm.lower()
    if algorithm in ("logistic", "logistic_regression", "logreg"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0,
                                       class_weight="balanced",
                                       random_state=random_state)),
        ]), "logistic_regression"

    if algorithm in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=random_state), "random_forest"

    if algorithm in ("gb", "gradient_boosting", "hgb"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06,
            random_state=random_state), "hist_gradient_boosting"

    if algorithm in ("lgbm", "lightgbm"):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "lightgbm is not installed. pip install lightgbm, or choose "
                "--algorithm logistic/rf/gb.") from exc
        return LGBMClassifier(n_estimators=600, learning_rate=0.05,
                              num_leaves=31, class_weight="balanced",
                              random_state=random_state, verbose=-1), "lightgbm"

    if algorithm in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "xgboost is not installed. pip install xgboost, or choose "
                "--algorithm logistic/rf/gb.") from exc
        return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=random_state,
                             eval_metric="mlogloss"), "xgboost"

    if algorithm in ("mlp",):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                                  early_stopping=True, random_state=random_state)),
        ]), "mlp"

    raise SystemExit(
        f"unknown algorithm '{algorithm}'. Choose from: logistic, rf, gb, "
        "lgbm, xgb, mlp")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def evaluate_predictions(y_true: Sequence[str], probabilities,
                         classes: Sequence[str]) -> Dict[str, Any]:
    """The full metric suite.

    Overall accuracy is reported last and deliberately without emphasis: on a
    three-class problem with an unbalanced test set it is the least informative
    number available.
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    from app.ensemble.calibrator import brier_score, expected_calibration_error

    probabilities = np.asarray(probabilities, dtype="float64")
    index = {c: i for i, c in enumerate(classes)}
    y_true = [str(y) for y in y_true]
    y_index = np.asarray([index[y] for y in y_true])
    y_pred_index = probabilities.argmax(axis=1)
    y_pred = [classes[i] for i in y_pred_index]

    metrics: Dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(classes)).tolist(),
        "confusion_matrix_labels": list(classes),
        "brier_score": brier_score(probabilities, y_true, classes),
        "expected_calibration_error": expected_calibration_error(
            probabilities, y_true, classes),
    }

    for klass in classes:
        binary_true = (np.asarray(y_true) == klass).astype(int)
        column = probabilities[:, index[klass]]
        entry: Dict[str, Any] = {
            "support": int(binary_true.sum()),
            "precision": float(precision_score(y_true, y_pred, labels=[klass],
                                               average="micro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=[klass],
                                         average="micro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=[klass],
                                 average="micro", zero_division=0)),
        }
        if 0 < binary_true.sum() < len(binary_true):
            entry["auroc"] = float(roc_auc_score(binary_true, column))
            entry["auprc"] = float(average_precision_score(binary_true, column))
        metrics["per_class"][klass] = entry

    # ---- the operationally important binary view --------------------------
    ai_column = sum(probabilities[:, index[c]]
                    for c in ("pure_ai", "humanized_ai") if c in index)
    binary_true = np.asarray([0 if y == "human" else 1 for y in y_true])
    if 0 < binary_true.sum() < len(binary_true):
        metrics["binary_ai_vs_human"] = {
            "auroc": float(roc_auc_score(binary_true, ai_column)),
            "auprc": float(average_precision_score(binary_true, ai_column)),
            "tpr_at_1pct_fpr": tpr_at_fpr(binary_true, ai_column, 0.01),
            "tpr_at_5pct_fpr": tpr_at_fpr(binary_true, ai_column, 0.05),
            "false_positive_rate_at_50": float(
                ((ai_column >= 0.5) & (binary_true == 0)).sum()
                / max(1, (binary_true == 0).sum())),
            "false_negative_rate_at_50": float(
                ((ai_column < 0.5) & (binary_true == 1)).sum()
                / max(1, (binary_true == 1).sum())),
        }
    return metrics


def tpr_at_fpr(binary_true, scores, target_fpr: float) -> float:
    """True-positive rate at a fixed false-positive rate.

    This is the number that matters for deployment.  Accusing 5% of honest
    writers is not acceptable, so the useful question is "how many machine texts
    do we catch if we only allow a 1% false-positive rate", not "what is the
    accuracy at an arbitrary 0.5 threshold".
    """
    import numpy as np
    from sklearn.metrics import roc_curve

    binary_true = np.asarray(binary_true)
    if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(binary_true, np.asarray(scores))
    eligible = tpr[fpr <= target_fpr]
    return float(eligible.max()) if eligible.size else 0.0


# --------------------------------------------------------------------------
# feature extraction helpers
# --------------------------------------------------------------------------


def extract_stylometric_vectors(records: Sequence[Record],
                                verbose: bool = True) -> Tuple[List[Dict[str, float]], List[str]]:
    from app.features import vectorizer

    vectors: List[Dict[str, float]] = []
    for index, record in enumerate(records):
        if verbose and index and index % 100 == 0:
            print(f"  ... {index}/{len(records)} feature vectors", flush=True)
        vector, _ = vectorizer.vector_from_text(record.text)
        vectors.append(vector)
    names = vectorizer.union_feature_names(vectors)
    return vectors, names


def extract_humanization_vectors(records: Sequence[Record],
                                 verbose: bool = True) -> Tuple[List[Dict[str, float]], List[str]]:
    """Run the humanization engine's three-view extractor over a dataset.

    Uses the *same* code path as the service, via a minimal analysis context, so
    the trained model sees exactly the features it will see at inference time.
    """
    from app.detectors.humanization_detector import HumanizationDetector
    from app.features import vectorizer

    detector = HumanizationDetector()
    vectors: List[Dict[str, float]] = []
    for index, record in enumerate(records):
        if verbose and index and index % 50 == 0:
            print(f"  ... {index}/{len(records)} humanization vectors", flush=True)
        context = build_context(record.text)
        original = detector._original_view(context)
        content = detector._content_view(context)
        expression = detector._expression_view(context)
        divergence = detector._divergence(original, content, expression)
        vector: Dict[str, float] = {}
        vector.update({f"orig__{k}": v for k, v in original.items()})
        vector.update({f"cont__{k}": v for k, v in content.items()})
        vector.update({f"expr__{k}": v for k, v in expression.items()})
        vector.update({f"div__{k}": v for k, v in divergence.items()})
        vectors.append(vector)
    return vectors, vectorizer.union_feature_names(vectors)


def build_context(text: str, mode: str = "standard"):
    """Construct an AnalysisContext outside the request path."""
    from app.core.context import AnalysisContext
    from app.preprocessing import category as category_module
    from app.preprocessing import chunker, cleaner
    from app.preprocessing import language as language_module
    from app.preprocessing import tokenizer as text_tokenizer

    normalised = cleaner.normalise(cleaner.validate(text))
    paragraphs, sentences = text_tokenizer.segment(normalised.text)
    chunks, exact = chunker.build_chunks(normalised.text, sentences)
    return AnalysisContext(
        text=normalised.text,
        original=normalised.original,
        statistics=normalised.statistics,
        style_signals=normalised.style_signals,
        paragraphs=paragraphs,
        sentences=sentences,
        chunks=chunks,
        words=text_tokenizer.words(normalised.text),
        language=language_module.detect_language(normalised.text),
        category=category_module.detect_category(normalised.text,
                                                 normalised.statistics),
        mode=mode,
        reliability_band=config.length_band(
            int(normalised.statistics.get("words", 0))).reliability,
        exact_tokens=exact,
    )


def stamp(dataset_path: Path, records: Sequence[Record]) -> Dict[str, Any]:
    return {
        "dataset": str(dataset_path),
        "n_samples": len(records),
        "date": datetime.now(timezone.utc).isoformat(),
        "summary": dataset_summary(records),
    }


def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"  n              : {metrics.get('n')}")
    print(f"  macro F1       : {metrics.get('macro_f1', float('nan')):.4f}")
    binary = metrics.get("binary_ai_vs_human") or {}
    if binary:
        print(f"  AI-vs-human AUROC        : {binary.get('auroc', float('nan')):.4f}")
        print(f"  TPR @ 1% FPR             : {binary.get('tpr_at_1pct_fpr', float('nan')):.4f}")
        print(f"  TPR @ 5% FPR             : {binary.get('tpr_at_5pct_fpr', float('nan')):.4f}")
    print(f"  Brier          : {metrics.get('brier_score', float('nan')):.4f}")
    print(f"  ECE            : {metrics.get('expected_calibration_error', float('nan')):.4f}")
    for klass, entry in (metrics.get("per_class") or {}).items():
        print(f"  {klass:14s} support={entry['support']:5d} "
              f"P={entry['precision']:.3f} R={entry['recall']:.3f} "
              f"F1={entry['f1']:.3f} AUROC={entry.get('auroc', float('nan')):.3f}")
    print(f"  accuracy       : {metrics.get('accuracy', float('nan')):.4f} "
          "(reported last on purpose - it is the least informative number here)")
