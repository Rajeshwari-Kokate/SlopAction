"""
Fit the genre / register classifier used by ``app/preprocessing/category.py``.

    python -m training.train_category --data data/corpus.jsonl

Why category detection matters here
-----------------------------------
Writing conventions differ enormously between an academic abstract and a Slack
message.  "Furthermore" is unremarkable in a literature review and conspicuous
in a text message, so the ensemble, the confidence system and the calibrator are
all category-aware.  A wrong category is therefore not a cosmetic error: it
moves the evidence weighting for the whole document.

The artefact contract is different from every other trainer here
-----------------------------------------------------------------
``category._trained_model()`` loads the joblib file and calls it directly::

    probabilities = model.predict_proba([text])[0]
    labels = list(model.classes_)

That is a **bare estimator taking raw strings**, not the ``{"model": ...,
"feature_names": ...}`` dict every other artefact in this project uses.  So this
script deliberately writes a plain scikit-learn ``Pipeline`` of
``TfidfVectorizer -> LogisticRegression`` whose ``classes_`` are category names
from ``config.CATEGORIES``.  Writing the dict artefact here would load without
error and then fail inside the ``try/except`` in ``detect_category``, silently
falling back to the embedding prototypes with no warning anywhere.

Honest limitations
------------------
* Tf-idf over words learns the vocabulary of the corpus, not the genre.  Fitted
  on one publisher's news it will call a different publisher's news "general".
* The labels come from the dataset's ``category`` field.  If those were assigned
  by the same heuristic the fallback uses, this model learns the heuristic, not
  the genre - check where your labels came from before believing the accuracy.
* Categories absent from the training data cannot be predicted at all.  They are
  reported here rather than quietly ignored.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from training import common, support  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_category",
        description="Fit the tf-idf + logistic-regression genre classifier.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(config.ArtifactPaths().category_model))
    parser.add_argument("--algorithm", default="logistic",
                        choices=("logistic", "linear_svc_calibrated"),
                        help="only estimators with predict_proba are allowed, "
                             "because category.py calls it directly")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--report", default=None)
    parser.add_argument("--min-per-category", type=int, default=5,
                        help="categories with fewer training rows than this are "
                             "dropped rather than half-learned")
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--top-terms", type=int, default=10,
                        help="how many indicative terms per category to report")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")
    counts = Counter(r.category for r in records)
    print(f"categories: {dict(counts)}")

    unknown = sorted(set(counts) - set(config.CATEGORIES))
    if unknown:
        raise SystemExit(
            f"dataset contains categories outside config.CATEGORIES: {unknown}. "
            "detect_category projects onto the configured list, so those rows "
            "would train a class the service can never report.")

    ratios = support.parse_ratios(args.ratios)
    splits = support.random_split(records, ratios, args.seed)
    support.print_split(splits)

    train_records = splits["train"]
    train_counts = Counter(r.category for r in train_records)
    usable = {c for c, n in train_counts.items() if n >= args.min_per_category}
    dropped = {c: n for c, n in train_counts.items() if c not in usable}
    if dropped:
        print(f"dropping under-represented categories {dropped} "
              f"(--min-per-category={args.min_per_category})")
    missing = sorted(set(config.CATEGORIES) - usable)
    if missing:
        print(f"NOTE: {missing} have no usable training data. This model can "
              "never predict them; detect_category will simply never return "
              "those names while it is installed.")
    if len(usable) < 2:
        raise SystemExit(
            f"only {len(usable)} category has enough training data; a "
            "one-class genre classifier is not worth installing.")

    train_records = [r for r in train_records if r.category in usable]
    X_train = [r.text for r in train_records]
    y_train = [r.category for r in train_records]

    pipeline = _build_pipeline(args)
    print(f"\nfitting {args.algorithm} on {len(y_train)} documents ...")
    pipeline.fit(X_train, y_train)
    model_classes = [str(c) for c in pipeline.classes_]
    print(f"  classes_: {model_classes}")
    print(f"  vocabulary: {len(pipeline.named_steps['tfidf'].vocabulary_)} terms")

    evaluations: Dict[str, Dict[str, Any]] = {}
    for split_name in ("validation", "test"):
        metrics = _evaluate(pipeline, model_classes, splits[split_name],
                            split_name)
        if metrics is not None:
            evaluations[split_name] = metrics

    indicative = _indicative_terms(pipeline, model_classes, args.top_terms)
    print("\n=== most indicative terms per category ===")
    for category, terms in indicative.items():
        print(f"  {category:14s} {', '.join(terms)}")

    if args.dry_run:
        print("\n--dry-run: artefact not written")
    else:
        import joblib

        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # a bare Pipeline, not the dict artefact - see the module docstring
        joblib.dump(pipeline, destination)
        print(f"\npipeline written to {destination}")
        print("detect_category will now report method='trained'.")

    support.write_report(args.report, {
        "component": "category",
        "dataset": str(data_path),
        "algorithm": args.algorithm,
        "seed": args.seed,
        "artefact_format": "bare sklearn Pipeline (predict_proba([text]))",
        "classes": model_classes,
        "categories_without_training_data": missing,
        "dropped_categories": dropped,
        "split": support.describe_split(splits),
        "metrics": evaluations,
        "indicative_terms": indicative,
        "artefact": None if args.dry_run else str(args.out),
    })
    return 0


def _build_pipeline(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vectoriser = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, strip_accents="unicode",
        ngram_range=(1, max(1, args.ngram_max)), min_df=1, max_df=0.95,
        max_features=args.max_features)

    if args.algorithm == "linear_svc_calibrated":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        classifier = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=args.seed), cv=3)
    else:
        classifier = LogisticRegression(
            max_iter=2000, C=4.0, class_weight="balanced",
            random_state=args.seed)

    return Pipeline([("tfidf", vectoriser), ("clf", classifier)])


def _evaluate(pipeline, model_classes: List[str], subset,
              split_name: str) -> Optional[Dict[str, Any]]:
    known = [r for r in subset if r.category in set(model_classes)]
    skipped = len(subset) - len(known)
    if not known:
        print(f"\n=== {split_name} === no rows with a trainable category")
        return None
    probabilities = support.project_probabilities(
        pipeline.predict_proba([r.text for r in known]),
        model_classes, model_classes)
    metrics = common.evaluate_predictions(
        [r.category for r in known], probabilities, model_classes)
    if skipped:
        metrics["rows_skipped_unknown_category"] = skipped
    common.print_metrics(f"category / {split_name}", metrics)
    if skipped:
        print(f"  ({skipped} row(s) skipped: their category is not in the "
              "model's class space)")
    return metrics


def _indicative_terms(pipeline, model_classes: List[str],
                      limit: int) -> Dict[str, List[str]]:
    """Highest-weight tf-idf terms per class - a sanity check, not evidence."""
    import numpy as np

    classifier = pipeline.named_steps["clf"]
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is None:
        return {}
    terms = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    coefficients = np.asarray(coefficients, dtype="float64")
    if coefficients.shape[0] == 1 and len(model_classes) == 2:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])
    output: Dict[str, List[str]] = {}
    for index, category in enumerate(model_classes):
        if index >= coefficients.shape[0]:
            break
        order = np.argsort(-coefficients[index])[:limit]
        output[category] = [str(t) for t in terms[order]]
    return output


if __name__ == "__main__":
    raise SystemExit(main())
