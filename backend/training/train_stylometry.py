"""
Fit the stylometric three-class classifier (Engine E).

    python -m training.train_stylometry --data data/corpus.jsonl

What this trains
----------------
``app/features/vectorizer.stylometric_vector`` produces ~200 model-free
measurements per document: sentence-length distribution and burstiness,
vocabulary richness, punctuation habits, POS/syntax shape, discourse and hedging
markers, and n-gram repetition.  This script fits a classifier over exactly that
vector - through ``common.extract_stylometric_vectors``, which calls the same
extraction code the service calls - and writes it to
``models/stylometry/stylometry_clf.joblib``.  Once that file exists,
``StylometryDetector`` stops emitting its labelled rule-based fallback and starts
reporting ``trained: true``.

Honest limitations
------------------
* Stylometry is the most domain-sensitive signal in this system.  A model fitted
  on student essays will mistake press releases for machine output.  The
  per-category slices printed by ``training/evaluate.py`` are the only way to
  see that happening; a single accuracy number will hide it completely.
* The POS block is a strict subset of itself when spaCy is missing.  Fit with
  the same POS backend you will serve with, or the model will meet zero-filled
  columns at inference time.  ``feature_coverage`` in the API response reports
  how badly that went.
* These features describe *style*, and style is imitable.  A humanizer that
  varies sentence length defeats a good part of this vector on purpose, which is
  why Engine G exists separately.

The report (``--report``) carries the top-25 attributions so the fitted model can
be inspected rather than trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.features import vectorizer  # noqa: E402
from training import common, support  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_stylometry",
        description="Fit the stylometric human/pure_ai/humanized_ai classifier.")
    parser.add_argument("--data", required=True,
                        help="labelled dataset in JSONL form")
    parser.add_argument("--out", default=str(config.ArtifactPaths().stylometry_model),
                        help="artefact path (default: the configured model path)")
    parser.add_argument("--algorithm", default="logistic",
                        help="logistic | rf | gb | lgbm | xgb | mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.7,0.15,0.15",
                        help="train,validation,test split ratios")
    parser.add_argument("--report", default=None,
                        help="write a JSON report (metrics + attributions) here")
    parser.add_argument("--top-features", type=int, default=25,
                        help="how many attributions to report (default 25)")
    parser.add_argument("--permutation-importance", action="store_true",
                        help="force permutation importance instead of the "
                             "model's own coefficients/importances")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="refuse to fit below this many training rows")
    parser.add_argument("--dry-run", action="store_true",
                        help="fit and evaluate but do not write the artefact")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")
    print(f"dataset: {common.dataset_summary(records)}")

    print("\nextracting stylometric feature vectors ...")
    vectors, feature_names = common.extract_stylometric_vectors(records)
    print(f"  {len(feature_names)} distinct features across {len(vectors)} rows")

    ratios = support.parse_ratios(args.ratios)
    splits = support.random_split(records, ratios, args.seed)
    support.print_split(splits)

    train_records = splits["train"]
    if len(train_records) < args.min_samples:
        raise SystemExit(
            f"only {len(train_records)} training rows; at least "
            f"{args.min_samples} are required. A stylometric model fitted on "
            "less than that memorises its corpus. Use --min-samples to "
            "override deliberately.")
    if len({r.label for r in train_records}) < 2:
        raise SystemExit(
            "the training split contains a single label; nothing to learn.")

    X_train = vectorizer.stack(
        [vectors[i] for i in support.row_indices(records, train_records)],
        feature_names)
    y_train = [r.label for r in train_records]

    estimator, algorithm = common.build_estimator(args.algorithm, args.seed)
    print(f"\nfitting {algorithm} on {len(y_train)} rows x "
          f"{len(feature_names)} features ...")
    estimator.fit(X_train, y_train)
    model_classes = [str(c) for c in estimator.classes_]
    print(f"  classes_: {model_classes}")

    evaluations: Dict[str, Dict[str, Any]] = {}
    for split_name in ("validation", "test"):
        metrics = _evaluate(estimator, model_classes, records, vectors,
                            feature_names, splits[split_name], split_name)
        if metrics is not None:
            evaluations[split_name] = metrics

    X_validation, y_validation = _matrix(records, vectors, feature_names,
                                         splits["validation"])
    importances = support.feature_importances(
        estimator, feature_names,
        X=X_validation if X_validation is not None else X_train,
        y=y_validation if y_validation is not None else y_train,
        top=args.top_features, permutation=args.permutation_importance,
        seed=args.seed)
    support.print_importances(
        f"top {args.top_features} stylometric features", importances)

    payload: Dict[str, Any] = {
        "model": estimator,
        "feature_names": feature_names,
        "algorithm": algorithm,
        "classes": model_classes,
        "trained_on": common.stamp(data_path, train_records),
        "metrics": evaluations.get("validation", {}),
        "test_metrics": evaluations.get("test", {}),
        "feature_importances": importances,
        "split": {"ratios": list(ratios), "seed": args.seed,
                  "sizes": {k: len(v) for k, v in splits.items()}},
    }

    if args.dry_run:
        print("\n--dry-run: artefact not written")
    else:
        written = common.save_artefact(Path(args.out), payload)
        print(f"\nartefact written to {written}")
        print("StylometryDetector will now report trained=true.")

    support.write_report(args.report, {
        "component": "stylometry",
        "dataset": str(data_path),
        "algorithm": algorithm,
        "seed": args.seed,
        "feature_count": len(feature_names),
        "split": support.describe_split(splits),
        "metrics": evaluations,
        "feature_importances": importances,
        "artefact": None if args.dry_run else str(args.out),
    })
    return 0


def _matrix(records, vectors, feature_names, subset):
    if not subset:
        return None, None
    indices = support.row_indices(records, subset)
    return (vectorizer.stack([vectors[i] for i in indices], feature_names),
            [records[i].label for i in indices])


def _evaluate(estimator, model_classes: List[str], records, vectors,
              feature_names, subset, split_name: str) -> Optional[Dict[str, Any]]:
    X, y = _matrix(records, vectors, feature_names, subset)
    if X is None:
        print(f"\n=== {split_name} === empty split, nothing evaluated")
        return None
    probabilities = support.project_probabilities(
        estimator.predict_proba(X), model_classes, config.CLASSES)
    metrics = common.evaluate_predictions(y, probabilities, list(config.CLASSES))
    common.print_metrics(f"stylometry / {split_name}", metrics)
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
