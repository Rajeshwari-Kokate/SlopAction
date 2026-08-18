"""
Fit the humanized-AI classifier (Engine G).

    python -m training.train_humanization --data data/corpus.jsonl

What this model's job actually is
---------------------------------
It has to separate ``humanized_ai`` from **both** of the other classes at once.
That is a harder and different question from "is this machine text", and it is
the question the rest of the system cannot answer: a two-class detector trained
on raw model output sees a paraphrased document as human, and a stylometry model
sees it as whatever the paraphraser's style happens to resemble.

The features come from ``common.extract_humanization_vectors``, which runs the
detector's own three-view extractor (original / content / expression) plus the
six cross-view divergence features.  Those divergence features are the point:
"same ideas, different words" is visible as high semantic redundancy with low
lexical repetition, and neither view shows it alone.

How the detector consumes this
------------------------------
``HumanizationDetector._trained_result`` calls ``predict_proba`` and reads the
column named ``humanized_ai``::

    lookup = dict(zip(model.classes_, probabilities))
    humanization_probability = lookup["humanized_ai"]

So the model is fitted on the **three original labels**, not on a collapsed
binary target - that keeps ``classes_`` containing the literal string
``humanized_ai``.  The engine then uses it as a binary humanized-vs-rest score,
which is exactly what the ensemble wants from this engine.  Fitting a two-class
``humanized`` / ``other`` model would also work numerically but would produce
``classes_ = ['humanized', 'other']`` and the detector would silently read 0.0
for every document.  ``--binary`` is offered for experiments and relabels to
``humanized_ai`` / ``human`` so the contract still holds; it is not the default.

Honest limitations
------------------
* The content view needs sentence embeddings.  Without a sentence-transformer
  the content block collapses to ``available=0`` and the divergence features
  that depend on it are zero - the model then fits on the remaining views and
  will be weaker in a way the artefact's metrics will not tell you about.  Fit
  with embeddings available.
* The original view needs a causal LM for surprisal features.  Same story.
* Humanizers are a moving target.  A model fitted on the output of three
  paraphrase tools tells you nothing about the fourth.  Hold humanizers out and
  read the per-humanizer slice in ``training/evaluate.py``.
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
        prog="python -m training.train_humanization",
        description="Fit the humanized-AI classifier consumed by Engine G.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out",
                        default=str(config.ArtifactPaths().humanization_model))
    parser.add_argument("--algorithm", default="logistic",
                        help="logistic | rf | gb | lgbm | xgb | mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--report", default=None)
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument(
        "--binary", action="store_true",
        help="collapse pure_ai and human into one negative class labelled "
             "'human' (keeps the required 'humanized_ai' class name)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")
    print(f"dataset: {common.dataset_summary(records)}")
    if not any(r.label == "humanized_ai" for r in records):
        raise SystemExit(
            "the dataset contains no humanized_ai rows. This model exists only "
            "to recognise that class; fitting it without examples of it would "
            "produce a classifier that can never emit the column the detector "
            "reads.")

    print("\nextracting three-view humanization vectors ...")
    vectors, feature_names = common.extract_humanization_vectors(records)
    print(f"  {len(feature_names)} distinct features across {len(vectors)} rows")
    _report_view_coverage(vectors)

    ratios = support.parse_ratios(args.ratios)
    splits = support.random_split(records, ratios, args.seed)
    support.print_split(splits)

    train_records = splits["train"]
    if len(train_records) < args.min_samples:
        raise SystemExit(
            f"only {len(train_records)} training rows; at least "
            f"{args.min_samples} are required.")

    targets = _targets(train_records, args.binary)
    if "humanized_ai" not in set(targets):
        raise SystemExit(
            "the training split contains no humanized_ai rows, so classes_ "
            "would not contain the label the detector reads.")

    X_train = vectorizer.stack(
        [vectors[i] for i in support.row_indices(records, train_records)],
        feature_names)

    estimator, algorithm = common.build_estimator(args.algorithm, args.seed)
    print(f"\nfitting {algorithm} on {len(targets)} rows x "
          f"{len(feature_names)} features ...")
    estimator.fit(X_train, targets)
    model_classes = [str(c) for c in estimator.classes_]
    print(f"  classes_: {model_classes}")
    if "humanized_ai" not in model_classes:
        raise SystemExit(
            "classes_ does not contain 'humanized_ai'; the detector would read "
            "0.0 for every document. Refusing to write this artefact.")

    class_space = model_classes if args.binary else list(config.CLASSES)
    evaluations: Dict[str, Dict[str, Any]] = {}
    for split_name in ("validation", "test"):
        metrics = _evaluate(estimator, model_classes, class_space, records,
                            vectors, feature_names, splits[split_name],
                            split_name, args.binary)
        if metrics is not None:
            evaluations[split_name] = metrics

    X_validation, y_validation = _matrix(records, vectors, feature_names,
                                         splits["validation"], args.binary)
    importances = support.feature_importances(
        estimator, feature_names,
        X=X_validation if X_validation is not None else X_train,
        y=y_validation if y_validation is not None else targets,
        top=args.top_features, permutation=args.permutation_importance,
        seed=args.seed)
    support.print_importances(
        f"top {args.top_features} humanization features", importances)

    payload: Dict[str, Any] = {
        "model": estimator,
        "feature_names": feature_names,
        "algorithm": algorithm,
        "classes": model_classes,
        "trained_on": common.stamp(data_path, train_records),
        "metrics": evaluations.get("validation", {}),
        "test_metrics": evaluations.get("test", {}),
        "feature_importances": importances,
        "framing": ("binary humanized-vs-rest" if args.binary
                    else "three-class, consumed as humanized-vs-rest"),
        "split": {"ratios": list(ratios), "seed": args.seed,
                  "sizes": {k: len(v) for k, v in splits.items()}},
    }

    if args.dry_run:
        print("\n--dry-run: artefact not written")
    else:
        written = common.save_artefact(Path(args.out), payload)
        print(f"\nartefact written to {written}")
        print("HumanizationDetector will now report trained=true and read the "
              "'humanized_ai' probability column.")

    support.write_report(args.report, {
        "component": "humanization",
        "dataset": str(data_path),
        "algorithm": algorithm,
        "seed": args.seed,
        "feature_count": len(feature_names),
        "classes": model_classes,
        "split": support.describe_split(splits),
        "metrics": evaluations,
        "feature_importances": importances,
        "artefact": None if args.dry_run else str(args.out),
    })
    return 0


def _targets(subset, binary: bool) -> List[str]:
    if not binary:
        return [r.label for r in subset]
    return ["humanized_ai" if r.label == "humanized_ai" else "human"
            for r in subset]


def _matrix(records, vectors, feature_names, subset, binary: bool):
    if not subset:
        return None, None
    indices = support.row_indices(records, subset)
    return (vectorizer.stack([vectors[i] for i in indices], feature_names),
            _targets([records[i] for i in indices], binary))


def _evaluate(estimator, model_classes: List[str], class_space: List[str],
              records, vectors, feature_names, subset, split_name: str,
              binary: bool) -> Optional[Dict[str, Any]]:
    X, y = _matrix(records, vectors, feature_names, subset, binary)
    if X is None:
        print(f"\n=== {split_name} === empty split, nothing evaluated")
        return None
    probabilities = support.project_probabilities(
        estimator.predict_proba(X), model_classes, class_space)
    metrics = common.evaluate_predictions(y, probabilities, class_space)
    common.print_metrics(f"humanization / {split_name}", metrics)
    return metrics


def _report_view_coverage(vectors: Sequence[Dict[str, float]]) -> None:
    """Say plainly which views were actually measurable."""
    total = max(1, len(vectors))
    content = sum(1 for v in vectors if v.get("cont__available", 0.0) > 0.5)
    lm = sum(1 for v in vectors if v.get("orig__lm_available", 0.0) > 0.5)
    print(f"  content view (embeddings) available on {content}/{total} rows")
    print(f"  original view LM surprisal available on {lm}/{total} rows")
    if content == 0:
        print("  WARNING: no document had sentence embeddings. Every "
              "paraphrase-specific divergence feature is zero, and this model "
              "cannot learn the signal it exists to learn.")
    if lm == 0:
        print("  WARNING: no document had language-model surprisal features.")


if __name__ == "__main__":
    raise SystemExit(main())
