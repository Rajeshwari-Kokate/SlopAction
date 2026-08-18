"""
Fit the meta-classifier that turns detector evidence into one estimate.

    python -m training.train_ensemble --data data/corpus.jsonl --mode standard

Why a learned combiner
----------------------
``app/ensemble/meta_classifier.py`` explains the rule this artefact exists to
enforce: the final answer is not the average of the detectors.  They disagree in
structured, domain-dependent ways, several are only meaningful in combination,
and one of them (stylometry) is confidently wrong on whole genres.  This script
runs every engine over every document, packs the resulting evidence into the
canonical ensemble vector, and fits a model from that vector to
``P(human), P(pure_ai), P(humanized_ai)``.

Train-time and inference-time vectors must match exactly
--------------------------------------------------------
The vector is built by ``meta_classifier.build_ensemble_vector`` - the same
function the service calls - and then the two fields that ``meta_classifier.
predict`` adds afterwards are added here in the same way::

    meta__chunk_weighted_signal   reliability-weighted per-chunk aggregate
    meta__chunk_signal_available  1.0 when chunks carried enough evidence

That aggregate comes from ``aggregation.analyse_units``, exactly as the pipeline
computes it before calling ``predict``.  If it were omitted here, the model would
be fitted without a feature it is handed at inference time, and it would be
handed a zero for it on every request - a silent, permanent distribution shift.

Cost
----
This is the slow trainer: every record pays for a full detector sweep, including
transformer inference and a language-model forward pass.  ``--cache`` stores the
extracted vectors keyed by dataset content and mode, so re-fitting with a
different algorithm or seed costs seconds instead of hours.  Cache entries are
invalidated automatically when the dataset, the mode or the app version changes.

Honest limitations
------------------
* A meta-classifier fitted while an engine was unavailable learns to ignore it.
  The availability flags are in the vector, so the damage is visible rather than
  hidden, but the honest fix is to fit with the same engines you will serve.
* The output of this model is *uncalibrated*.  Run ``training/calibrate.py``
  afterwards; until a calibrator exists the API keeps calling the number a
  detection score.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.core.types import DetectorResult  # noqa: E402
from app.ensemble import aggregation, meta_classifier  # noqa: E402
from app.features import vectorizer  # noqa: E402
from training import common, support  # noqa: E402

CACHE_VERSION = 1


# --------------------------------------------------------------------------
# feature extraction (also used by calibrate.py and evaluate.py)
# --------------------------------------------------------------------------


def run_engines(context) -> Dict[str, DetectorResult]:
    """Run the engine registry over one context, honouring the analysis mode.

    Mirrors ``pipeline.analyse``: engines the mode excludes are recorded as
    unavailable rather than dropped, so the vector layout does not change
    between modes.
    """
    from app import pipeline

    results: Dict[str, DetectorResult] = {}
    for name, engine in pipeline.ENGINES:
        if not context.enabled(name):
            results[name] = DetectorResult.unavailable(
                name, f"not run in '{context.mode}' analysis mode")
            continue
        results[name] = engine.analyse(context)
    return results


def ensemble_vector(text: str, mode: str = "standard"
                    ) -> Tuple[Dict[str, float], Dict[str, DetectorResult],
                               Dict[str, Any]]:
    """Build one ensemble vector exactly as the request path would.

    Returns ``(vector, detector_results, unit_analysis)``.
    """
    context = common.build_context(text, mode=mode)
    results = run_engines(context)
    units = aggregation.analyse_units(context, results)
    chunk_signal = units["document_signal_from_chunks"]

    vector = meta_classifier.build_ensemble_vector(context, results)
    # ---- kept byte-for-byte in step with meta_classifier.predict ----------
    if chunk_signal is not None:
        vector["meta__chunk_weighted_signal"] = float(chunk_signal)
        vector["meta__chunk_signal_available"] = 1.0
    else:
        vector["meta__chunk_signal_available"] = 0.0
    return vector, results, units


def _dataset_signature(records: Sequence[Any], mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"v{CACHE_VERSION}|{mode}|{config.APP_VERSION}|".encode())
    for record in records:
        digest.update(record.text.encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(str(record.label).encode())
        digest.update(b"\x01")
    return digest.hexdigest()


def extract_vectors(records: Sequence[Any], mode: str = "standard",
                    cache: Optional[str] = None, refresh: bool = False,
                    verbose: bool = True
                    ) -> Tuple[List[Dict[str, float]], List[str]]:
    """Extract (or load) one ensemble vector per record.

    The cache is keyed by the dataset contents, the analysis mode and the app
    version.  A stale cache is rebuilt rather than used, because a vector built
    by an older feature set is worse than no cache at all.
    """
    import joblib

    signature = _dataset_signature(records, mode)
    cache_path = Path(cache) if cache else None

    if cache_path and cache_path.exists() and not refresh:
        try:
            payload = joblib.load(cache_path)
            if payload.get("signature") == signature:
                vectors = list(payload["vectors"])
                names = list(payload["feature_names"])
                if verbose:
                    print(f"loaded {len(vectors)} cached ensemble vectors from "
                          f"{cache_path}")
                return vectors, names
            if verbose:
                print(f"cache {cache_path} is stale (dataset, mode or app "
                      "version changed); re-extracting")
        except Exception as exc:  # noqa: BLE001 - a bad cache must not be fatal
            print(f"cache {cache_path} could not be read ({exc}); re-extracting")

    vectors: List[Dict[str, float]] = []
    for index, record in enumerate(records):
        if verbose and index and index % 10 == 0:
            print(f"  ... {index}/{len(records)} ensemble vectors", flush=True)
        vector, _, _ = ensemble_vector(record.text, mode=mode)
        vectors.append(vector)
    names = vectorizer.union_feature_names(vectors)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"signature": signature, "mode": mode, "vectors": vectors,
                     "feature_names": names,
                     "n_records": len(records)}, cache_path)
        if verbose:
            print(f"cached {len(vectors)} ensemble vectors to {cache_path}")
    return vectors, names


def engine_availability(vectors: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Share of rows on which each engine was available, for the report."""
    from app import pipeline

    total = max(1, len(vectors))
    return {
        name: round(sum(1 for v in vectors
                        if v.get(f"{name}__available", 0.0) > 0.5) / total, 4)
        for name, _ in pipeline.ENGINES
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_ensemble",
        description="Fit the meta-classifier over the full detector evidence.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(config.ArtifactPaths().meta_classifier))
    parser.add_argument("--algorithm", default="logistic",
                        help="logistic | rf | gb | lgbm | xgb | mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--mode", default="standard",
                        choices=sorted(config.MODE_DETECTORS),
                        help="which engines run during extraction; fit with "
                             "the mode you will serve")
    parser.add_argument("--cache", default=None,
                        help="joblib path for the extracted vectors "
                             "(extraction is the expensive part)")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")
    print(f"dataset: {common.dataset_summary(records)}")
    print(f"\nrunning the detector sweep in '{args.mode}' mode "
          f"({', '.join(config.MODE_DETECTORS[args.mode])}) ...")

    vectors, feature_names = extract_vectors(
        records, mode=args.mode, cache=args.cache, refresh=args.refresh_cache)
    availability = engine_availability(vectors)
    print(f"  {len(feature_names)} ensemble features")
    print(f"  engine availability: {availability}")
    unavailable = [k for k, v in availability.items() if v == 0.0]
    if unavailable:
        print(f"  WARNING: {unavailable} produced no signal on any row. The "
              "model will learn to ignore them, and will keep ignoring them "
              "after you fix the environment.")

    ratios = support.parse_ratios(args.ratios)
    splits = support.random_split(records, ratios, args.seed)
    support.print_split(splits)

    train_records = splits["train"]
    if len(train_records) < args.min_samples:
        raise SystemExit(
            f"only {len(train_records)} training rows; at least "
            f"{args.min_samples} are required.")
    if len({r.label for r in train_records}) < 2:
        raise SystemExit("the training split contains a single label.")

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
        f"top {args.top_features} ensemble features", importances)

    payload: Dict[str, Any] = {
        "model": estimator,
        "feature_names": feature_names,
        "algorithm": algorithm,
        "classes": model_classes,
        "trained_on": common.stamp(data_path, train_records),
        "metrics": evaluations.get("validation", {}),
        "test_metrics": evaluations.get("test", {}),
        "feature_importances": importances,
        "analysis_mode": args.mode,
        "engine_availability": availability,
        "split": {"ratios": list(ratios), "seed": args.seed,
                  "sizes": {k: len(v) for k, v in splits.items()}},
    }

    if args.dry_run:
        print("\n--dry-run: artefact not written")
    else:
        written = common.save_artefact(Path(args.out), payload)
        print(f"\nartefact written to {written}")
        print("The ensemble will now report trained=true. The numbers are "
              "still uncalibrated - run training/calibrate.py next.")

    support.write_report(args.report, {
        "component": "ensemble",
        "dataset": str(data_path),
        "algorithm": algorithm,
        "analysis_mode": args.mode,
        "seed": args.seed,
        "feature_count": len(feature_names),
        "engine_availability": availability,
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
    common.print_metrics(f"ensemble / {split_name}", metrics)
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
