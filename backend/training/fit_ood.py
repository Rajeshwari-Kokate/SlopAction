"""
Fit the statistical out-of-distribution reference.

    python -m training.fit_ood --data data/corpus.jsonl

What this is for
----------------
A detector validated on English prose should not express a confident opinion
about a sonnet, a Python file or a chat log.  ``OODDetector`` already catches
those structurally.  This artefact adds the statistical half: the Mahalanobis
distance of a document's stylometric vector from the distribution the system was
actually trained on, expressed as a percentile of the distances seen during
fitting.

Without this file ``_statistical`` reports itself unavailable rather than
inventing a distance, which is correct but means unusual-but-well-formed prose
passes unremarked.

What is computed
----------------
``mean``                the per-feature mean of the reference distribution
``precision``           the inverse covariance, estimated with Ledoit-Wolf
                        shrinkage.  Plain inversion is not an option: with ~200
                        stylometric features and a realistic corpus the sample
                        covariance is ill-conditioned or singular, and its
                        inverse produces distances that are numerical noise.
``distance_quantiles``  sorted in-distribution distances, used by the detector
                        for percentile lookup
``feature_names``       the projection order, as everywhere else in this project

Honest limitations
------------------
* This measures distance from *the fitting corpus*, not from "normal English".
  Fit it on the same data the detectors were fitted on, or the OOD score will
  flag the detectors' own training distribution as unusual.
* Mahalanobis distance assumes a single elliptical blob.  A corpus that is
  genuinely multi-modal (essays plus tweets) gets a reference whose centre sits
  between the modes, and both modes look mildly out of distribution.  Fit
  per-domain references, or accept the conservatism.
* The score is a reliability discount.  It never moves the AI/human numbers;
  it lowers the confidence attached to them.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.features import vectorizer  # noqa: E402
from training import common, support  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.fit_ood",
        description="Fit the Mahalanobis reference used for statistical OOD "
                    "detection.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(config.ArtifactPaths().ood_reference))
    parser.add_argument("--algorithm", default="ledoit_wolf",
                        choices=("ledoit_wolf", "oas", "empirical"),
                        help="covariance estimator; shrinkage is the default "
                             "for a reason")
    parser.add_argument("--seed", type=int, default=0,
                        help="only used when --sample-size subsamples")
    parser.add_argument("--labels", default=None,
                        help="restrict the reference to these labels "
                             "(e.g. 'human' for a human-only reference)")
    parser.add_argument("--split", default=None,
                        help="restrict to records carrying this split value")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="subsample the reference set to this many rows")
    parser.add_argument("--quantile-points", type=int, default=1001,
                        help="size of the stored distance grid when the corpus "
                             "is larger than it")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="refuse to fit a reference below this many rows")
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import numpy as np

    args = build_parser().parse_args(argv)
    data_path = Path(args.data)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")

    records = _select(records, args)
    print(f"reference set: {len(records)} records")
    print(f"  {common.dataset_summary(records)}")
    if len(records) < args.min_samples:
        raise SystemExit(
            f"only {len(records)} reference documents; at least "
            f"{args.min_samples} are required. A covariance estimated from "
            "fewer rows than features describes the sample, not the "
            "distribution, and every percentile it produces is fiction.")

    print("\nextracting stylometric feature vectors ...")
    vectors, feature_names = common.extract_stylometric_vectors(records)
    X = vectorizer.stack(vectors, feature_names)
    n_samples, n_features = X.shape
    print(f"  {n_samples} rows x {n_features} features")
    if n_samples < 5 * n_features:
        print(f"  WARNING: {n_samples} rows for {n_features} features. "
              "Shrinkage keeps the estimate usable, but the distances will be "
              "dominated by the shrinkage target rather than by this corpus.")

    constant = [feature_names[i] for i in np.where(X.std(axis=0) == 0)[0]]
    if constant:
        print(f"  {len(constant)} feature(s) are constant across the corpus "
              "and carry no OOD information (kept; shrinkage handles them)")

    mean, precision, estimator_name, shrinkage = _fit_precision(X, args.algorithm)
    centred = X - mean
    distances = np.sqrt(np.maximum(
        0.0, np.einsum("ij,jk,ik->i", centred, precision, centred)))
    distances.sort()

    quantiles = _quantile_grid(distances, args.quantile_points)
    percentiles = {
        f"p{p}": round(float(np.percentile(distances, p)), 4)
        for p in (5, 25, 50, 75, 90, 95, 99)
    }
    print(f"\nfitted {estimator_name}"
          + (f" (shrinkage={shrinkage:.4f})" if shrinkage is not None else ""))
    print(f"  in-distribution Mahalanobis distances: {percentiles}")
    print(f"  stored quantile grid: {len(quantiles)} points")

    payload: Dict[str, Any] = {
        "mean": mean,
        "precision": precision,
        "feature_names": feature_names,
        "distance_quantiles": quantiles,
        "fitted_on": str(data_path),
        "n_samples": int(n_samples),
        "algorithm": estimator_name,
        "shrinkage": None if shrinkage is None else float(shrinkage),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "distance_percentiles": percentiles,
        "label_filter": args.labels,
        "summary": common.dataset_summary(records),
    }

    if args.dry_run:
        print("\n--dry-run: reference not written")
    else:
        import joblib

        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, destination)
        print(f"\nreference written to {destination}")
        _verify(destination, records[0].text)

    support.write_report(args.report, {
        "component": "ood_reference",
        "dataset": str(data_path),
        "algorithm": estimator_name,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "shrinkage": None if shrinkage is None else float(shrinkage),
        "distance_percentiles": percentiles,
        "constant_features": constant,
        "artefact": None if args.dry_run else str(args.out),
    })
    return 0


def _select(records: Sequence[Any], args) -> List[Any]:
    selected = list(records)
    if args.split:
        selected = [r for r in selected if r.split == args.split]
        if not selected:
            raise SystemExit(f"no records carry split={args.split!r}")
    if args.labels:
        wanted = {l.strip() for l in args.labels.split(",") if l.strip()}
        unknown = wanted - set(config.CLASSES)
        if unknown:
            raise SystemExit(f"unknown label(s) {sorted(unknown)}")
        selected = [r for r in selected if r.label in wanted]
        if not selected:
            raise SystemExit(f"no records with label(s) {sorted(wanted)}")
    if args.sample_size and args.sample_size < len(selected):
        import random

        random.Random(args.seed).shuffle(selected)
        selected = selected[:args.sample_size]
    return selected


def _fit_precision(X, algorithm: str):
    """Mean vector and (regularised) precision matrix."""
    import numpy as np

    if algorithm == "empirical":
        from sklearn.covariance import EmpiricalCovariance

        estimator = EmpiricalCovariance(store_precision=True).fit(X)
        print("  NOTE: --algorithm empirical inverts the sample covariance "
              "directly. With more features than samples that inverse is not "
              "meaningful; prefer ledoit_wolf unless you know why not.")
        return (np.asarray(estimator.location_), np.asarray(estimator.precision_),
                "empirical_covariance", None)

    if algorithm == "oas":
        from sklearn.covariance import OAS

        estimator = OAS(store_precision=True).fit(X)
        return (np.asarray(estimator.location_), np.asarray(estimator.precision_),
                "oracle_approximating_shrinkage", float(estimator.shrinkage_))

    from sklearn.covariance import LedoitWolf

    estimator = LedoitWolf(store_precision=True).fit(X)
    return (np.asarray(estimator.location_), np.asarray(estimator.precision_),
            "ledoit_wolf", float(estimator.shrinkage_))


def _quantile_grid(distances, points: int):
    """Sorted distances, thinned to a fixed grid when the corpus is large.

    ``ood_detector`` converts a distance to a percentile with
    ``searchsorted(quantiles, d) / len(quantiles)``, which only requires the
    array to be sorted and representative.
    """
    import numpy as np

    if points <= 0 or len(distances) <= points:
        return np.asarray(distances, dtype="float64")
    probabilities = np.linspace(0.0, 100.0, points)
    return np.asarray(np.percentile(distances, probabilities), dtype="float64")


def _verify(path: Path, sample_text: str) -> None:
    """Run the detector's own loader over the artefact we just wrote."""
    from app.detectors.ood_detector import OODDetector
    from app.utils import model_loader

    model_loader.reset_cache()
    context = common.build_context(sample_text)
    result = OODDetector().analyse(context)
    statistical = (result.raw or {}).get("statistical", {})
    if not statistical.get("available"):
        raise SystemExit(
            "the artefact was written but OODDetector could not use it: "
            f"{statistical.get('reason')}")
    print(f"  round-trip check: distance="
          f"{statistical.get('mahalanobis_distance')} "
          f"percentile={statistical.get('training_percentile')} "
          f"trained={result.trained}")


if __name__ == "__main__":
    raise SystemExit(main())
