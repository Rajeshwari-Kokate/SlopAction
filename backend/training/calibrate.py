"""
Fit the probability calibrator for the meta-classifier.

    python -m training.calibrate --data data/corpus.jsonl --method temperature

Why this stage is not optional
------------------------------
A softmax output is not a probability.  A model can be 90% accurate and still
say "0.95" on cases it gets right 70% of the time.  Reporting that as "95% AI"
about a named human being is the exact failure this project is built to avoid,
so until a calibrator artefact exists the API is required to call its output a
*detection score* and ``is_probability`` stays false.  This script is what
changes that, and it is only entitled to do so because it measures the result.

What it does
------------
1. Rebuilds the ensemble feature vectors (shared cache with
   ``training/train_ensemble.py``, so this is cheap after a fit).
2. Loads the **trained meta-classifier** and takes its raw, uncalibrated
   probabilities on a held-out split.  Calibrating on the fitting split would
   learn the model's memorisation and report a beautiful, meaningless ECE.
3. Fits ``temperature`` (one scalar, NLL-minimising, arg-max preserving),
   ``platt`` (per-class vector scaling) or ``isotonic`` (per-class,
   non-parametric), and writes exactly the payload
   ``app/ensemble/calibrator.load()`` expects for that method.
4. Prints ECE and Brier before and after, plus a reliability table, on both the
   calibration split and - when one exists - a second untouched split.

Choosing a method
-----------------
``temperature`` is the default and should stay the default on anything under a
few thousand validation rows: one parameter cannot overfit and it never changes
which class wins.  ``platt`` needs more data; ``isotonic`` needs the most and
will happily memorise a small split - watch the difference between the
calibration-split and held-out numbers printed below, because that gap is the
whole story.  ``--method auto`` fits all three and picks the lowest held-out ECE,
which is honest only when the held-out split is real; it says so if it is not.

Honest limitation
-----------------
Calibration is fitted on one distribution.  A calibrator fitted on essays does
not transfer to chat logs, and nothing in the artefact prevents it from being
applied to them.  The OOD detector is what limits the damage.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.ensemble import calibrator as calibrator_module  # noqa: E402
from app.features import vectorizer  # noqa: E402
from training import common, support, train_ensemble  # noqa: E402

METHODS = ("temperature", "platt", "isotonic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.calibrate",
        description="Fit temperature / Platt / isotonic calibration for the "
                    "meta-classifier.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(config.ArtifactPaths().calibrator))
    parser.add_argument("--model", default=str(config.ArtifactPaths().meta_classifier),
                        help="meta-classifier artefact to calibrate")
    parser.add_argument("--method", default=config.CALIBRATION_METHOD,
                        choices=(*METHODS, "auto"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.7,0.15,0.15",
                        help="must match the ratios and seed used by "
                             "train_ensemble, or the calibration split will "
                             "contain rows the model was fitted on")
    parser.add_argument("--split", default="validation",
                        choices=("validation", "test"),
                        help="which split to fit the calibrator on")
    parser.add_argument("--mode", default="standard",
                        choices=sorted(config.MODE_DETECTORS))
    parser.add_argument("--cache", default=None,
                        help="ensemble-vector cache shared with train_ensemble")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--bins", type=int, default=10,
                        help="reliability-diagram bins (default 10)")
    parser.add_argument("--report", default=None)
    parser.add_argument("--min-samples", type=int, default=20,
                        help="refuse to fit a calibrator on fewer rows")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)
    classes = list(config.CLASSES)

    artefact = _load_meta_classifier(Path(args.model))
    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")

    vectors, _ = train_ensemble.extract_vectors(
        records, mode=args.mode, cache=args.cache, refresh=args.refresh_cache)

    ratios = support.parse_ratios(args.ratios)
    splits = support.random_split(records, ratios, args.seed)
    support.print_split(splits)

    fit_records = splits[args.split]
    holdout_name = "test" if args.split == "validation" else "validation"
    holdout_records = splits[holdout_name]
    if len(fit_records) < args.min_samples:
        raise SystemExit(
            f"the '{args.split}' split has {len(fit_records)} rows; at least "
            f"{args.min_samples} are required. A calibrator fitted on a handful "
            "of documents is worse than none, because it makes the output look "
            "trustworthy.")

    feature_names = list(artefact["feature_names"])
    model = artefact["model"]
    model_classes = [str(c) for c in getattr(model, "classes_", classes)]

    fit_probabilities, fit_labels = _raw_probabilities(
        model, model_classes, classes, records, vectors, feature_names,
        fit_records)
    holdout = None
    if holdout_records:
        holdout = _raw_probabilities(
            model, model_classes, classes, records, vectors, feature_names,
            holdout_records)

    print(f"\ncalibrating on the '{args.split}' split "
          f"({len(fit_labels)} rows), checking on '{holdout_name}' "
          f"({len(holdout_records)} rows)")

    candidates = METHODS if args.method == "auto" else (args.method,)
    fitted: Dict[str, Dict[str, Any]] = {}
    for method in candidates:
        payload = _fit(method, fit_probabilities, fit_labels, classes)
        instance = calibrator_module._BUILDERS[method](payload)
        entry: Dict[str, Any] = {"payload": payload, "calibrator": instance}
        entry["fit_split"] = _compare(
            instance, fit_probabilities, fit_labels, classes, args.bins,
            f"{method} / {args.split} (calibration split)")
        if holdout is not None:
            entry["holdout"] = _compare(
                instance, holdout[0], holdout[1], classes, args.bins,
                f"{method} / {holdout_name} (untouched)")
        fitted[method] = entry

    chosen = _choose(fitted, args.method, holdout is not None)
    entry = fitted[chosen]
    payload = entry["payload"]

    reference = entry.get("holdout") or entry["fit_split"]
    payload["metadata"] = {
        "fitted_on": str(data_path),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_calibration_samples": len(fit_labels),
        "calibration_split": args.split,
        "analysis_mode": args.mode,
        "meta_classifier": str(args.model),
        "ece_before": round(reference["before"]["ece"], 6),
        "ece_after": round(reference["after"]["ece"], 6),
        "brier_before": round(reference["before"]["brier"], 6),
        "brier_after": round(reference["after"]["brier"], 6),
        "measured_on": reference["split"],
        "selection": ("auto: lowest held-out ECE" if args.method == "auto"
                      else "explicit --method"),
    }

    print(f"\nselected method: {chosen}")
    if args.method == "auto":
        for method, item in fitted.items():
            source = item.get("holdout") or item["fit_split"]
            print(f"  {method:12s} ECE {source['before']['ece']:.4f} -> "
                  f"{source['after']['ece']:.4f} on {source['split']}")

    if args.dry_run:
        print("\n--dry-run: calibrator not written")
    else:
        import joblib

        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, destination)
        print(f"\ncalibrator written to {destination}")
        loaded = calibrator_module.load()
        print(f"  round-trip check: {loaded.describe().get('method')}, "
              f"fitted={loaded.fitted}")
        if not loaded.fitted:
            raise SystemExit(
                "the artefact was written but calibrator.load() did not accept "
                "it; the payload format is wrong and the API would silently "
                "keep reporting an uncalibrated detection score.")
        print("  the API will now report score_type=calibrated_probability "
              "(provided the meta-classifier is also installed).")

    support.write_report(args.report, {
        "component": "calibration",
        "dataset": str(data_path),
        "method": chosen,
        "requested_method": args.method,
        "analysis_mode": args.mode,
        "calibration_split": args.split,
        "seed": args.seed,
        "split": support.describe_split(splits),
        "results": {method: {"fit_split": item["fit_split"],
                             "holdout": item.get("holdout")}
                    for method, item in fitted.items()},
        "metadata": payload.get("metadata"),
        "artefact": None if args.dry_run else str(args.out),
    })
    return 0


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _load_meta_classifier(path: Path) -> Dict[str, Any]:
    import joblib

    if not path.exists():
        raise SystemExit(
            f"no meta-classifier at {path}. There is nothing to calibrate: the "
            "untrained ensemble fallback is a pooled heuristic, and calibrating "
            "a heuristic would dress it up as a probability. Run "
            "python -m training.train_ensemble first.")
    artefact = joblib.load(path)
    if not isinstance(artefact, dict) or "model" not in artefact \
            or "feature_names" not in artefact:
        raise SystemExit(
            f"{path} is not a valid meta-classifier artefact (needs 'model' and "
            "'feature_names').")
    return artefact


def _raw_probabilities(model, model_classes: List[str], classes: List[str],
                       records, vectors, feature_names, subset
                       ) -> Tuple[Any, List[str]]:
    indices = support.row_indices(records, subset)
    X = vectorizer.stack([vectors[i] for i in indices], feature_names)
    probabilities = support.project_probabilities(
        model.predict_proba(X), model_classes, classes)
    return probabilities, [records[i].label for i in indices]


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _fit(method: str, probabilities, labels: Sequence[str],
         classes: Sequence[str]) -> Dict[str, Any]:
    if method == "temperature":
        temperature = calibrator_module.fit_temperature(
            probabilities, labels, classes)
        print(f"\nfitted temperature T = {temperature:.4f} "
              f"({'sharpening' if temperature < 1 else 'softening'} the "
              "distribution)")
        return {"method": "temperature", "temperature": float(temperature),
                "metadata": {}}

    if method == "platt":
        weights, biases = _fit_vector_scaling(probabilities, labels, classes)
        print("\nfitted per-class vector scaling:")
        for klass in classes:
            print(f"  {klass:14s} w={weights[klass]:+.4f} b={biases[klass]:+.4f}")
        return {"method": "platt", "weights": weights, "biases": biases,
                "metadata": {}}

    if method == "isotonic":
        regressors = _fit_isotonic(probabilities, labels, classes)
        print(f"\nfitted isotonic regressors for {sorted(regressors)}")
        return {"method": "isotonic", "regressors": regressors, "metadata": {}}

    raise SystemExit(f"unknown calibration method '{method}'")


def _fit_vector_scaling(probabilities, labels: Sequence[str],
                        classes: Sequence[str]):
    """Multinomial vector scaling: one weight and one bias per class.

    Minimises NLL of ``softmax(w_k * log p_k + b_k)`` with an analytic gradient.
    """
    import numpy as np
    from scipy.optimize import minimize

    logits = np.log(np.clip(np.asarray(probabilities, dtype="float64"),
                            1e-12, 1.0))
    index = {c: i for i, c in enumerate(classes)}
    targets = np.zeros_like(logits)
    for row, label in enumerate(labels):
        targets[row, index[str(label)]] = 1.0
    k = len(classes)

    def objective(parameters):
        weights = parameters[:k]
        biases = parameters[k:]
        scaled = logits * weights + biases
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exponentials = np.exp(scaled)
        probability = exponentials / exponentials.sum(axis=1, keepdims=True)
        loss = float(-(np.log(np.clip(probability, 1e-12, 1.0))
                       * targets).sum(axis=1).mean())
        residual = (probability - targets) / len(labels)
        gradient = np.concatenate([(residual * logits).sum(axis=0),
                                   residual.sum(axis=0)])
        return loss, gradient

    start = np.concatenate([np.ones(k), np.zeros(k)])
    outcome = minimize(objective, start, jac=True, method="L-BFGS-B")
    weights = {c: float(outcome.x[i]) for i, c in enumerate(classes)}
    biases = {c: float(outcome.x[k + i]) for i, c in enumerate(classes)}
    return weights, biases


def _fit_isotonic(probabilities, labels: Sequence[str],
                  classes: Sequence[str]) -> Dict[str, Any]:
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    probabilities = np.asarray(probabilities, dtype="float64")
    index = {c: i for i, c in enumerate(classes)}
    labels = [str(l) for l in labels]
    regressors: Dict[str, Any] = {}
    for klass in classes:
        column = probabilities[:, index[klass]]
        target = np.asarray([1.0 if l == klass else 0.0 for l in labels])
        if len(set(target.tolist())) < 2:
            print(f"  skipping isotonic fit for '{klass}': the calibration "
                  "split has only one outcome for it")
            continue
        regressor = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip")
        regressor.fit(column, target)
        regressors[klass] = regressor
    if not regressors:
        raise SystemExit(
            "no class had both positive and negative examples; isotonic "
            "calibration cannot be fitted on this split.")
    return regressors


def _choose(fitted: Dict[str, Dict[str, Any]], requested: str,
            has_holdout: bool) -> str:
    if requested != "auto":
        return requested
    if not has_holdout:
        print("\nWARNING: --method auto with no held-out split. The comparison "
              "below is in-sample, so isotonic will usually look best and will "
              "usually be wrong. Falling back to temperature.")
        return "temperature"
    return min(fitted, key=lambda m: fitted[m]["holdout"]["after"]["ece"])


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def _apply(instance, probabilities, classes: Sequence[str]):
    import numpy as np

    rows = []
    for row in np.asarray(probabilities, dtype="float64"):
        mapping = {c: float(v) for c, v in zip(classes, row)}
        adjusted = instance.apply(mapping)
        rows.append([float(adjusted.get(c, 0.0)) for c in classes])
    return np.asarray(rows, dtype="float64")


def _quality(probabilities, labels: Sequence[str],
             classes: Sequence[str]) -> Dict[str, float]:
    import numpy as np

    probabilities = np.asarray(probabilities, dtype="float64")
    index = {c: i for i, c in enumerate(classes)}
    targets = np.asarray([index[str(l)] for l in labels])
    predictions = probabilities.argmax(axis=1)
    return {
        "ece": calibrator_module.expected_calibration_error(
            probabilities, labels, classes),
        "brier": calibrator_module.brier_score(probabilities, labels, classes),
        "accuracy": float((predictions == targets).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def _reliability_table(probabilities, labels: Sequence[str],
                       classes: Sequence[str], bins: int) -> List[Dict[str, Any]]:
    import numpy as np

    probabilities = np.asarray(probabilities, dtype="float64")
    index = {c: i for i, c in enumerate(classes)}
    targets = np.asarray([index[str(l)] for l in labels])
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == targets).astype("float64")

    edges = np.linspace(0.0, 1.0, bins + 1)
    table: List[Dict[str, Any]] = []
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if i == 0:
            mask |= confidence <= edges[0]
        count = int(mask.sum())
        table.append({
            "bin": f"({edges[i]:.2f}, {edges[i + 1]:.2f}]",
            "count": count,
            "mean_confidence": round(float(confidence[mask].mean()), 4)
            if count else None,
            "empirical_accuracy": round(float(correct[mask].mean()), 4)
            if count else None,
            "gap": round(float(correct[mask].mean() - confidence[mask].mean()), 4)
            if count else None,
        })
    return table


def _print_reliability(title: str, table: Sequence[Dict[str, Any]]) -> None:
    print(f"\n  reliability diagram - {title}")
    print("    bin             count   mean conf   empirical acc      gap")
    for row in table:
        if not row["count"]:
            continue
        print(f"    {row['bin']:14s} {row['count']:5d}      "
              f"{row['mean_confidence']:.4f}         {row['empirical_accuracy']:.4f}   "
              f"{row['gap']:+.4f}")


def _compare(instance, probabilities, labels: Sequence[str],
             classes: Sequence[str], bins: int, title: str) -> Dict[str, Any]:
    calibrated = _apply(instance, probabilities, classes)
    before = _quality(probabilities, labels, classes)
    after = _quality(calibrated, labels, classes)

    print(f"\n=== {title} ===")
    print(f"  n              : {len(labels)}")
    print(f"  ECE            : {before['ece']:.4f} -> {after['ece']:.4f}")
    print(f"  Brier          : {before['brier']:.4f} -> {after['brier']:.4f}")
    print(f"  mean confidence: {before['mean_confidence']:.4f} -> "
          f"{after['mean_confidence']:.4f}")
    print(f"  accuracy       : {before['accuracy']:.4f} -> "
          f"{after['accuracy']:.4f} (temperature scaling must not change this)")

    table_before = _reliability_table(probabilities, labels, classes, bins)
    table_after = _reliability_table(calibrated, labels, classes, bins)
    _print_reliability("before", table_before)
    _print_reliability("after", table_after)

    return {
        "split": title,
        "n": len(labels),
        "before": before,
        "after": after,
        "reliability_before": table_before,
        "reliability_after": table_after,
    }


if __name__ == "__main__":
    raise SystemExit(main())
