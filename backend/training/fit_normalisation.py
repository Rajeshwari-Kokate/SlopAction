"""
Fit the feature-normalisation constants that ship as engineering priors.

    python -m training.fit_normalisation --data data/corpus.jsonl

What is being replaced
----------------------
``app/detectors/normalisation.py`` maps raw detector measurements onto a
comparable 0-1 *signal* with a logistic curve per feature::

    signal = sigmoid(direction * (value - midpoint) / scale)

The shipped ``midpoint``/``scale``/``direction`` triples are engineering priors
taken from published ranges - reasonable, but not measured on this corpus, and
in the case of ``binoculars_score`` explicitly only valid for one model pair.
This script measures each referenced feature on human versus AI documents and
writes ``models/normalisation.json``, after which ``normalisation.is_fitted()``
becomes true and the API says so.

What the numbers mean
---------------------
``direction``  +1 when AI documents show larger values, -1 when they show
               smaller ones.  Taken from the class medians, not from theory.
``midpoint``   the decision point.  By default the threshold maximising Youden's
               J (TPR - FPR) on the oriented feature, which is the point that
               separates the two classes best; ``--midpoint median`` uses the
               midpoint of the class medians instead, which is more stable on
               small samples and less sharp.
``scale``      how far the raw feature has to move to change the signal
               appreciably.  Derived from the class separation and the pooled
               spread together, so a feature with widely-overlapping classes
               produces a gentle curve rather than a step.

What this is NOT
----------------
It is not calibration.  The output is a monotone rescaling used for the
untrained ensemble fallback, per-chunk display scores and detector-agreement
measurement.  It is labelled a signal everywhere it is used, never a
probability, and fitting it here does not change that.

Refusing to fit
---------------
A constant fitted on eleven documents is worse than a documented prior, because
it looks measured.  Every feature must clear ``--min-samples`` per class and
``--min-auc`` separation, or it is skipped with a printed reason and the shipped
default survives (``_load`` merges the file over the defaults, so skipping is
safe).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.detectors import normalisation  # noqa: E402
from training import common, support, train_ensemble  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.fit_normalisation",
        description="Measure per-feature normalisation constants on labelled "
                    "data.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out",
                        default=str(Path(config.MODELS_DIR) / "normalisation.json"),
                        help="normalisation.py reads models/normalisation.json")
    parser.add_argument("--mode", default="standard",
                        choices=sorted(config.MODE_DETECTORS),
                        help="deep is required to reach binoculars_score")
    parser.add_argument("--cache", default=None,
                        help="ensemble-vector cache shared with train_ensemble")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", default=None,
                        help="comma separated subset of reference names to fit")
    parser.add_argument("--midpoint", default="youden",
                        choices=("youden", "median"))
    parser.add_argument("--min-samples", type=int, default=25,
                        help="minimum measured documents per class (default 25)")
    parser.add_argument("--min-auc", type=float, default=0.55,
                        help="minimum separation before a constant is written")
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")
    labels = [r.label for r in records]
    if "human" not in labels:
        raise SystemExit(
            "the dataset contains no human rows. Every constant here is defined "
            "by the human-versus-AI contrast; there is nothing to measure.")
    if not any(l != "human" for l in labels):
        raise SystemExit("the dataset contains no AI rows.")

    print(f"\nrunning the detector sweep in '{args.mode}' mode to collect raw "
          "measurements ...")
    vectors, _ = train_ensemble.extract_vectors(
        records, mode=args.mode, cache=args.cache, refresh=args.refresh_cache)

    wanted = _wanted_features(args.features)
    is_ai = [label != "human" for label in labels]

    fitted: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    print(f"\nfitting {len(wanted)} referenced features "
          f"(min {args.min_samples}/class, min AUC {args.min_auc})")

    for name in wanted:
        values, source = _collect(vectors, name)
        outcome, reason = _fit_reference(
            name, values, is_ai, source, args.midpoint, args.min_samples,
            args.min_auc)
        if outcome is None:
            skipped[name] = reason
            print(f"  SKIP {name:28s} {reason}")
            continue
        fitted[name] = outcome
        print(f"  fit  {name:28s} midpoint={outcome['midpoint']:+.4f} "
              f"scale={outcome['scale']:.4f} direction={outcome['direction']:+d} "
              f"auc={outcome['diagnostics']['auc']:.3f} "
              f"(n={outcome['diagnostics']['n_human']}/"
              f"{outcome['diagnostics']['n_ai']})")

    if not fitted:
        raise SystemExit(
            "no feature cleared the guards, so nothing was written and the "
            "shipped engineering priors remain in force. That is the correct "
            "outcome for a dataset this small - it is not a crash.")

    payload = {
        "fitted_on": (f"{data_path} | n={len(records)} | mode={args.mode} | "
                      f"{datetime.now(timezone.utc).isoformat()}"),
        "references": {
            name: {"midpoint": entry["midpoint"], "scale": entry["scale"],
                   "direction": entry["direction"], "note": entry["note"]}
            for name, entry in fitted.items()
        },
    }

    _print_comparison(fitted)

    if args.dry_run:
        print("\n--dry-run: models/normalisation.json not written")
    else:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten to {destination}")
        normalisation._load.cache_clear()
        print(f"  normalisation.is_fitted() -> {normalisation.is_fitted()}")
        print(f"  fitted references: {len(fitted)}; "
              f"{len(skipped)} kept their shipped default")

    support.write_report(args.report, {
        "component": "normalisation",
        "dataset": str(data_path),
        "analysis_mode": args.mode,
        "midpoint_rule": args.midpoint,
        "min_samples": args.min_samples,
        "min_auc": args.min_auc,
        "fitted": fitted,
        "skipped": skipped,
        "artefact": None if args.dry_run else str(args.out),
    })
    return 0


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------


def _wanted_features(selection: Optional[str]) -> List[str]:
    available = list(normalisation.DEFAULT_REFERENCES)
    if not selection:
        return available
    chosen = [f.strip() for f in selection.split(",") if f.strip()]
    unknown = [f for f in chosen if f not in available]
    if unknown:
        raise SystemExit(
            f"unknown reference feature(s) {unknown}. normalisation.py "
            f"references: {', '.join(available)}")
    return chosen


def _collect(vectors: Sequence[Dict[str, float]],
             name: str) -> Tuple[List[Optional[float]], str]:
    """Pull one referenced feature out of the ensemble vectors.

    ``build_ensemble_vector`` namespaces every detector feature as
    ``{engine}__f_{key}``, so the reference name is looked up as an exact
    per-engine key first and only then by suffix.  Engine order breaks ties, so
    ``perplexity`` resolves to the probability engine rather than to the
    humanization engine's copy of it.  A row where the owning engine was
    unavailable yields ``None`` and is dropped, never zero-filled.
    """
    from app import pipeline

    engines = [engine_name for engine_name, _ in pipeline.ENGINES]
    candidates = [f"{engine}__f_{name}" for engine in engines]

    present = {key for vector in vectors for key in vector}
    key = next((c for c in candidates if c in present), None)
    if key is None:
        suffix = f"__{name}"
        matches = [k for k in sorted(present)
                   if "__f_" in k and k.endswith(suffix)]
        matches.sort(key=lambda k: engines.index(k.split("__", 1)[0])
                     if k.split("__", 1)[0] in engines else len(engines))
        key = matches[0] if matches else None
    if key is None:
        return [None] * len(vectors), ""
    return [vector.get(key) for vector in vectors], key


# --------------------------------------------------------------------------
# fitting one reference
# --------------------------------------------------------------------------


def _fit_reference(name: str, values: Sequence[Optional[float]],
                   is_ai: Sequence[bool], source: str, midpoint_rule: str,
                   min_samples: int, min_auc: float
                   ) -> Tuple[Optional[Dict[str, Any]], str]:
    import numpy as np

    if not source:
        return None, "not produced by any engine in this run"

    pairs = [(float(v), flag) for v, flag in zip(values, is_ai)
             if v is not None and np.isfinite(float(v))]
    human = np.asarray([v for v, flag in pairs if not flag], dtype="float64")
    ai = np.asarray([v for v, flag in pairs if flag], dtype="float64")

    if len(human) < min_samples or len(ai) < min_samples:
        return None, (f"only {len(human)} human / {len(ai)} AI measurements "
                      f"(need {min_samples} of each)")
    if np.std(np.concatenate([human, ai])) <= 0:
        return None, "the feature is constant across the whole corpus"

    direction = 1 if float(np.median(ai)) >= float(np.median(human)) else -1
    oriented = np.concatenate([human, ai]) * direction
    binary = np.concatenate([np.zeros(len(human)), np.ones(len(ai))])

    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(binary, oriented))
    if auc < min_auc:
        return None, (f"AUC {auc:.3f} is below --min-auc {min_auc}; the classes "
                      "are not separated by this feature on this corpus")

    false_positive, true_positive, thresholds = roc_curve(binary, oriented)
    finite = np.isfinite(thresholds)
    youden = (true_positive - false_positive)[finite]
    best_threshold = float(thresholds[finite][int(np.argmax(youden))])

    median_midpoint = (float(np.median(human)) + float(np.median(ai))) / 2.0
    midpoint = (best_threshold * direction if midpoint_rule == "youden"
                else median_midpoint)

    separation = abs(float(np.median(ai)) - float(np.median(human)))
    pooled_std = float(np.sqrt((np.var(human) + np.var(ai)) / 2.0))
    scale = max(separation / 2.0, 0.25 * pooled_std, 1e-6)

    default = normalisation.DEFAULT_REFERENCES.get(name)
    note = (f"fitted on {len(human)} human / {len(ai)} AI documents from "
            f"{source}; midpoint rule={midpoint_rule}, AUC={auc:.3f}"
            + ("" if default is None or default.direction == direction else
               f"; NOTE direction flipped from the shipped prior "
               f"({default.direction:+d} -> {direction:+d})"))

    return {
        "midpoint": round(float(midpoint), 6),
        "scale": round(float(scale), 6),
        "direction": int(direction),
        "note": note,
        "diagnostics": {
            "source_feature": source,
            "n_human": int(len(human)),
            "n_ai": int(len(ai)),
            "median_human": round(float(np.median(human)), 6),
            "median_ai": round(float(np.median(ai)), 6),
            "pooled_std": round(pooled_std, 6),
            "auc": round(auc, 4),
            "youden_midpoint": round(float(best_threshold * direction), 6),
            "median_midpoint": round(median_midpoint, 6),
        },
    }, ""


def _print_comparison(fitted: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== fitted vs shipped prior ===")
    print("  feature                       fitted mid   prior mid   "
          "fitted scale  prior scale  dir")
    for name, entry in fitted.items():
        prior = normalisation.DEFAULT_REFERENCES.get(name)
        prior_mid = f"{prior.midpoint:+.4f}" if prior else "     -"
        prior_scale = f"{prior.scale:.4f}" if prior else "    -"
        flag = "" if prior is None or prior.direction == entry["direction"] \
            else "  <- direction flipped"
        print(f"  {name:28s} {entry['midpoint']:+10.4f}  {prior_mid:>10s}  "
              f"{entry['scale']:12.4f} {prior_scale:>12s}  "
              f"{entry['direction']:+d}{flag}")


if __name__ == "__main__":
    raise SystemExit(main())
