"""
Standalone evaluation CLI - the report that is allowed to be believed.

    python -m training.evaluate --data data/corpus.jsonl --target ensemble

Read this first
---------------
**Overall accuracy alone is not an acceptable report for this system.**  A
single number on a pooled test set hides every failure mode that matters:

* the detector that is 96% accurate overall and 61% on humanized text;
* the detector that is excellent on the two generators in the training set and
  no better than chance on the third;
* the detector whose false-positive rate on human writing is 2% for native
  speakers and 14% for second-language writers;
* the detector that is fine on 500-word essays and unusable on 90-word answers.

Every one of those ships as "94% accurate".  So this tool reports the metric
suite **sliced**: overall, per true label, per category, per length bucket, per
generator model (with a separate unseen-generator section when generators are
held out), per humanizer and attack type, and across the raw / paraphrased /
human-edited transformation families.

TPR at fixed FPR, not accuracy at 0.5
-------------------------------------
The deployment question is never "what is the accuracy".  It is "how much
machine text do we catch if we are only willing to wrongly accuse 1% of honest
writers".  Every slice therefore reports TPR@1%FPR and TPR@5%FPR.  For slices
that contain no human rows - a per-generator slice is all AI by construction -
the false-positive rate is measured against the human rows of the whole
evaluation set, and the output says which negatives were used.

Brier score and ECE are reported everywhere too, because a detector that is
right for the wrong reasons with 0.99 confidence is a liability, not an asset.

Targets
-------
``ensemble``      the full pipeline: every engine, the meta-classifier, and the
                  calibrator (unless ``--no-calibration``)
``stylometry``    Engine E's artefact alone
``humanization``  Engine G's artefact alone
``transformer``   Engine A's checkpoint alone

Slices smaller than ``--min-slice`` rows are listed but not scored: a macro-F1
computed on four documents is noise with a decimal point.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.features import vectorizer  # noqa: E402
from training import common, support  # noqa: E402

TARGETS = ("ensemble", "stylometry", "humanization", "transformer")

PARAPHRASE_MARKERS = ("paraphras", "spin", "rewrit", "quillbot", "undetectable",
                      "humaniz", "humanis", "synonym", "back_translat",
                      "backtranslat")
HUMAN_EDIT_MARKERS = ("human_edit", "human-edit", "manual", "light_edit",
                      "heavy_edit", "post_edit", "hand_edit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.evaluate",
        description="Sliced evaluation of a trained component.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="ensemble", choices=TARGETS)
    parser.add_argument("--model", default=None,
                        help="artefact path override for the chosen target")
    parser.add_argument("--mode", default="standard",
                        choices=sorted(config.MODE_DETECTORS),
                        help="analysis mode for --target ensemble/transformer")
    parser.add_argument("--cache", default=None,
                        help="ensemble-vector cache shared with train_ensemble")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--split", default="all",
                        choices=("all", "train", "validation", "test"),
                        help="evaluate one split of --ratios/--seed, or all "
                             "rows (default: all)")
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-generators", default=None,
                        help="comma separated generator_model values that were "
                             "NOT in training; reported as their own section")
    parser.add_argument("--held-out-humanizers", default=None,
                        help="comma separated humanizer values that were NOT "
                             "in training")
    parser.add_argument("--no-calibration", action="store_true",
                        help="report the meta-classifier's raw probabilities")
    parser.add_argument("--min-slice", type=int, default=10,
                        help="smallest slice that gets scored (default 10)")
    parser.add_argument("--json", default=None,
                        help="dump every slice's full metrics here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)
    classes = list(config.CLASSES)

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")

    if args.split != "all":
        ratios = support.parse_ratios(args.ratios)
        splits = support.random_split(records, ratios, args.seed)
        records = splits[args.split]
        print(f"evaluating the '{args.split}' split: {len(records)} records")
    if not records:
        raise SystemExit("nothing to evaluate")
    print(f"dataset: {common.dataset_summary(records)}")

    probabilities, provenance = _predict(args, records, classes)
    labels = [r.label for r in records]

    print(f"\ntarget    : {args.target}")
    for key, value in provenance.items():
        print(f"{key:10s}: {value}")

    slices = _build_slices(records, args)
    report: Dict[str, Any] = {
        "component": args.target,
        "dataset": str(data_path),
        "split": args.split,
        "n": len(records),
        "provenance": provenance,
        "classes": classes,
        "slices": {},
    }

    overall = common.evaluate_predictions(labels, probabilities, classes)
    common.print_metrics(f"{args.target} / OVERALL", overall)
    _print_confusion(overall)
    report["overall"] = overall

    human_pool = [i for i, label in enumerate(labels) if label == "human"]
    for group_name, members in slices:
        rows = _score_group(members, labels, probabilities, classes,
                            human_pool, args.min_slice)
        report["slices"][group_name] = rows
        _print_group(group_name, rows)

    print("\nReminder: read the weakest slice, not the overall row. A component "
          "is only as trustworthy as its worst-supported population.")

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print(f"\nfull report written to {destination}")
    return 0


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------


def _predict(args, records: Sequence[Any],
             classes: List[str]) -> Tuple[Any, Dict[str, Any]]:
    if args.target == "ensemble":
        return _predict_ensemble(args, records, classes)
    if args.target == "stylometry":
        return _predict_vector_model(
            args, records, classes,
            path=Path(args.model or config.ArtifactPaths().stylometry_model),
            extractor=common.extract_stylometric_vectors,
            trainer="python -m training.train_stylometry")
    if args.target == "humanization":
        return _predict_vector_model(
            args, records, classes,
            path=Path(args.model or config.ArtifactPaths().humanization_model),
            extractor=common.extract_humanization_vectors,
            trainer="python -m training.train_humanization")
    return _predict_transformer(args, records, classes)


def _load_artefact(path: Path, trainer: str) -> Dict[str, Any]:
    import joblib

    if not path.exists():
        raise SystemExit(
            f"no artefact at {path}. There is nothing to evaluate - the "
            "component is running its labelled fallback, and evaluating a "
            f"fallback as if it were a model would be dishonest. Run {trainer} "
            "first.")
    artefact = joblib.load(path)
    if not isinstance(artefact, dict) or "model" not in artefact \
            or "feature_names" not in artefact:
        raise SystemExit(f"{path} is not a valid artefact")
    return artefact


def _predict_vector_model(args, records, classes: List[str], path: Path,
                          extractor, trainer: str):
    artefact = _load_artefact(path, trainer)
    vectors, _ = extractor(records)
    X = vectorizer.stack(vectors, list(artefact["feature_names"]))
    model = artefact["model"]
    model_classes = [str(c) for c in model.classes_]
    probabilities = support.project_probabilities(
        model.predict_proba(X), model_classes, classes)
    return probabilities, {
        "artefact": str(path),
        "algorithm": artefact.get("algorithm"),
        "classes": model_classes,
        "trained_on": artefact.get("trained_on", {}).get("dataset"),
        "calibrated": False,
    }


def _predict_ensemble(args, records, classes: List[str]):
    from app.ensemble import calibrator as calibrator_module
    from training import train_ensemble

    path = Path(args.model or config.ArtifactPaths().meta_classifier)
    artefact = _load_artefact(path, "python -m training.train_ensemble")
    vectors, _ = train_ensemble.extract_vectors(
        records, mode=args.mode, cache=args.cache, refresh=args.refresh_cache)
    X = vectorizer.stack(vectors, list(artefact["feature_names"]))
    model = artefact["model"]
    model_classes = [str(c) for c in model.classes_]
    probabilities = support.project_probabilities(
        model.predict_proba(X), model_classes, classes)

    calibrator = calibrator_module.load()
    if args.no_calibration or not calibrator.fitted:
        return probabilities, {
            "artefact": str(path),
            "algorithm": artefact.get("algorithm"),
            "analysis_mode": args.mode,
            "calibrated": False,
            "note": ("--no-calibration" if args.no_calibration else
                     "no calibrator installed; these are raw scores and must "
                     "not be read as probabilities"),
        }

    import numpy as np

    adjusted = []
    for row in probabilities:
        mapping = {c: float(v) for c, v in zip(classes, row)}
        applied = calibrator.apply(mapping)
        adjusted.append([float(applied.get(c, 0.0)) for c in classes])
    return np.asarray(adjusted, dtype="float64"), {
        "artefact": str(path),
        "algorithm": artefact.get("algorithm"),
        "analysis_mode": args.mode,
        "calibrated": True,
        "calibration": calibrator.describe(),
    }


def _predict_transformer(args, records, classes: List[str]):
    import numpy as np

    from app.detectors.transformer_detector import TransformerDetector

    detector = TransformerDetector()
    rows: List[List[float]] = []
    unavailable = 0
    for index, record in enumerate(records):
        if index and index % 20 == 0:
            print(f"  ... {index}/{len(records)} documents scored", flush=True)
        context = common.build_context(record.text, mode=args.mode)
        result = detector.analyse(context)
        if not result.available or not result.class_probabilities:
            unavailable += 1
            if unavailable == 1:
                raise SystemExit(
                    "the transformer engine is unavailable: "
                    f"{result.reason}. Evaluating it is impossible until a "
                    "checkpoint loads.")
        rows.append([float(result.class_probabilities.get(c, 0.0))
                     for c in classes])
    checkpoint = (detector.analyse(
        common.build_context(records[0].text, mode=args.mode)).raw or {})
    return np.asarray(rows, dtype="float64"), {
        "checkpoint": checkpoint.get("checkpoint"),
        "label_space": checkpoint.get("label_space"),
        "models_humanized_class": checkpoint.get("models_humanized_class"),
        "calibrated": False,
    }


# --------------------------------------------------------------------------
# slices
# --------------------------------------------------------------------------


def _build_slices(records: Sequence[Any],
                  args) -> List[Tuple[str, Dict[str, List[int]]]]:
    def group(key) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for index, record in enumerate(records):
            out.setdefault(str(key(record)), []).append(index)
        return dict(sorted(out.items()))

    slices: List[Tuple[str, Dict[str, List[int]]]] = [
        ("true label", group(lambda r: r.label)),
        ("category", group(lambda r: r.category)),
        ("length bucket", group(lambda r: r.length_bucket)),
        ("generator model", group(lambda r: r.generator_model or "-")),
        ("humanizer", group(lambda r: r.humanizer or "-")),
        ("attack type", group(lambda r: r.attack_type or "-")),
        ("transformation family", group(_transformation_family)),
        ("source", group(lambda r: r.source)),
    ]

    held_out = _split_list(args.held_out_generators)
    if held_out:
        slices.append(("unseen generator", _seen_unseen(
            records, held_out, lambda r: r.generator_model)))
    held_out_humanizers = _split_list(args.held_out_humanizers)
    if held_out_humanizers:
        slices.append(("unseen humanizer", _seen_unseen(
            records, held_out_humanizers, lambda r: r.humanizer)))
    return slices


def _split_list(value: Optional[str]) -> List[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _seen_unseen(records: Sequence[Any], held_out: Sequence[str],
                 key) -> Dict[str, List[int]]:
    wanted = set(held_out)
    out: Dict[str, List[int]] = {"unseen (held out)": [], "seen in training": [],
                                 "human (no generator)": []}
    for index, record in enumerate(records):
        value = key(record)
        if record.label == "human" or value is None:
            out["human (no generator)"].append(index)
        elif str(value) in wanted:
            out["unseen (held out)"].append(index)
        else:
            out["seen in training"].append(index)
    return {k: v for k, v in out.items() if v}


def _transformation_family(record) -> str:
    """raw AI vs paraphrased vs human-edited, from whatever metadata exists."""
    if record.label == "human":
        return "human"
    marker = f"{record.attack_type or ''} {record.humanizer or ''}".lower()
    if any(token in marker for token in HUMAN_EDIT_MARKERS):
        return "human_edited"
    if any(token in marker for token in PARAPHRASE_MARKERS):
        return "paraphrased"
    if record.label == "humanized_ai":
        return "humanized (unspecified method)"
    return "raw_ai"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def _score_group(members: Dict[str, List[int]], labels: Sequence[str],
                 probabilities, classes: List[str], human_pool: List[int],
                 min_slice: int) -> Dict[str, Any]:
    import numpy as np

    probabilities = np.asarray(probabilities, dtype="float64")
    rows: Dict[str, Any] = {}
    for name, indices in members.items():
        entry: Dict[str, Any] = {
            "n": len(indices),
            "label_mix": dict(Counter(labels[i] for i in indices)),
        }
        if len(indices) < min_slice:
            entry["scored"] = False
            entry["reason"] = (f"only {len(indices)} rows; below --min-slice "
                               f"{min_slice}, so any metric would be noise")
        else:
            entry["scored"] = True
            entry["metrics"] = common.evaluate_predictions(
                [labels[i] for i in indices], probabilities[indices], classes)
        entry["operating_point"] = _binary_view(
            indices, labels, probabilities, classes, human_pool)
        rows[name] = entry
    return rows


def _binary_view(indices: Sequence[int], labels: Sequence[str], probabilities,
                 classes: List[str], human_pool: Sequence[int]
                 ) -> Optional[Dict[str, Any]]:
    """TPR at fixed FPR for one slice.

    A slice with no human rows borrows the evaluation set's human rows as
    negatives, because "TPR at 1% FPR" is meaningless without negatives and
    silently dropping the number would hide the operating point that matters.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    index = {c: i for i, c in enumerate(classes)}
    ai_column = sum(probabilities[:, index[c]]
                    for c in ("pure_ai", "humanized_ai") if c in index)

    positives = [i for i in indices if labels[i] != "human"]
    negatives = [i for i in indices if labels[i] == "human"]
    source = "slice"
    if not positives:
        return {"available": False,
                "reason": "no AI rows in this slice; TPR is undefined"}
    if not negatives:
        negatives = [i for i in human_pool if i not in set(indices)]
        source = "all human rows in the evaluation set"
    if not negatives:
        return {"available": False,
                "reason": "no human rows anywhere to measure a false-positive "
                          "rate against"}

    scores = np.concatenate([ai_column[negatives], ai_column[positives]])
    binary = np.concatenate([np.zeros(len(negatives)), np.ones(len(positives))])
    return {
        "available": True,
        "negatives": source,
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "auroc": float(roc_auc_score(binary, scores)),
        "tpr_at_1pct_fpr": common.tpr_at_fpr(binary, scores, 0.01),
        "tpr_at_5pct_fpr": common.tpr_at_fpr(binary, scores, 0.05),
        "mean_ai_score_positive": float(ai_column[positives].mean()),
        "mean_ai_score_negative": float(ai_column[negatives].mean()),
    }


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------


def _print_confusion(metrics: Dict[str, Any]) -> None:
    labels = metrics.get("confusion_matrix_labels") or []
    matrix = metrics.get("confusion_matrix") or []
    if not matrix:
        return
    width = max(len(l) for l in labels) + 2
    print("\n  confusion matrix (rows = true, columns = predicted)")
    print(" " * (width + 2) + "".join(f"{l:>15s}" for l in labels))
    for label, row in zip(labels, matrix):
        print(f"  {label:<{width}s}" + "".join(f"{value:>15d}" for value in row))


def _print_group(title: str, rows: Dict[str, Any]) -> None:
    print(f"\n=== by {title} ===")
    header = (f"  {'slice':<28s} {'n':>5s} {'macroF1':>8s} {'AUROC':>7s} "
              f"{'TPR@1%':>7s} {'TPR@5%':>7s} {'Brier':>7s} {'ECE':>7s}")
    print(header)
    for name, entry in rows.items():
        operating = entry.get("operating_point") or {}
        metrics = entry.get("metrics") or {}
        if entry.get("scored"):
            macro = f"{metrics['macro_f1']:8.4f}"
            brier = f"{metrics['brier_score']:7.4f}"
            ece = f"{metrics['expected_calibration_error']:7.4f}"
        else:
            macro = f"{'-':>8s}"
            brier = ece = f"{'-':>7s}"
        if operating.get("available"):
            auroc = f"{operating['auroc']:7.4f}"
            tpr1 = f"{operating['tpr_at_1pct_fpr']:7.4f}"
            tpr5 = f"{operating['tpr_at_5pct_fpr']:7.4f}"
            marker = "*" if operating["negatives"] != "slice" else " "
        else:
            auroc = tpr1 = tpr5 = f"{'-':>7s}"
            marker = " "
        print(f"  {name[:28]:<28s} {entry['n']:5d} {macro} {auroc} "
              f"{tpr1} {tpr5} {brier} {ece}{marker}")
    if any((e.get("operating_point") or {}).get("negatives", "slice") != "slice"
           for e in rows.values()):
        print("  * false-positive rate measured against the evaluation set's "
              "human rows (this slice has none of its own)")
    unscored = [n for n, e in rows.items() if not e.get("scored")]
    if unscored:
        print(f"  not scored (too few rows): {', '.join(unscored)}")


if __name__ == "__main__":
    raise SystemExit(main())
