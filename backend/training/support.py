"""
Small shared helpers for the training CLIs.

Why this module exists
----------------------
Nine command line entry points need the same four boring things: a dataset
split, a way to map a split's records back to the feature rows extracted for
them, a JSON report writer, and a probability-column projection so every metric
in the project is computed in the canonical ``config.CLASSES`` order.  Copying
those into nine files is how they drift apart, and a split that drifts between
two trainers silently leaks validation data into training.

It deliberately contains no modelling logic.  Anything that decides what a model
learns lives in the trainer that owns it.

The split bridge
----------------
``training/splits.py`` owns the real splitting policy (grouped, stratified,
generator/humanizer aware).  It is imported defensively here because the
trainers must remain runnable while that module is being written: when it is
absent a deterministic, label-stratified fallback is used and a loud warning is
printed on every run.  The fallback is a stop-gap, not a policy - it knows
nothing about grouping by topic or holding out generators, so a model split by
it will report optimistic numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402

SPLIT_KEYS = ("train", "validation", "test")

_SPLIT_WARNING_SHOWN = False


# --------------------------------------------------------------------------
# dataset splitting
# --------------------------------------------------------------------------


def _splits_module():
    """Import ``training.splits`` if it exists, else ``None`` (once, loudly)."""
    global _SPLIT_WARNING_SHOWN
    try:
        from training import splits as splits_module  # type: ignore
    except ImportError as exc:
        if not _SPLIT_WARNING_SHOWN:
            _SPLIT_WARNING_SHOWN = True
            print(
                "WARNING: training/splits.py could not be imported "
                f"({exc}). Falling back to a deterministic label-stratified "
                "random split. That fallback does NOT group by topic and does "
                "NOT hold out generators or humanizers, so every metric it "
                "produces is optimistic. Install training/splits.py before "
                "trusting any number from this run.",
                file=sys.stderr, flush=True)
        return None
    return splits_module


def random_split(records: Sequence[Any], ratios: Sequence[float] = (0.7, 0.15, 0.15),
                 seed: int = 0) -> Dict[str, List[Any]]:
    """``{"train": [...], "validation": [...], "test": [...]}``.

    Delegates to ``training.splits.random_split`` when that module is present.
    Both the positional-tuple and keyword-mapping spellings of ``ratios`` are
    attempted, because the caller must not have to care which one that module
    settled on.
    """
    module = _splits_module()
    if module is not None:
        splitter = getattr(module, "random_split", None)
        if splitter is None:
            raise SystemExit(
                "training/splits.py exists but does not define random_split("
                "records, ratios, seed).")
        mapping = dict(zip(SPLIT_KEYS, ratios))
        for candidate in (tuple(ratios), mapping):
            try:
                result = splitter(list(records), candidate, seed)
            except (TypeError, ValueError):
                continue
            return _normalise_split_result(result)
        raise SystemExit(
            "training/splits.py: random_split rejected both the tuple and the "
            f"mapping spelling of ratios={tuple(ratios)!r}.")
    return _fallback_split(records, ratios, seed)


def _normalise_split_result(result: Any) -> Dict[str, List[Any]]:
    if not isinstance(result, dict):
        raise SystemExit(
            "training/splits.py: random_split must return a dict keyed by "
            f"{SPLIT_KEYS}, got {type(result).__name__}.")
    missing = [k for k in SPLIT_KEYS if k not in result]
    if missing:
        raise SystemExit(
            f"training/splits.py: random_split result is missing {missing}.")
    return {key: list(result[key]) for key in SPLIT_KEYS}


def _fallback_split(records: Sequence[Any], ratios: Sequence[float],
                    seed: int) -> Dict[str, List[Any]]:
    """Deterministic stratified split used only when splits.py is absent."""
    import random

    train_ratio, validation_ratio = float(ratios[0]), float(ratios[1])
    buckets: Dict[str, List[Any]] = {}
    for record in records:
        buckets.setdefault(getattr(record, "label", "unknown"), []).append(record)

    output: Dict[str, List[Any]] = {key: [] for key in SPLIT_KEYS}
    for label in sorted(buckets):
        group = list(buckets[label])
        random.Random(seed + hash(label) % 10_000).shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_validation = int(round(n * validation_ratio))
        # never let rounding starve validation/test of every example
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_validation = max(1, min(n_validation, n - n_train - 1))
        output["train"].extend(group[:n_train])
        output["validation"].extend(group[n_train:n_train + n_validation])
        output["test"].extend(group[n_train + n_validation:])
    return output


def parse_ratios(text: str) -> Tuple[float, float, float]:
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise SystemExit(
            f"--ratios needs three comma separated numbers, got {text!r}")
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--ratios must be numeric, got {text!r}") from None
    total = sum(values)
    if total <= 0:
        raise SystemExit("--ratios must sum to a positive number")
    return tuple(v / total for v in values)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# mapping split records back to extracted rows
# --------------------------------------------------------------------------


def row_indices(all_records: Sequence[Any],
                subset: Sequence[Any]) -> List[int]:
    """Positions of ``subset`` inside ``all_records``.

    Feature extraction happens once over the whole dataset, so each split has to
    be turned back into row numbers.  ``Record`` is an unhashable dataclass, so
    identity is used first; a text-keyed lookup is the fallback for splitters
    that copy or rebuild their records.  A record that matches neither is a bug
    worth stopping for, not a row worth silently dropping.
    """
    by_identity = {id(record): index for index, record in enumerate(all_records)}
    by_text: Dict[str, int] = {}
    for index, record in enumerate(all_records):
        by_text.setdefault(getattr(record, "text", ""), index)

    indices: List[int] = []
    for record in subset:
        index = by_identity.get(id(record))
        if index is None:
            index = by_text.get(getattr(record, "text", None))
        if index is None:
            raise SystemExit(
                "a split contains a record that is not in the source dataset; "
                "the splitter must return the records it was given")
        indices.append(index)
    return indices


# --------------------------------------------------------------------------
# probabilities
# --------------------------------------------------------------------------


def project_probabilities(probabilities, model_classes: Sequence[Any],
                          target_classes: Sequence[str] = config.CLASSES):
    """Reorder ``predict_proba`` output into ``target_classes`` order.

    A class the model never saw becomes a zero column and the rows are
    renormalised.  Every metric in the project is then computed on the same
    column order, whatever order the estimator happened to learn.
    """
    import numpy as np

    probabilities = np.asarray(probabilities, dtype="float64")
    lookup = {str(c): i for i, c in enumerate(model_classes)}
    columns = []
    for klass in target_classes:
        index = lookup.get(str(klass))
        columns.append(probabilities[:, index] if index is not None
                       else np.zeros(probabilities.shape[0]))
    stacked = np.stack(columns, axis=1)
    totals = stacked.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return stacked / totals


# --------------------------------------------------------------------------
# feature attribution
# --------------------------------------------------------------------------


def feature_importances(model, feature_names: Sequence[str], X=None, y=None,
                        top: int = 25, permutation: bool = False,
                        seed: int = 0) -> Dict[str, Any]:
    """What the model actually leaned on, in a form a human can read.

    Linear models report the mean absolute standardised coefficient (the
    pipelines built by ``common.build_estimator`` scale their inputs, so those
    magnitudes are comparable across features).  Tree ensembles report impurity
    importances.  Anything else - and any model when ``permutation=True`` - gets
    permutation importance measured on the data passed in, which is the only
    attribution that means the same thing for every estimator.
    """
    import numpy as np

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", model)

    values: Optional[Any] = None
    kind = ""
    if not permutation:
        coefficients = getattr(estimator, "coef_", None)
        importances = getattr(estimator, "feature_importances_", None)
        if coefficients is not None:
            values = np.abs(np.asarray(coefficients, dtype="float64"))
            values = values.mean(axis=0) if values.ndim > 1 else values
            kind = "mean_absolute_standardised_coefficient"
        elif importances is not None:
            values = np.asarray(importances, dtype="float64")
            kind = "impurity_importance"

    if values is None:
        if X is None or y is None or len(set(map(str, y))) < 2:
            return {"kind": "unavailable", "top": [],
                    "note": "no coefficients, no importances and no data to "
                            "permute"}
        from sklearn.inspection import permutation_importance

        outcome = permutation_importance(
            model, X, list(y), n_repeats=5, random_state=seed,
            scoring="f1_macro")
        values = np.asarray(outcome.importances_mean, dtype="float64")
        kind = "permutation_importance_macro_f1_drop"

    order = np.argsort(-np.abs(values))[:top]
    return {
        "kind": kind,
        "top": [{"feature": str(feature_names[i]),
                 "value": round(float(values[i]), 6)} for i in order],
        "note": ("Attribution describes this fitted model on this dataset. It "
                 "is not evidence that the feature causes AI-ness."),
    }


def print_importances(title: str, block: Dict[str, Any]) -> None:
    print(f"\n=== {title} ({block.get('kind')}) ===")
    if not block.get("top"):
        print(f"  unavailable: {block.get('note')}")
        return
    width = max(len(entry["feature"]) for entry in block["top"])
    for rank, entry in enumerate(block["top"], start=1):
        print(f"  {rank:2d}. {entry['feature']:<{width}s}  {entry['value']:+.6f}")


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def write_report(path: Optional[str], payload: Dict[str, Any]) -> Optional[Path]:
    if not path:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nreport written to {destination}")
    return destination


def describe_split(splits: Dict[str, List[Any]]) -> Dict[str, Any]:
    from collections import Counter

    return {
        key: {"n": len(records),
              "labels": dict(Counter(getattr(r, "label", "?") for r in records))}
        for key, records in splits.items()
    }


def print_split(splits: Dict[str, List[Any]]) -> None:
    print("\n=== split ===")
    for key, detail in describe_split(splits).items():
        print(f"  {key:11s} n={detail['n']:5d}  {detail['labels']}")
