"""
Fine-tune the Engine-A sequence classifier on human / pure_ai / humanized_ai.

    python -m training.train_transformer --data data/corpus.jsonl --epochs 3

Why fine-tune at all
--------------------
Every publicly available AI-text classifier this project can load is **two
class**.  Such a checkpoint cannot separate pure AI from humanized AI - asking
it to is a category error, and ``transformer_detector`` refuses to fake it: it
reports ``models_humanized=False`` and lets other engines supply the split.  A
three-class checkpoint fitted here is what removes that limitation, and it is
expected to become the strongest single component in the system.

The label contract
------------------
``id2label`` is written as exactly::

    {0: "human", 1: "pure_ai", 2: "humanized_ai"}

Those three strings are matched natively by ``transformer_detector._map_label``,
so the detector maps them with no substring guessing at all.  Renaming them to
anything cuter (``ai``, ``paraphrased``, ``LABEL_2``) still works through the
fallback rules, but the fallback is there for foreign checkpoints, not for ours.

``model_loader.load_sequence_classifier`` prepends ``models/transformer`` to its
candidate list when that directory is non-empty, so the moment this script
finishes, the service loads the fine-tuned model instead of the public one.  No
configuration change is needed - and that also means a half-written output
directory will be picked up, which is why the model is saved only after
training completes.

What is included
----------------
* class weighting in the loss, because a corpus with 5% humanized rows otherwise
  produces a model that never predicts that class and still scores well;
* early stopping on validation **macro** F1 - not accuracy, which a three-class
  imbalanced problem rewards for ignoring the small class;
* evaluation through ``common.evaluate_predictions``, the same metric suite the
  other components report.

Failure behaviour
-----------------
Missing ``transformers`` / ``datasets`` / ``accelerate``, or a base model that
cannot be downloaded, are reported as errors with the remedy spelled out.  This
script never falls back to a smaller model or continues with random weights: a
silently degraded fine-tune is indistinguishable from a good one until it is in
production.

Honest limitations
------------------
* Transformer detectors overfit to the generator distribution they were fitted
  on faster than any other component here.  Hold generators out and read the
  unseen-generator slice from ``training/evaluate.py`` before believing
  anything.
* 512 tokens is the usual ceiling.  Long documents are chunked at inference
  time, so fine-tune on chunk-sized examples if your corpus is much longer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from training import common, support  # noqa: E402

#: the label order the detector expects, by index
LABELS: List[str] = list(config.CLASSES)          # human, pure_ai, humanized_ai
LABEL_TO_ID: Dict[str, int] = {label: index for index, label in enumerate(LABELS)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_transformer",
        description="Fine-tune a three-class sequence classifier for Engine A.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default=config.MODELS.detector_model,
                        help="HuggingFace id or local path "
                             f"(default: {config.MODELS.detector_model})")
    parser.add_argument("--out", default=str(config.ArtifactPaths().transformer_dir))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    parser.add_argument("--fp16", action="store_true",
                        help="mixed precision; requires a CUDA device")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="steps to accumulate before an optimiser update")
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2,
                        help="early-stopping patience in evaluations")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="disable inverse-frequency class weighting")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="refuse to fine-tune below this many training rows")
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate arguments, data and the environment, "
                             "then stop before training")
    return parser


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def require_dependencies() -> Dict[str, str]:
    """Fail with the remedy, not with a traceback."""
    missing: List[str] = []
    versions: Dict[str, str] = {}
    for package in ("torch", "transformers", "datasets", "accelerate"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "cannot fine-tune: missing " + ", ".join(missing) + ".\n"
            "  pip install " + " ".join(missing) + "\n"
            "  (a GPU build of torch is strongly recommended; on CPU this "
            "script is measured in hours per epoch)")
    return versions


def load_base_model(name: str, max_length: int):
    """Load tokenizer + model with the three-class head, or explain why not."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=len(LABELS),
            id2label={index: label for index, label in enumerate(LABELS)},
            label2id=dict(LABEL_TO_ID),
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # noqa: BLE001 - the message is the product here
        raise SystemExit(
            f"could not load base model '{name}': {type(exc).__name__}: {exc}\n"
            "  * check network access to huggingface.co (this environment may "
            "block it)\n"
            "  * or pre-download the checkpoint and pass a local directory to "
            "--base-model\n"
            "  * or set HF_HOME to a cache that already contains it\n"
            "  Refusing to continue: training from a different checkpoint than "
            "you asked for would produce an artefact nobody can reproduce."
        ) from exc

    model.config.id2label = {index: label for index, label in enumerate(LABELS)}
    model.config.label2id = dict(LABEL_TO_ID)
    if getattr(tokenizer, "model_max_length", max_length) < max_length:
        print(f"NOTE: tokenizer caps sequences at "
              f"{tokenizer.model_max_length} tokens; --max-length "
              f"{max_length} will be clipped to that.")
    return tokenizer, model


def training_arguments(**kwargs):
    """Build ``TrainingArguments`` across transformers' renamed keywords.

    ``evaluation_strategy`` became ``eval_strategy``; unknown keywords raise
    ``TypeError`` rather than being ignored, so each rename is tried in turn and
    anything still unsupported is dropped with a printed note instead of taking
    the whole run down.
    """
    from transformers import TrainingArguments

    renames = {"evaluation_strategy": "eval_strategy"}
    attempt = dict(kwargs)
    for _ in range(len(attempt) + len(renames) + 1):
        try:
            return TrainingArguments(**attempt)
        except TypeError as exc:
            message = str(exc)
            culprit = next((k for k in list(attempt)
                            if f"'{k}'" in message or f"{k}=" in message), None)
            if culprit is None:
                raise
            if culprit in renames and renames[culprit] not in attempt:
                attempt[renames[culprit]] = attempt.pop(culprit)
                continue
            print(f"NOTE: this transformers version does not accept "
                  f"'{culprit}'; continuing without it.")
            attempt.pop(culprit)
    raise SystemExit(
        "TrainingArguments could not be constructed for this transformers "
        "version; pin transformers>=4.30 or report this.")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def build_datasets(splits: Dict[str, List[Any]], tokenizer, max_length: int):
    from datasets import Dataset

    output = {}
    for name, records in splits.items():
        if not records:
            continue
        encoded = tokenizer([r.text for r in records], truncation=True,
                            max_length=max_length)
        output[name] = Dataset.from_dict({
            **encoded,
            "labels": [LABEL_TO_ID[r.label] for r in records],
        })
    return output


def class_weights(records: Sequence[Any]):
    """Inverse-frequency weights, normalised to mean 1."""
    import torch

    counts = [max(1, sum(1 for r in records if r.label == label))
              for label in LABELS]
    total = float(sum(counts))
    weights = [total / (len(LABELS) * count) for count in counts]
    mean = sum(weights) / len(weights)
    return torch.tensor([w / mean for w in weights], dtype=torch.float32)


def build_trainer_class(weights):
    """Trainer subclass applying class weights inside the loss."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=None if weights is None else weights.to(logits.device))
            loss = loss_fn(logits.view(-1, len(LABELS)), labels.view(-1))
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def build_compute_metrics():
    import numpy as np

    def compute_metrics(evaluation):
        logits = evaluation.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits, dtype="float64")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        y_true = [LABELS[int(i)] for i in evaluation.label_ids]
        metrics = common.evaluate_predictions(y_true, probabilities, LABELS)
        flat = {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "brier": metrics["brier_score"],
            "ece": metrics["expected_calibration_error"],
        }
        for klass, entry in metrics["per_class"].items():
            flat[f"f1_{klass}"] = entry["f1"]
        binary = metrics.get("binary_ai_vs_human") or {}
        if binary:
            flat["ai_auroc"] = binary["auroc"]
            flat["tpr_at_1pct_fpr"] = binary["tpr_at_1pct_fpr"]
        return flat

    return compute_metrics


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)
    out_dir = Path(args.out)

    versions = require_dependencies()
    print(f"environment: {versions}")

    import torch

    if args.fp16 and not torch.cuda.is_available():
        raise SystemExit(
            "--fp16 was requested but no CUDA device is visible. Mixed "
            "precision on CPU either errors or silently runs in fp32; drop "
            "--fp16 or run this on a GPU.")
    if args.gradient_accumulation < 1:
        raise SystemExit("--gradient-accumulation must be at least 1")

    records = common.read_jsonl(data_path)
    print(f"loaded {len(records)} records from {data_path}")
    print(f"dataset: {common.dataset_summary(records)}")
    missing = [label for label in LABELS if not any(r.label == label
                                                    for r in records)]
    if missing:
        raise SystemExit(
            f"the dataset has no examples of {missing}. A three-class head "
            "fitted without them emits a class it has never seen, which is "
            "worse than the two-class checkpoint it replaces.")

    ratios = support.parse_ratios(args.ratios)
    splits = support.random_split(records, ratios, args.seed)
    support.print_split(splits)
    if len(splits["train"]) < args.min_samples:
        raise SystemExit(
            f"only {len(splits['train'])} training rows; at least "
            f"{args.min_samples} are required to fine-tune a transformer. "
            "Below that the model memorises the corpus and every metric it "
            "reports is meaningless.")
    if not splits["validation"]:
        raise SystemExit(
            "the validation split is empty, so early stopping has nothing to "
            "watch. Adjust --ratios.")

    effective_batch = args.batch_size * max(1, args.gradient_accumulation)
    print(f"\nbase model      : {args.base_model}")
    print(f"labels          : {dict(enumerate(LABELS))}")
    print(f"epochs          : {args.epochs}")
    print(f"batch (effective): {args.batch_size} x "
          f"{args.gradient_accumulation} = {effective_batch}")
    print(f"lr / max_length : {args.lr} / {args.max_length}")
    print(f"device          : {'cuda' if torch.cuda.is_available() else 'cpu'}"
          f"{' (fp16)' if args.fp16 else ''}")
    print(f"output          : {out_dir}")

    tokenizer, model = load_base_model(args.base_model, args.max_length)
    print(f"loaded base model with {model.num_parameters():,} parameters")

    weights = None if args.no_class_weights else class_weights(splits["train"])
    if weights is not None:
        print(f"class weights   : "
              f"{dict(zip(LABELS, [round(float(w), 4) for w in weights]))}")

    if args.dry_run:
        print("\n--dry-run: environment, data and base model are usable; "
              "stopping before training.")
        return 0

    from transformers import DataCollatorWithPadding, EarlyStoppingCallback

    datasets = build_datasets(splits, tokenizer, args.max_length)
    arguments = training_arguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
    )

    trainer_class = build_trainer_class(weights)
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("\ntraining ...")
    trainer.train()

    evaluations: Dict[str, Dict[str, Any]] = {}
    for split_name in ("validation", "test"):
        dataset = datasets.get(split_name)
        if dataset is None:
            continue
        metrics = _evaluate(trainer, dataset, split_name)
        evaluations[split_name] = metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "training_summary.json").write_text(
        json.dumps({
            "base_model": args.base_model,
            "labels": LABELS,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "seed": args.seed,
            "class_weights": None if weights is None
            else [round(float(w), 6) for w in weights],
            "trained_on": common.stamp(data_path, splits["train"]),
            "metrics": evaluations,
        }, indent=2, default=str), encoding="utf-8")
    print(f"\nmodel written to {out_dir}")
    print("model_loader.load_sequence_classifier() prefers this directory, so "
          "the service now uses it with no configuration change.")

    support.write_report(args.report, {
        "component": "transformer",
        "dataset": str(data_path),
        "base_model": args.base_model,
        "labels": LABELS,
        "environment": versions,
        "split": support.describe_split(splits),
        "metrics": evaluations,
        "artefact": str(out_dir),
    })
    return 0


def _evaluate(trainer, dataset, split_name: str) -> Dict[str, Any]:
    import numpy as np

    output = trainer.predict(dataset)
    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype="float64")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y_true = [LABELS[int(i)] for i in output.label_ids]
    metrics = common.evaluate_predictions(y_true, probabilities, LABELS)
    common.print_metrics(f"transformer / {split_name}", metrics)
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
