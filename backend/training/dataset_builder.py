"""
Assemble a labelled three-class dataset from real sources.

    python -m training.dataset_builder \
        --human data/human/*.jsonl --ai data/ai/ \
        --generate-humanized --split random --out data/dataset.jsonl

What this tool does and does not do
-----------------------------------
It ingests text you already have, normalises it into the ``Record`` schema,
removes duplicates, enforces length limits, optionally derives ``humanized_ai``
samples with ``humanizers.py``, checks for topic leakage and writes JSONL.

It never invents text.  There is no "--synthesise-human" flag and there will
not be one: a human class built from generated prose is a contradiction, and a
detector trained on it would be measuring nothing.  If no sources are supplied
the tool exits and tells you what to collect.

The humanized class
-------------------
``--generate-humanized`` derives humanized samples from your AI pool using the
bundled simulated attacks.  Read the honesty note at the top of
``humanizers.py`` before believing any robustness number that comes out of it.
Real paraphraser output belongs in ``--external-humanized``, which passes text
through untouched; when both are present, keep the external pool for
evaluation rather than pooling the two.

The leakage check
-----------------
Every build runs ``splits.check_topic_leakage`` and prints the verdict.  If the
human pool and the AI pool do not share subject matter, the resulting
classifier learns the subject and not the author - it will look excellent in
validation and fail on the first real document it sees.  That check is the most
valuable thing this script does, so it cannot be disabled.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training import humanizers, splits
from training.common import (RECORD_FIELDS, Record, dataset_summary,
                             write_jsonl)

from app.core import config

RULE = "=" * 78


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Normalise line endings and strip trailing whitespace per line.

    Nothing more aggressive than that: encoding quirks and odd spacing are
    signals the detector is supposed to see, so cleaning them here would hide
    exactly the artefacts an obfuscation tool leaves behind.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _record_from_payload(payload: Dict[str, Any], label: str,
                         default_category: str,
                         default_source: str) -> Optional[Record]:
    text = _clean(payload.get("text") or "")
    if not text:
        return None
    fields = {k: v for k, v in payload.items()
              if k in RECORD_FIELDS and k not in ("text", "label", "split",
                                                  "length_bucket")}
    fields.setdefault("source", default_source)
    fields.setdefault("category", default_category)
    if not fields.get("category"):
        fields["category"] = default_category
    return Record(text=text, label=label, **fields)


def _load_jsonl(path: Path, label: str, default_category: str) -> List[Record]:
    """Read a JSONL file whose rows carry at least ``text``.

    Rows are not required to carry a label - the pool they were passed in with
    decides that, so the same file can be reused as ``--ai`` and as
    ``--external-humanized`` without editing it.
    """
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ! {path}:{line_number}: invalid JSON, skipped")
                continue
            if not isinstance(payload, dict):
                print(f"  ! {path}:{line_number}: not an object, skipped")
                continue
            record = _record_from_payload(payload, label, default_category,
                                          default_source=path.stem)
            if record is not None:
                records.append(record)
    return records


def _load_text_directory(root: Path, label: str,
                         default_category: str) -> List[Record]:
    """Read ``*.txt`` beneath a directory, one document per file.

    The immediate parent directory becomes the topic when the files are nested
    (``corpus/climate/essay1.txt`` -> topic ``climate``).  That convention is
    what makes the topic-leakage check usable on plain-text corpora; flat
    directories yield no topic and the check will say so rather than guess.
    """
    records: List[Record] = []
    for path in sorted(root.rglob("*.txt")):
        try:
            text = _clean(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            print(f"  ! {path}: {exc}, skipped")
            continue
        if not text:
            continue
        relative = path.relative_to(root)
        topic = relative.parts[-2] if len(relative.parts) > 1 else None
        records.append(Record(text=text, label=label, source=root.name,
                              category=default_category, topic=topic))
    return records


def load_sources(paths: Sequence[str], label: str,
                 default_category: str) -> List[Record]:
    records: List[Record] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise SystemExit(f"source not found: {path}")
        if path.is_dir():
            loaded = _load_text_directory(path, label, default_category)
        elif path.suffix.lower() in (".jsonl", ".json", ".ndjson"):
            loaded = _load_jsonl(path, label, default_category)
        elif path.suffix.lower() == ".txt":
            text = _clean(path.read_text(encoding="utf-8", errors="replace"))
            loaded = ([Record(text=text, label=label, source=path.stem,
                              category=default_category)] if text else [])
        else:
            raise SystemExit(
                f"unsupported source '{path}': pass a .jsonl file, a .txt file "
                "or a directory of .txt files")
        print(f"  {path} -> {len(loaded)} {label} samples")
        records.extend(loaded)
    return records


# --------------------------------------------------------------------------
# cleaning passes
# --------------------------------------------------------------------------


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_exact(records: Sequence[Record]) -> Tuple[List[Record], Dict[str, Any]]:
    """Drop exact repeats and report texts that carry conflicting labels.

    A text present under two labels is a data bug, not a hard example: keeping
    it teaches the model that the same document is both human and machine.
    """
    seen: Dict[str, Record] = {}
    kept: List[Record] = []
    duplicates = 0
    conflicts: List[str] = []
    for record in records:
        key = _key(record.text)
        previous = seen.get(key)
        if previous is None:
            seen[key] = record
            kept.append(record)
            continue
        duplicates += 1
        if previous.label != record.label:
            conflicts.append(f"{previous.label} vs {record.label}: "
                             f"{record.text[:60]!r}")
    return kept, {"removed": duplicates, "label_conflicts": conflicts[:20],
                  "n_label_conflicts": len(conflicts)}


def deduplicate_near(records: Sequence[Record],
                     threshold: float = 0.9) -> Tuple[List[Record], Dict[str, Any]]:
    """Keep one representative per near-duplicate cluster.

    Run *before* humanized samples are derived.  A light-edit humanized copy is
    by construction a near-duplicate of its source, so running this afterwards
    would delete the very class the dataset is being built for.
    """
    groups = splits.near_duplicate_groups(records, threshold=threshold)
    drop = {index for group in groups for index in group[1:]}
    kept = [record for index, record in enumerate(records) if index not in drop]
    return kept, {"removed": len(drop), "clusters": len(groups),
                  "threshold": threshold}


def enforce_length(records: Sequence[Record], min_words: int,
                   max_words: Optional[int]) -> Tuple[List[Record], Dict[str, Any]]:
    """Apply the length floor and ceiling.

    The floor defaults to the application's own ``MIN_WORDS``: below it the
    runtime refuses to classify at all, so training on shorter text would teach
    the model on inputs it will never be asked about.
    """
    kept: List[Record] = []
    too_short = too_long = 0
    for record in records:
        words = len(record.text.split())
        if words < min_words:
            too_short += 1
            continue
        if max_words is not None and words > max_words:
            too_long += 1
            continue
        kept.append(record)
    return kept, {"too_short": too_short, "too_long": too_long,
                  "min_words": min_words, "max_words": max_words}


# --------------------------------------------------------------------------
# humanized derivation
# --------------------------------------------------------------------------


def generate_humanized(ai_records: Sequence[Record], pipelines: Sequence[str],
                       ratio: float, seed: int,
                       min_words: int) -> Tuple[List[Record], Dict[str, Any]]:
    """Derive ``humanized_ai`` samples from the AI pool.

    Provenance is preserved deliberately: ``generator_model``, ``category`` and
    ``topic`` are copied from the source so that generator-held-out and
    domain-held-out splits keep a humanized sample on the same side as the AI
    text it came from.  Losing that link is the quiet way to leak.
    """
    if not ai_records or ratio <= 0:
        return [], {"generated": 0, "pipelines": list(pipelines)}
    unknown = [p for p in pipelines if p not in humanizers.PIPELINES]
    if unknown:
        raise SystemExit(
            f"unknown pipeline(s) {unknown}. Available: "
            f"{sorted(humanizers.PIPELINES)}")

    target = int(round(len(ai_records) * float(ratio)))
    generated: List[Record] = []
    attack_counter: Counter = Counter()
    pipeline_counter: Counter = Counter()
    unchanged = dropped_short = 0
    seen = {_key(r.text) for r in ai_records}

    for index in range(target):
        source = ai_records[index % len(ai_records)]
        pipeline = pipelines[index % len(pipelines)]
        text, applied = humanizers.apply_pipeline(
            source.text, pipeline, seed=seed + index)
        if not applied or _key(text) in seen:
            unchanged += 1
            continue
        if len(text.split()) < min_words:
            dropped_short += 1
            continue
        seen.add(_key(text))
        generated.append(Record(
            text=text,
            label="humanized_ai",
            source=source.source,
            generator_model=source.generator_model,
            category=source.category,
            humanizer=pipeline,
            language=source.language,
            attack_type=",".join(applied),
            topic=source.topic,
        ))
        attack_counter.update(applied)
        pipeline_counter[pipeline] += 1

    return generated, {
        "generated": len(generated),
        "requested": target,
        "unchanged_or_duplicate": unchanged,
        "dropped_below_min_words": dropped_short,
        "pipelines": dict(pipeline_counter),
        "attacks": dict(attack_counter),
        "note": ("simulated humanizers - see humanizers.py; real paraphraser "
                 "output must be supplied with --external-humanized"),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_summary(title: str, summary: Dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    print(f"  samples        : {summary['n_samples']}")
    print(f"  labels         : {summary['labels']}")
    print(f"  categories     : {summary['categories']}")
    print(f"  generators     : {summary['generators']}")
    print(f"  humanizers     : {summary['humanizers']}")
    print(f"  length buckets : {summary['length_buckets']}")
    print(f"  distinct topics: {summary['topics']}")


def print_leakage(report: Dict[str, Any]) -> None:
    """Print the leakage verdict where nobody can miss it."""
    severity = report.get("severity", "unknown")
    print()
    print(RULE)
    print(f"  TOPIC LEAKAGE CHECK: {severity.upper()}")
    print(RULE)
    print(f"  {report.get('verdict', '')}")
    length = report.get("length") or {}
    if length:
        print(f"  human mean {length.get('human_mean_words')} words, "
              f"machine mean {length.get('machine_mean_words')} words "
              f"(length-only AUC {length.get('separability_auc')})")
    topic = report.get("topic_overlap") or {}
    if topic:
        print(f"  topic overlap: {topic.get('shared')} shared, "
              f"coverage {topic.get('coverage'):.2f}")
    for warning in report.get("warnings", []):
        print(f"  ! {warning}")
    if severity == "fail":
        print()
        print("  " + "*" * 70)
        print("  *  DO NOT TRAIN ON THIS DATASET AS IT STANDS.")
        print("  *  The labels are predictable from something other than the")
        print("  *  writing: subject matter, source or document length. A model")
        print("  *  fitted here will report a high validation score and then")
        print("  *  misclassify real text, because it never learned authorship.")
        print("  *  Collect human and AI text on the SAME topics, from")
        print("  *  comparable sources, at comparable lengths, and rebuild.")
        print("  " + "*" * 70)
    print(RULE)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.dataset_builder",
        description="Assemble a labelled human / pure_ai / humanized_ai dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Sources may be .jsonl files (rows with a 'text' field, other "
                "Record fields optional), single .txt files, or directories of "
                ".txt files (nested one level per topic)."))
    parser.add_argument("--human", nargs="+", default=[], metavar="PATH",
                        help="sources of human-written text")
    parser.add_argument("--ai", nargs="+", default=[], metavar="PATH",
                        help="sources of raw model output")
    parser.add_argument("--external-humanized", nargs="+", default=[],
                        metavar="PATH", dest="external_humanized",
                        help="already-humanized AI text from a real tool "
                             "(ingested untouched - this is the pool that "
                             "gives an honest robustness number)")
    parser.add_argument("--generate-humanized", action="store_true",
                        help="derive humanized_ai samples from the AI pool "
                             "with the bundled simulated attacks")
    parser.add_argument("--humanized-ratio", type=float, default=1.0,
                        metavar="FLOAT",
                        help="humanized samples per AI sample (default 1.0)")
    parser.add_argument("--pipelines", nargs="+",
                        default=sorted(humanizers.PIPELINES),
                        metavar="NAME",
                        help=f"humanizer pipelines to use; available: "
                             f"{', '.join(sorted(humanizers.PIPELINES))}")
    parser.add_argument("--category", default="general",
                        choices=list(config.CATEGORIES),
                        help="category assigned when a source carries none")
    parser.add_argument("--min-words", type=int, default=config.MIN_WORDS,
                        help=f"drop samples below this (default "
                             f"{config.MIN_WORDS}, the runtime's own floor)")
    parser.add_argument("--max-words", type=int, default=None,
                        help="drop samples above this")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9,
                        help="Jaccard threshold for near-duplicate removal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="output JSONL path")
    parser.add_argument("--split", default="none",
                        choices=["none", "random", "generator", "domain",
                                 "humanizer"])
    parser.add_argument("--ratios", nargs=3, type=float,
                        default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"),
                        help="ratios for --split random")
    parser.add_argument("--test-generators", nargs="+", default=[],
                        metavar="NAME", help="held-out models for --split generator")
    parser.add_argument("--test-categories", nargs="+", default=[],
                        metavar="NAME", help="held-out categories for --split domain")
    parser.add_argument("--test-humanizers", nargs="+", default=[],
                        metavar="NAME",
                        help="held-out humanizers for --split humanizer")
    parser.add_argument("--report", default=None, metavar="PATH",
                        help="write a JSON report of the build")
    return parser


def _apply_split(records: List[Record],
                 args: argparse.Namespace) -> Optional[Dict[str, List[Record]]]:
    if args.split == "none":
        return None
    if args.split == "random":
        return splits.random_split(records, tuple(args.ratios), seed=args.seed)
    if args.split == "generator":
        if not args.test_generators:
            raise SystemExit("--split generator requires --test-generators")
        return splits.generator_held_out(records, args.test_generators,
                                         seed=args.seed)
    if args.split == "domain":
        if not args.test_categories:
            raise SystemExit("--split domain requires --test-categories")
        return splits.domain_held_out(records, args.test_categories,
                                      seed=args.seed)
    if args.split == "humanizer":
        if not args.test_humanizers:
            raise SystemExit("--split humanizer requires --test-humanizers")
        return splits.humanizer_held_out(records, args.test_humanizers,
                                         seed=args.seed)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not (args.human or args.ai or args.external_humanized):
        raise SystemExit(
            "no sources supplied - this tool assembles a dataset, it does not "
            "invent one.\n"
            "Provide at least:\n"
            "  --human PATH...   human-written text (JSONL rows with a 'text' "
            "field, or a directory of .txt files)\n"
            "  --ai PATH...      raw model output on the SAME topics as the "
            "human pool\n"
            "Optionally:\n"
            "  --external-humanized PATH...  output from a real paraphrase "
            "tool\n"
            "  --generate-humanized          derive humanized samples from the "
            "AI pool instead\n"
            "Human text must be collected, never generated: a synthetic human "
            "class makes every downstream number meaningless.")

    print(RULE)
    print("  ingesting sources")
    print(RULE)
    human = load_sources(args.human, "human", args.category)
    ai = load_sources(args.ai, "pure_ai", args.category)
    external = load_sources(args.external_humanized, "humanized_ai",
                            args.category)

    records = human + ai + external
    stages: Dict[str, Any] = {"ingested": len(records)}

    records, exact_stats = deduplicate_exact(records)
    stages["exact_duplicates"] = exact_stats
    print(f"\n  exact duplicates removed  : {exact_stats['removed']}")
    if exact_stats["n_label_conflicts"]:
        print(f"  ! {exact_stats['n_label_conflicts']} texts appeared under "
              "more than one label; the first occurrence was kept")

    records, near_stats = deduplicate_near(records,
                                           threshold=args.near_duplicate_threshold)
    stages["near_duplicates"] = near_stats
    print(f"  near duplicates removed   : {near_stats['removed']} "
          f"({near_stats['clusters']} clusters at Jaccard "
          f">= {near_stats['threshold']})")

    records, length_stats = enforce_length(records, args.min_words,
                                           args.max_words)
    stages["length_filter"] = length_stats
    print(f"  dropped below {args.min_words} words     : "
          f"{length_stats['too_short']}")
    if args.max_words:
        print(f"  dropped above {args.max_words} words   : "
              f"{length_stats['too_long']}")

    humanizer_stats: Dict[str, Any] = {"generated": 0}
    if args.generate_humanized:
        ai_pool = [r for r in records if r.label == "pure_ai"]
        if not ai_pool:
            print("\n  ! --generate-humanized was requested but the AI pool is "
                  "empty after filtering; nothing to derive from")
        else:
            derived, humanizer_stats = generate_humanized(
                ai_pool, args.pipelines, args.humanized_ratio, args.seed,
                args.min_words)
            records.extend(derived)
            print(f"\n  humanized samples derived : {len(derived)} "
                  f"from {len(ai_pool)} AI samples")
            print(f"  pipelines used            : "
                  f"{humanizer_stats.get('pipelines')}")
            print("  NOTE: these are simulated attacks. A commercial "
                  "paraphraser is stronger;")
            print("        robustness measured against them is an upper bound "
                  "on your real robustness.")
    stages["humanizer"] = humanizer_stats

    if not records:
        raise SystemExit(
            "every sample was filtered out. Check --min-words and that the "
            "sources actually contain text.")

    summary = dataset_summary(records)
    print_summary("dataset", summary)

    leakage = splits.check_topic_leakage(records)
    print_leakage(leakage)

    split_map = _apply_split(records, args)
    split_report: Dict[str, Any] = {}
    if split_map is not None:
        for name, group in split_map.items():
            print_summary(f"split: {name}", dataset_summary(group))
        split_report = splits.summarise_split(split_map)
        for warning in split_report.get("warnings", []):
            print(f"  ! {warning}")

    buckets = splits.length_stratified(records)
    print("\n--- length strata (evaluation lens) ---")
    for name in ("short", "medium", "long"):
        group = buckets.get(name, [])
        labels = dict(Counter(r.label for r in group))
        print(f"  {name:7s} n={len(group):5d} {labels}")

    out_path = Path(args.out).expanduser()
    written = write_jsonl(out_path, records)
    print(f"\n  wrote {written} records to {out_path}")

    if args.report:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
            "stages": stages,
            "summary": summary,
            "leakage": leakage,
            "split": {"strategy": args.split,
                      "sizes": ({k: len(v) for k, v in split_map.items()}
                                if split_map else {}),
                      "detail": split_report},
            "length_strata": {k: len(v) for k, v in buckets.items()},
            "output": str(out_path),
        }
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"  wrote report to {report_path}")

    if leakage.get("severity") == "fail":
        print("\n  Reminder: the leakage check FAILED. The file was written so "
              "you can inspect it,\n  but it is not fit for training in its "
              "current form.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
