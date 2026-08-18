"""
Dataset splitting with leakage control.

A detector's headline accuracy is mostly a statement about how its data was
split.  Three failure modes are common enough to be worth engineering against,
and this module exists to make each of them hard:

1.  **Near-duplicate straddle.**  Scraped corpora are full of reposts, boilerplate
    and templated intros.  If a document lands in train and its near-twin lands
    in test, the test number measures memorisation.  ``near_duplicate_groups``
    finds those clusters so a split can keep them together.

2.  **Topic straddle.**  The same essay prompt answered twice is not two
    independent samples.  ``random_split`` therefore splits *groups* - topic
    plus duplicate cluster - never individual records.

3.  **Topic-label confounding.**  The worst one, and the one nobody notices
    until deployment: the human pool is scraped from Reddit and the AI pool is
    generated essays, so "is this about video games or about photosynthesis"
    separates the classes perfectly and the model never learns authorship at
    all.  ``check_topic_leakage`` is built to catch exactly that and to say so
    in words rather than burying it in a metric.

Held-out splits (generator, domain, humanizer) answer the only question that
matters for deployment: does this thing work on a model, a subject area or a
paraphraser it has never seen?  Random-split numbers systematically flatter a
detector, so they should never be quoted on their own.

Every split function returns ``dict[str, list[Record]]`` and stamps
``Record.split`` in place so the assignment survives a round trip through JSONL.
"""

from __future__ import annotations

import random
import re
import zlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from training.common import CLASSES, Record, dataset_summary, length_bucket

# --------------------------------------------------------------------------
# near-duplicate detection
# --------------------------------------------------------------------------

SHINGLE_SIZE = 5           # characters per shingle
SKETCH_SIZE = 128          # bottom-k sketch width
MIN_SHARED_SKETCH = 8      # shared sketch values needed to become a candidate
MAX_POSTINGS = 400         # ignore sketch values shared by this many records


def _shingles(text: str) -> Set[int]:
    """Hashed character 5-grams of the whitespace-normalised, lower-cased text.

    Character shingles rather than word shingles because the humanizer attacks
    this project generates deliberately perturb individual characters; word
    shingles would call a typo-injected copy a different document.
    """
    normalised = re.sub(r"\s+", " ", text or "").strip().lower()
    if not normalised:
        return set()
    if len(normalised) <= SHINGLE_SIZE:
        return {zlib.crc32(normalised.encode("utf-8"))}
    return {
        zlib.crc32(normalised[i:i + SHINGLE_SIZE].encode("utf-8"))
        for i in range(len(normalised) - SHINGLE_SIZE + 1)
    }


def _sketch(shingles: Set[int]) -> Tuple[int, ...]:
    """Bottom-k sketch: the k smallest shingle hashes, ascending.

    A bottom-k sketch is a hash-order sample of the shingle set, so two
    documents that share most of their shingles share most of their sketch.
    That property is what makes the inverted index below a usable candidate
    generator without computing all n^2 pairs.
    """
    return tuple(sorted(shingles)[:SKETCH_SIZE])


def _jaccard(left: Set[int], right: Set[int]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / float(len(left) + len(right) - intersection)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self._parent[max(root_left, root_right)] = min(root_left, root_right)

    def groups(self) -> List[List[int]]:
        buckets: Dict[int, List[int]] = defaultdict(list)
        for item in range(len(self._parent)):
            buckets[self.find(item)].append(item)
        return [sorted(members) for _, members in sorted(buckets.items())]


def near_duplicate_groups(records: Sequence[Record],
                          threshold: float = 0.85) -> List[List[int]]:
    """Cluster near-duplicate texts, returning groups of record indices.

    Only groups with two or more members are returned; a record that appears in
    no group is unique at this threshold.

    Complexity, honestly stated
    ---------------------------
    Sketching is O(total characters).  Candidate generation is an inverted
    index over sketch values, which is O(n * SKETCH_SIZE) to build and then
    proportional to the number of *co-occurring* pairs to scan - not to n^2 in
    practice, but it degenerates towards n^2 if the corpus really is one
    document repeated.  Two guards keep that bounded: sketch values held by
    more than ``MAX_POSTINGS`` records are skipped (boilerplate shared by half
    the corpus tells you nothing), and a pair must share at least
    ``MIN_SHARED_SKETCH`` of ``SKETCH_SIZE`` values before exact verification.
    Verification recomputes both shingle sets and takes the true Jaccard, so
    the reported groups are exact at the threshold; only *recall* is
    approximate, and it is deliberately biased towards over-generating
    candidates.
    """
    count = len(records)
    if count < 2:
        return []
    threshold = max(0.0, min(float(threshold), 1.0))

    sketches: List[Tuple[int, ...]] = []
    for record in records:
        sketches.append(_sketch(_shingles(record.text)))

    postings: Dict[int, List[int]] = defaultdict(list)
    for index, sketch in enumerate(sketches):
        for value in sketch:
            postings[value].append(index)

    shared: Dict[int, Counter] = defaultdict(Counter)
    for value, holders in postings.items():
        if len(holders) < 2 or len(holders) > MAX_POSTINGS:
            continue
        for position, left in enumerate(holders):
            for right in holders[position + 1:]:
                shared[left][right] += 1

    union = _UnionFind(count)
    shingle_cache: Dict[int, Set[int]] = {}

    def _cached(index: int) -> Set[int]:
        if index not in shingle_cache:
            shingle_cache[index] = _shingles(records[index].text)
        return shingle_cache[index]

    for left, counter in shared.items():
        for right, hits in counter.items():
            if hits < MIN_SHARED_SKETCH:
                continue
            if union.find(left) == union.find(right):
                continue
            if _jaccard(_cached(left), _cached(right)) >= threshold:
                union.union(left, right)

    return [group for group in union.groups() if len(group) > 1]


# --------------------------------------------------------------------------
# grouping and assignment
# --------------------------------------------------------------------------


def _grouping(records: Sequence[Record],
              group_by_duplicates: bool = True,
              threshold: float = 0.85) -> List[List[int]]:
    """Indices grouped by topic and by near-duplicate cluster.

    Topic and duplicate constraints are unioned rather than applied in
    sequence, because a duplicate pair that carries two different topic labels
    must still stay on the same side of the split.
    """
    union = _UnionFind(len(records))
    by_topic: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.topic:
            by_topic[str(record.topic).strip().lower()].append(index)
    for members in by_topic.values():
        for other in members[1:]:
            union.union(members[0], other)
    if group_by_duplicates:
        for group in near_duplicate_groups(records, threshold=threshold):
            for other in group[1:]:
                union.union(group[0], other)
    return union.groups()


def _assign_groups(groups: Sequence[Sequence[int]], total: int,
                   ratios: Sequence[float], names: Sequence[str],
                   rng: random.Random) -> Dict[str, List[int]]:
    """Greedy group packing: each group goes to whichever split is furthest
    behind its target.  Shuffling first keeps the choice unbiased with respect
    to corpus order, which is otherwise usually sorted by source.
    """
    order = list(groups)
    rng.shuffle(order)
    # large groups first, so the biggest indivisible lumps are placed while
    # there is still room to balance around them
    order.sort(key=len, reverse=True)
    targets = [ratio * total for ratio in ratios]
    filled = [0.0] * len(names)
    buckets: Dict[str, List[int]] = {name: [] for name in names}
    for group in order:
        deficits = [targets[i] - filled[i] for i in range(len(names))]
        choice = max(range(len(names)), key=lambda i: deficits[i])
        buckets[names[choice]].extend(group)
        filled[choice] += len(group)
    return buckets


def _materialise(records: Sequence[Record],
                 buckets: Dict[str, List[int]]) -> Dict[str, List[Record]]:
    out: Dict[str, List[Record]] = {}
    for name, indices in buckets.items():
        chosen = [records[i] for i in sorted(indices)]
        for record in chosen:
            record.split = name
        out[name] = chosen
    return out


def random_split(records: Sequence[Record],
                 ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 seed: int = 0,
                 group_by_duplicates: bool = True) -> Dict[str, List[Record]]:
    """Grouped random split into train/validation/test.

    Groups - a topic, or a near-duplicate cluster - are never divided.  The
    realised ratios will therefore drift from the requested ones whenever the
    corpus contains a few very large topics; that drift is the honest price of
    not leaking, and ``summarise_split`` reports the actual sizes.
    """
    names = ("train", "validation", "test")
    if len(ratios) != 3:
        raise ValueError("ratios must be (train, validation, test)")
    total_ratio = float(sum(ratios))
    if total_ratio <= 0:
        raise ValueError("ratios must sum to a positive number")
    normalised = [r / total_ratio for r in ratios]
    if not records:
        return {name: [] for name in names}
    groups = _grouping(records, group_by_duplicates=group_by_duplicates)
    buckets = _assign_groups(groups, len(records), normalised, names,
                             random.Random(seed))
    return _materialise(records, buckets)


def _carve(pool: Sequence[Record], seed: int,
           validation_fraction: float,
           group_by_duplicates: bool = True) -> Tuple[List[Record], List[Record]]:
    """Split a training pool into train/validation with the same group rules."""
    if not pool:
        return [], []
    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction == 0.0:
        return list(pool), []
    split = random_split(pool, (1.0 - fraction, fraction, 0.0), seed=seed,
                         group_by_duplicates=group_by_duplicates)
    return split["train"], split["validation"]


def _finalise(train: Sequence[Record], validation: Sequence[Record],
              test: Sequence[Record]) -> Dict[str, List[Record]]:
    for name, group in (("train", train), ("validation", validation),
                        ("test", test)):
        for record in group:
            record.split = name
    return {"train": list(train), "validation": list(validation),
            "test": list(test)}


def _normalised_set(values: Optional[Iterable[str]]) -> Set[str]:
    return {str(v).strip().lower() for v in (values or []) if str(v).strip()}


def generator_held_out(records: Sequence[Record],
                       test_generators: List[str],
                       seed: int = 0,
                       negative_share: float = 0.3,
                       validation_fraction: float = 0.15) -> Dict[str, List[Record]]:
    """Train on some generator models, test on models never seen in training.

    This is the split that tells you whether the detector learned "machine
    text" or learned "GPT-4o's habits".  Records with no ``generator_model``
    (human text, and any AI text whose provenance was not recorded) cannot be
    held out by model, so a ``negative_share`` of them is routed to the test
    side - without negatives the test split has nothing to discriminate
    against and its metrics are meaningless.
    """
    wanted = _normalised_set(test_generators)
    if not wanted:
        raise ValueError("test_generators must not be empty")

    test: List[Record] = []
    train_pool: List[Record] = []
    unattributed: List[Record] = []
    for record in records:
        model = (record.generator_model or "").strip().lower()
        if not model:
            unattributed.append(record)
        elif model in wanted:
            test.append(record)
        else:
            train_pool.append(record)

    share = max(0.0, min(float(negative_share), 0.9))
    if unattributed:
        carved = random_split(unattributed, (1.0 - share, 0.0, share), seed=seed)
        train_pool.extend(carved["train"])
        test.extend(carved["test"])

    train, validation = _carve(train_pool, seed + 1, validation_fraction)
    return _finalise(train, validation, test)


def domain_held_out(records: Sequence[Record],
                    test_categories: List[str],
                    seed: int = 0,
                    validation_fraction: float = 0.15) -> Dict[str, List[Record]]:
    """Train on some categories, test on unseen ones.

    Categories apply to human and machine text alike, so unlike the generator
    split this one needs no artificial routing of negatives - provided the
    corpus actually carries both labels inside the held-out categories, which
    ``summarise_split`` will tell you.
    """
    wanted = _normalised_set(test_categories)
    if not wanted:
        raise ValueError("test_categories must not be empty")
    test = [r for r in records if (r.category or "").strip().lower() in wanted]
    train_pool = [r for r in records
                  if (r.category or "").strip().lower() not in wanted]
    train, validation = _carve(train_pool, seed + 1, validation_fraction)
    return _finalise(train, validation, test)


def humanizer_held_out(records: Sequence[Record],
                       test_humanizers: List[str],
                       seed: int = 0,
                       negative_share: float = 0.3,
                       validation_fraction: float = 0.15) -> Dict[str, List[Record]]:
    """Train on some humanizers, test on a paraphraser never seen in training.

    The headline robustness claim for this project lives or dies here.  Because
    the bundled humanizers are simulations, a strong result on this split is
    evidence of transfer *between simulated attacks* only; the same split run
    with an externally collected humanizer in ``test_humanizers`` is the real
    measurement.
    """
    wanted = _normalised_set(test_humanizers)
    if not wanted:
        raise ValueError("test_humanizers must not be empty")

    test: List[Record] = []
    train_pool: List[Record] = []
    others: List[Record] = []
    for record in records:
        humanizer = (record.humanizer or "").strip().lower()
        if not humanizer:
            others.append(record)
        elif humanizer in wanted:
            test.append(record)
        else:
            train_pool.append(record)

    share = max(0.0, min(float(negative_share), 0.9))
    if others:
        carved = random_split(others, (1.0 - share, 0.0, share), seed=seed)
        train_pool.extend(carved["train"])
        test.extend(carved["test"])

    train, validation = _carve(train_pool, seed + 1, validation_fraction)
    return _finalise(train, validation, test)


def length_stratified(records: Sequence[Record]) -> Dict[str, List[Record]]:
    """Per-length-bucket views, for evaluation rather than for training.

    Unlike the other functions here the keys are length buckets, not
    train/validation/test: this is a lens onto an existing split, because
    detector reliability collapses on short text and a single pooled number
    hides that completely.
    """
    buckets: Dict[str, List[Record]] = {"short": [], "medium": [], "long": []}
    for record in records:
        bucket = record.length_bucket or length_bucket(len(record.text.split()))
        buckets.setdefault(bucket, []).append(record)
    return buckets


# --------------------------------------------------------------------------
# leakage diagnostics
# --------------------------------------------------------------------------


def _rank_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Mann-Whitney rank AUC with tie correction, in pure Python.

    Used to ask "could a one-feature model separate the classes on this alone",
    so ties must be handled properly - a corpus where every document is padded
    to the same length would otherwise look perfectly separable.
    """
    if not positive or not negative:
        return float("nan")
    combined = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks = [0.0] * len(combined)
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, combined) if label == 1)
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _overlap(left: Set[str], right: Set[str]) -> Dict[str, Any]:
    if not left or not right:
        return {"jaccard": 0.0, "coverage": 0.0, "shared": 0,
                "left_only": len(left - right), "right_only": len(right - left)}
    shared = left & right
    return {
        "jaccard": len(shared) / float(len(left | right)),
        # coverage answers "does the smaller pool live inside the larger one",
        # which is the question that matters when the pools differ in size
        "coverage": len(shared) / float(min(len(left), len(right))),
        "shared": len(shared),
        "left_only": len(left - right),
        "right_only": len(right - left),
    }


def check_topic_leakage(records: Sequence[Record]) -> Dict[str, Any]:
    """Detect the topic/authorship confound and say so in plain language.

    The failure mode: the human pool and the AI pool are drawn from different
    subject matter, different sources, or different lengths.  A classifier
    trained on that data reaches an excellent validation score by recognising
    the *pool*, not the author, and then collapses the moment it meets human
    text about an AI-pool topic.  Nothing downstream of this function can
    detect that - by the time the model is fitted, the shortcut is already
    learned - so it is checked here, before a dataset is ever written.

    Returns a report dict with ``verdict`` (a sentence a person can read),
    ``severity`` in ``ok``/``warn``/``fail``, and a ``warnings`` list.
    """
    warnings: List[str] = []
    report: Dict[str, Any] = {
        "n_samples": len(records),
        "per_label": {},
        "warnings": warnings,
    }
    if not records:
        report.update(severity="fail", passed=False,
                      verdict="empty dataset: nothing to check.")
        warnings.append("dataset is empty")
        return report

    by_label: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    def _topics(group: Sequence[Record]) -> Set[str]:
        return {str(r.topic).strip().lower() for r in group if r.topic}

    def _sources(group: Sequence[Record]) -> Set[str]:
        return {str(r.source).strip().lower() for r in group
                if r.source and str(r.source).strip().lower() != "unknown"}

    def _categories(group: Sequence[Record]) -> Set[str]:
        return {str(r.category).strip().lower() for r in group if r.category}

    def _lengths(group: Sequence[Record]) -> List[float]:
        return [float(len(r.text.split())) for r in group]

    for label in CLASSES:
        group = by_label.get(label, [])
        lengths = _lengths(group)
        report["per_label"][label] = {
            "n": len(group),
            "topics": sorted(_topics(group))[:50],
            "n_topics": len(_topics(group)),
            "sources": sorted(_sources(group))[:50],
            "n_sources": len(_sources(group)),
            "categories": sorted(_categories(group)),
            "mean_words": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            "topic_coverage": (round(sum(1 for r in group if r.topic)
                                     / float(len(group)), 3) if group else 0.0),
        }

    human = by_label.get("human", [])
    machine = [r for r in records if r.label != "human"]

    if not human or not machine:
        missing = "human" if not human else "machine (pure_ai/humanized_ai)"
        warnings.append(
            f"dataset contains no {missing} samples; a detector cannot be "
            "trained from this and no leakage check is meaningful")
        report.update(severity="fail", passed=False,
                      verdict=f"unusable dataset: no {missing} samples.")
        return report

    human_topics, machine_topics = _topics(human), _topics(machine)
    human_sources, machine_sources = _sources(human), _sources(machine)
    human_categories, machine_categories = _categories(human), _categories(machine)

    report["topic_overlap"] = _overlap(human_topics, machine_topics)
    report["source_overlap"] = _overlap(human_sources, machine_sources)
    report["category_overlap"] = _overlap(human_categories, machine_categories)

    human_lengths, machine_lengths = _lengths(human), _lengths(machine)
    human_mean = sum(human_lengths) / len(human_lengths)
    machine_mean = sum(machine_lengths) / len(machine_lengths)
    length_auc = _rank_auc(machine_lengths, human_lengths)
    report["length"] = {
        "human_mean_words": round(human_mean, 1),
        "machine_mean_words": round(machine_mean, 1),
        "ratio": round(machine_mean / human_mean, 3) if human_mean else 0.0,
        # 0.5 means length carries no information about the label
        "separability_auc": round(length_auc, 3),
    }

    severity = "ok"

    def _escalate(level: str) -> None:
        nonlocal severity
        order = {"ok": 0, "warn": 1, "fail": 2}
        if order[level] > order[severity]:
            severity = level

    topic_coverage_human = sum(1 for r in human if r.topic) / float(len(human))
    topic_coverage_machine = sum(1 for r in machine if r.topic) / float(len(machine))
    if topic_coverage_human < 0.5 or topic_coverage_machine < 0.5:
        warnings.append(
            "topic metadata is missing on a majority of one or both pools "
            f"(human {topic_coverage_human:.0%}, machine "
            f"{topic_coverage_machine:.0%}); "
            "topic leakage CANNOT be ruled out - label your sources with topics")
        _escalate("warn")
    elif report["topic_overlap"]["coverage"] < 0.25:
        warnings.append(
            "human and machine samples come from almost disjoint topic pools "
            f"(shared topics: {report['topic_overlap']['shared']}, "
            f"coverage {report['topic_overlap']['coverage']:.2f}). A classifier "
            "trained on this learns the SUBJECT, not the author")
        _escalate("fail")
    elif report["topic_overlap"]["coverage"] < 0.6:
        warnings.append(
            "topic pools overlap only partially "
            f"(coverage {report['topic_overlap']['coverage']:.2f}); expect the "
            "detector to lean on subject matter for the non-shared part")
        _escalate("warn")

    if human_categories and machine_categories:
        if report["category_overlap"]["coverage"] < 0.3:
            warnings.append(
                "human and machine samples barely share categories "
                f"(coverage {report['category_overlap']['coverage']:.2f}); "
                "the category is close to a label in disguise")
            _escalate("fail")

    if human_sources and machine_sources and report["source_overlap"]["shared"] == 0:
        warnings.append(
            "no source overlap between human and machine pools: every source "
            "is perfectly predictive of the label. This is usual when AI text "
            "is generated rather than collected, but it means any "
            "source-correlated artefact (formatting, boilerplate, encoding) "
            "is a free shortcut - normalise the pools before training")
        _escalate("warn")

    if length_auc == length_auc:      # not NaN
        if length_auc >= 0.85 or length_auc <= 0.15:
            warnings.append(
                f"document length alone separates the classes at AUC "
                f"{length_auc:.2f} (human mean {human_mean:.0f} words, machine "
                f"mean {machine_mean:.0f}); truncate or resample so the length "
                "distributions match")
            _escalate("fail")
        elif length_auc >= 0.7 or length_auc <= 0.3:
            warnings.append(
                f"length is moderately predictive of the label (AUC "
                f"{length_auc:.2f}); check that length features are not doing "
                "the work")
            _escalate("warn")

    counts = {label: len(by_label.get(label, [])) for label in CLASSES}
    if counts.get("humanized_ai", 0) == 0:
        warnings.append(
            "no humanized_ai samples: the three-class model cannot be trained "
            "and the detector will be blind to paraphrased machine text")
        _escalate("warn")
    largest, smallest = max(counts.values()), min(v for v in counts.values() if v)
    if largest > 10 * smallest:
        warnings.append(
            f"severe class imbalance {counts}; per-class recall, not accuracy, "
            "is the number to read")
        _escalate("warn")

    report["label_counts"] = counts
    report["severity"] = severity
    report["passed"] = severity != "fail"
    if severity == "fail":
        report["verdict"] = (
            "LEAKY: this dataset lets a classifier separate the labels without "
            "looking at authorship at all. Fix the pools before training - the "
            "resulting model would score well here and fail on real text.")
    elif severity == "warn":
        report["verdict"] = (
            "SUSPECT: the pools are not obviously confounded, but there are "
            "shortcuts a model could take. Read the warnings and treat "
            "held-out numbers with suspicion.")
    else:
        report["verdict"] = (
            "OK: human and machine pools share topics, categories and length "
            "distribution, so the label is not trivially predictable from "
            "anything other than the writing itself.")
    return report


def summarise_split(splits: Dict[str, List[Record]]) -> Dict[str, Any]:
    """Describe a split and flag anything that straddles it.

    Exact-text and topic straddle are reported as counts rather than as a
    boolean because a single leaked topic in a 20k-record corpus is a nuisance
    while a hundred is a rewrite of the split.
    """
    report: Dict[str, Any] = {"splits": {}, "warnings": []}
    warnings: List[str] = report["warnings"]

    for name, records in splits.items():
        report["splits"][name] = dataset_summary(records)

    names = [n for n in ("train", "validation", "test") if n in splits]
    texts = {n: {re.sub(r"\s+", " ", r.text.strip().lower()) for r in splits[n]}
             for n in names}
    topics = {n: {str(r.topic).strip().lower() for r in splits[n] if r.topic}
              for n in names}

    overlaps: Dict[str, Any] = {}
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            key = f"{left}|{right}"
            shared_text = len(texts[left] & texts[right])
            shared_topic = len(topics[left] & topics[right])
            overlaps[key] = {"exact_text": shared_text,
                             "topics": shared_topic}
            if shared_text:
                warnings.append(
                    f"{shared_text} identical documents appear in both {left} "
                    f"and {right}: the score on {right} is inflated")
            if shared_topic:
                warnings.append(
                    f"{shared_topic} topics straddle {left} and {right}. For a "
                    "held-out generator/domain/humanizer split that is "
                    "expected and harmless; for a random split it means the "
                    "grouping failed and the score on "
                    f"{right} is optimistic")
    report["overlaps"] = overlaps

    for name in names:
        labels = {r.label for r in splits[name]}
        if splits[name] and len(labels) < 2:
            warnings.append(
                f"split '{name}' contains only {sorted(labels)}; no metric "
                "computed on it will mean anything")
        if not splits[name]:
            warnings.append(f"split '{name}' is empty")

    if "test" in splits and splits["test"]:
        test_generators = {r.generator_model for r in splits["test"]
                           if r.generator_model}
        train_generators = {r.generator_model for r in splits.get("train", [])
                            if r.generator_model}
        report["generator_overlap_train_test"] = sorted(
            test_generators & train_generators)
        test_humanizers = {r.humanizer for r in splits["test"] if r.humanizer}
        train_humanizers = {r.humanizer for r in splits.get("train", [])
                            if r.humanizer}
        report["humanizer_overlap_train_test"] = sorted(
            test_humanizers & train_humanizers)

    return report
