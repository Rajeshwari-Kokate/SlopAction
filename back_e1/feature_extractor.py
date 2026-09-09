import math
import re
from collections import Counter
from statistics import mean, pstdev


AI_STYLE_PHRASES = [
    "in today's rapidly evolving",
    "in today's world",
    "rapidly evolving landscape",
    "it is important to note",
    "it is worth noting",
    "in conclusion",
    "moreover",
    "furthermore",
    "delve into",
    "transformative",
    "plays a crucial role",
    "in the realm of",
    "a testament to",
    "navigate the complexities",
    "ever-evolving",
    "unlock the potential",
    "seamlessly",
    "robust",
    "leverage",
    "multifaceted",
    "comprehensive approach",
]

TRANSITIONS = {
    "moreover", "furthermore", "therefore", "however", "additionally",
    "consequently", "thus", "hence", "overall", "ultimately",
    "firstly", "secondly", "finally", "notably", "importantly"
}

FIRST_PERSON = {
    "i", "me", "my", "mine", "we", "us", "our", "ours"
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "to",
    "of", "in", "on", "for", "with", "at", "by", "from", "is", "am",
    "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "as", "not", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "should", "may", "might",
    "you", "your", "they", "their", "he", "she", "his", "her", "them"
}

VAGUE_WORDS = {
    "important", "significant", "various", "many", "several", "numerous",
    "things", "aspects", "factors", "elements", "issues", "solutions",
    "benefits", "challenges", "effective", "efficient", "powerful",
    "valuable", "essential", "crucial", "innovative", "dynamic"
}


FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "paragraph_count",
    "avg_sentence_length",
    "sentence_length_std",
    "min_sentence_length",
    "max_sentence_length",
    "unique_word_ratio",
    "hapax_ratio",
    "avg_word_length",
    "word_length_std",
    "stopword_ratio",
    "first_person_ratio",
    "transition_ratio",
    "vague_word_ratio",
    "ai_phrase_ratio",
    "comma_ratio",
    "semicolon_ratio",
    "colon_ratio",
    "question_ratio",
    "exclamation_ratio",
    "digit_ratio",
    "uppercase_ratio",
    "repeated_word_ratio",
    "repeated_bigram_ratio",
    "repeated_trigram_ratio",
    "sentence_starter_repetition",
    "paragraph_length_std",
    "lexical_density",
    "sentence_burstiness"
]


def _safe_div(a, b):
    return a / b if b else 0.0


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _tokenize_words(text):
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def _sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|[\n\r]+", text)
        if s.strip()
    ]


def _paragraphs(text):
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]
    return paragraphs if paragraphs else ([text.strip()] if text.strip() else [])


def _ngram_repetition(tokens, n):
    if len(tokens) < n:
        return 0.0

    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeats = sum(count - 1 for count in counts.values() if count > 1)

    return _safe_div(repeats, len(grams))


def extract_features(text):
    words = _tokenize_words(text)
    sentences = _sentences(text)
    paragraphs = _paragraphs(text)

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)

    sentence_lengths = [
        len(_tokenize_words(sentence))
        for sentence in sentences
        if _tokenize_words(sentence)
    ]

    paragraph_lengths = [
        len(_tokenize_words(paragraph))
        for paragraph in paragraphs
        if _tokenize_words(paragraph)
    ]

    word_lengths = [len(word) for word in words]

    word_counter = Counter(words)
    unique_words = len(word_counter)
    hapax = sum(1 for count in word_counter.values() if count == 1)

    repeated_word_tokens = sum(
        count - 1 for count in word_counter.values() if count > 1
    )

    transition_count = sum(1 for word in words if word in TRANSITIONS)
    first_person_count = sum(1 for word in words if word in FIRST_PERSON)
    stopword_count = sum(1 for word in words if word in STOPWORDS)
    vague_count = sum(1 for word in words if word in VAGUE_WORDS)

    lower = text.lower()
    ai_phrase_hits = sum(lower.count(phrase) for phrase in AI_STYLE_PHRASES)

    punctuation_base = max(len(text), 1)

    alphabetic_chars = [c for c in text if c.isalpha()]
    uppercase_chars = sum(1 for c in alphabetic_chars if c.isupper())

    digit_chars = sum(1 for c in text if c.isdigit())

    starters = []
    for sentence in sentences:
        sentence_words = _tokenize_words(sentence)
        if sentence_words:
            starters.append(sentence_words[0])

    starter_counts = Counter(starters)
    repeated_starters = sum(
        count - 1 for count in starter_counts.values() if count > 1
    )

    content_words = [
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    ]

    avg_sentence_length = mean(sentence_lengths) if sentence_lengths else 0.0
    sentence_std = pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    features = {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "paragraph_count": float(paragraph_count),
        "avg_sentence_length": float(avg_sentence_length),
        "sentence_length_std": float(sentence_std),
        "min_sentence_length": float(min(sentence_lengths) if sentence_lengths else 0),
        "max_sentence_length": float(max(sentence_lengths) if sentence_lengths else 0),
        "unique_word_ratio": _safe_div(unique_words, word_count),
        "hapax_ratio": _safe_div(hapax, word_count),
        "avg_word_length": float(mean(word_lengths) if word_lengths else 0),
        "word_length_std": float(pstdev(word_lengths) if len(word_lengths) > 1 else 0),
        "stopword_ratio": _safe_div(stopword_count, word_count),
        "first_person_ratio": _safe_div(first_person_count, word_count),
        "transition_ratio": _safe_div(transition_count, word_count),
        "vague_word_ratio": _safe_div(vague_count, word_count),
        "ai_phrase_ratio": _safe_div(ai_phrase_hits, max(sentence_count, 1)),
        "comma_ratio": text.count(",") / punctuation_base,
        "semicolon_ratio": text.count(";") / punctuation_base,
        "colon_ratio": text.count(":") / punctuation_base,
        "question_ratio": text.count("?") / punctuation_base,
        "exclamation_ratio": text.count("!") / punctuation_base,
        "digit_ratio": digit_chars / punctuation_base,
        "uppercase_ratio": _safe_div(uppercase_chars, len(alphabetic_chars)),
        "repeated_word_ratio": _safe_div(repeated_word_tokens, word_count),
        "repeated_bigram_ratio": _ngram_repetition(words, 2),
        "repeated_trigram_ratio": _ngram_repetition(words, 3),
        "sentence_starter_repetition": _safe_div(repeated_starters, len(starters)),
        "paragraph_length_std": float(
            pstdev(paragraph_lengths) if len(paragraph_lengths) > 1 else 0
        ),
        "lexical_density": _safe_div(len(content_words), word_count),
        "sentence_burstiness": _safe_div(sentence_std, avg_sentence_length),
    }

    return features


def calculate_slop_metrics(text, features=None):
    if features is None:
        features = extract_features(text)

    lower = text.lower()
    words = _tokenize_words(text)

    # Generic language
    phrase_hits = sum(lower.count(phrase) for phrase in AI_STYLE_PHRASES)
    vague_hits = sum(1 for word in words if word in VAGUE_WORDS)

    generic = (
        18
        + phrase_hits * 8
        + _safe_div(vague_hits, max(len(words), 1)) * 260
        + features["transition_ratio"] * 170
    )

    # Repetition
    repetition = (
        features["repeated_word_ratio"] * 80
        + features["repeated_bigram_ratio"] * 180
        + features["repeated_trigram_ratio"] * 220
        + features["sentence_starter_repetition"] * 35
    )

    # Uniformity:
    # Very low sentence burstiness means sentence lengths are highly similar.
    burstiness = features["sentence_burstiness"]
    uniformity = 100 * (1 - _clamp(burstiness / 0.75))

    # Lack of specificity:
    # Numbers, first-person details and lexical richness reduce this score.
    specificity_evidence = (
        min(features["digit_ratio"] * 1600, 24)
        + min(features["first_person_ratio"] * 350, 18)
        + min(max(features["unique_word_ratio"] - 0.48, 0) * 75, 22)
    )

    lack_specificity = 76 - specificity_evidence + features["vague_word_ratio"] * 180

    generic = round(_clamp(generic / 100) * 100)
    repetition = round(_clamp(repetition / 100) * 100)
    uniformity = round(_clamp(uniformity / 100) * 100)
    lack_specificity = round(_clamp(lack_specificity / 100) * 100)

    slop_score = round(
        generic * 0.32
        + repetition * 0.28
        + uniformity * 0.16
        + lack_specificity * 0.24
    )

    return {
        "generic_language": generic,
        "repetition": repetition,
        "sentence_uniformity": uniformity,
        "lack_of_specificity": lack_specificity,
        "slop_score": slop_score
    }
