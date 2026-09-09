from pathlib import Path
import math
import joblib

from feature_extractor import FEATURE_NAMES

MODEL_PATH = Path(__file__).parent / "models" / "slopaction_model.pkl"

_bundle = None


def _load_bundle():
    global _bundle

    if _bundle is None and MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)

    return _bundle


def model_is_ready():
    return MODEL_PATH.exists()


def _heuristic_probability(features):
    """
    Temporary fallback before ML training.
    This is NOT the final detector.

    It only keeps the API usable while we collect/train the dataset.
    """
    uniformity = 1 - min(features["sentence_burstiness"] / 0.85, 1)
    repetition = min(
        features["repeated_bigram_ratio"] * 4
        + features["repeated_trigram_ratio"] * 6
        + features["repeated_word_ratio"] * 0.6,
        1
    )

    formulaic = min(
        features["ai_phrase_ratio"] * 0.16
        + features["transition_ratio"] * 7
        + features["vague_word_ratio"] * 5,
        1
    )

    low_personal = 1 - min(features["first_person_ratio"] * 18, 1)

    score = (
        0.34 * formulaic
        + 0.28 * uniformity
        + 0.20 * repetition
        + 0.18 * low_personal
    )

    return max(0.05, min(0.95, score))


def predict_ai_probability(features):
    bundle = _load_bundle()

    if bundle is None:
        probability = _heuristic_probability(features)
        return probability, "Low", "heuristic-fallback"

    model = bundle["model"]
    feature_names = bundle.get("feature_names", FEATURE_NAMES)

    vector = [[features[name] for name in feature_names]]

    probabilities = model.predict_proba(vector)[0]

    classes = list(model.classes_)

    if 1 in classes:
        ai_index = classes.index(1)
    elif "AI" in classes:
        ai_index = classes.index("AI")
    else:
        ai_index = int(probabilities.argmax())

    probability = float(probabilities[ai_index])

    distance_from_middle = abs(probability - 0.5)

    if distance_from_middle >= 0.35:
        confidence = "High"
    elif distance_from_middle >= 0.18:
        confidence = "Medium"
    else:
        confidence = "Low"

    return probability, confidence, "ml"
