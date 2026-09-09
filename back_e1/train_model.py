from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from feature_extractor import FEATURE_NAMES, extract_features


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "slopaction_model.pkl"
METRICS_PATH = BASE_DIR / "training_metrics.json"


def normalize_label(value):
    value = str(value).strip().lower()

    if value in {"1", "ai", "artificial", "generated", "machine"}:
        return 1

    if value in {"0", "human", "person", "real"}:
        return 0

    raise ValueError(f"Unknown label: {value}")


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "dataset.csv was not found. Add your Human/AI samples first."
        )

    df = pd.read_csv(DATASET_PATH)

    required = {"text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"dataset.csv is missing columns: {', '.join(sorted(missing))}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df["label"] = df["label"].apply(normalize_label)

    print(f"Samples: {len(df)}")
    print(df["label"].value_counts().rename(index={0: "HUMAN", 1: "AI"}))

    print("\nExtracting 30 stylometric features...")

    rows = []
    for index, text in enumerate(df["text"], start=1):
        features = extract_features(text)
        rows.append([features[name] for name in FEATURE_NAMES])

        if index % 100 == 0:
            print(f"  processed {index}/{len(df)}")

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df["label"]

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both HUMAN and AI samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["HUMAN", "AI"],
        zero_division=0
    ))

    feature_importance = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nTop 10 features:")
    for name, importance in feature_importance[:10]:
        print(f"{name:30s} {importance:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "model_type": "RandomForestClassifier",
        "version": 1
    }

    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "top_features": [
            {"feature": name, "importance": float(importance)}
            for name, importance in feature_importance[:10]
        ]
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
