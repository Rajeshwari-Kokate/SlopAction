from flask import Flask, request, jsonify
from flask_cors import CORS

from feature_extractor import extract_features, calculate_slop_metrics
from predictor import predict_ai_probability, model_is_ready

app = Flask(__name__)
CORS(app)


@app.get("/")
def home():
    return jsonify({
        "project": "SlopAction",
        "status": "backend running",
        "model_ready": model_is_ready()
    })


@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400

    word_count = len(text.split())

    if word_count < 20:
        return jsonify({
            "error": "Please provide at least 20 words for analysis."
        }), 400

    features = extract_features(text)
    slop = calculate_slop_metrics(text, features)

    ai_probability, confidence, mode = predict_ai_probability(features)

    result = {
        "ai_score": round(ai_probability * 100),
        "slop_score": slop["slop_score"],
        "generic_language": slop["generic_language"],
        "repetition": slop["repetition"],
        "sentence_uniformity": slop["sentence_uniformity"],
        "lack_of_specificity": slop["lack_of_specificity"],
        "confidence": confidence,
        "mode": mode,
        "model_ready": model_is_ready(),
        "word_count": word_count
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
