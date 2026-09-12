"""
app.py
------
Step 5: DEPLOYMENT (local reference implementation)

Exposes:
    GET  /                -> serves the 3D UI (static/index.html)
    POST /api/predict     -> {"text": "..."} -> {"label": ..., "scores": {...}}
    GET  /api/metrics      -> training-time intrinsic metrics (for the UI's
                              "model health" panel)
    GET  /api/health       -> simple liveness probe for monitoring/K8s

In production this same Flask app would be containerised (Dockerfile),
put behind Gunicorn + Nginx, and horizontally scaled behind a load
balancer / API gateway. See README.md for the full deployment &
monitoring write-up.
"""

import os
import sys
import json
import time
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_sentiment  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    start = time.time()
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 chars)."}), 400

    try:
        result = predict_sentiment(text)
    except FileNotFoundError:
        return jsonify({
            "error": "Model not trained yet. Run `python src/train_model.py` first."
        }), 503
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500

    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info("predict text_len=%d label=%s latency_ms=%s",
                len(text), result["label"], latency_ms)

    result["latency_ms"] = latency_ms
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "No metrics yet, train the model first."}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


@app.route("/api/health")
def api_health():
    model_ready = os.path.exists(os.path.join(MODEL_DIR, "sentiment_clf.joblib"))
    return jsonify({"status": "ok", "model_ready": model_ready})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
