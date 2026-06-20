from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from ultrasound_nlp.nlp_pipeline import (
    ORGAN_CONFIG,
    analyze_report,
    cluster_for_report,
    dataset_stats,
    dictionary_view,
    model_metrics,
    nearest_vector_terms,
    ner_metrics,
    predict_label,
    sample_report,
    similar_reports,
    vector_model_metrics,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/organs")
def organs():
    return jsonify({"organs": [{"key": key, "name": value["name"]} for key, value in ORGAN_CONFIG.items()]})


@app.get("/api/report/sample")
def report_sample():
    organ = request.args.get("organ", "thyroid")
    index = request.args.get("index", "0")
    try:
        sample_index = int(index)
    except ValueError:
        sample_index = 0
    return jsonify(sample_report(organ, sample_index))


@app.get("/api/stats")
def stats():
    return jsonify(dataset_stats())


@app.get("/api/dictionary")
def dictionary():
    return jsonify(dictionary_view())


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", ""))
    organ = str(payload.get("organ", "thyroid"))
    return jsonify(analyze_report(text, organ))


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", ""))
    organ = str(payload.get("organ", "thyroid"))
    top_k = int(payload.get("top_k", 3) or 3)
    return jsonify(predict_label(text, organ, top_k))


@app.get("/api/model/metrics")
def metrics():
    return jsonify(model_metrics())


@app.get("/api/ner/metrics")
def sequence_ner_metrics():
    return jsonify(ner_metrics())


@app.get("/api/vector/metrics")
def vectors_metrics():
    return jsonify(vector_model_metrics())


@app.post("/api/cluster")
def cluster():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", ""))
    organ = str(payload.get("organ", "thyroid"))
    top_k = int(payload.get("top_k", 5) or 5)
    return jsonify(cluster_for_report(text, organ, top_k))


@app.post("/api/vector/nearest")
def vector_nearest():
    payload = request.get_json(silent=True) or {}
    term = str(payload.get("term", "结节"))
    top_k = int(payload.get("top_k", 10) or 10)
    return jsonify(nearest_vector_terms(term, top_k))


@app.post("/api/similar")
def similar():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", ""))
    organ = str(payload.get("organ", "thyroid"))
    try:
        limit = int(payload.get("limit", 5) or 5)
    except ValueError:
        limit = 5
    return jsonify({"items": similar_reports(text, organ, limit)})


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "not found"}), 404


def run(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
