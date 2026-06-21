from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from ultrasound_nlp.nlp_pipeline import (
    ORGAN_CONFIG,
    analyze_report,
    dataset_stats,
    dictionary_view,
    sample_report,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/report/sample")
def report_sample():
    organ = request.args.get("organ", "thyroid")
    try:
        index = int(request.args.get("index", "0"))
    except ValueError:
        index = 0
    return jsonify(sample_report(organ, index))


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    return jsonify(
        analyze_report(
            str(payload.get("text", "")),
            str(payload.get("organ", "thyroid")),
        )
    )


@app.get("/api/stats")
def stats():
    return jsonify(dataset_stats())


@app.get("/api/dictionary")
def dictionary():
    return jsonify(dictionary_view())


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "not found"}), 404


def run(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
