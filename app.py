from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nlp_pipeline import (
    ORGAN_CONFIG,
    analyze_report,
    dataset_stats,
    dictionary_view,
    sample_report,
    similar_reports,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class UltrasoundNLPHandler(BaseHTTPRequestHandler):
    server_version = "UltrasoundNLP/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            safe_name = parsed.path.removeprefix("/static/").replace("/", "\\")
            path = (STATIC_DIR / safe_name).resolve()
            if not str(path).startswith(str(STATIC_DIR.resolve())):
                self._send_json({"error": "invalid static path"}, HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(path)
            return
        if parsed.path == "/api/report/sample":
            query = parse_qs(parsed.query)
            organ = query.get("organ", ["thyroid"])[0]
            index = int(query.get("index", ["0"])[0] or "0")
            self._send_json(sample_report(organ, index))
            return
        if parsed.path == "/api/stats":
            self._send_json(dataset_stats())
            return
        if parsed.path == "/api/dictionary":
            self._send_json(dictionary_view())
            return
        if parsed.path == "/api/organs":
            self._send_json({"organs": [{"key": key, "name": value["name"]} for key, value in ORGAN_CONFIG.items()]})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/analyze":
            text = str(payload.get("text", ""))
            organ = str(payload.get("organ", "thyroid"))
            self._send_json(analyze_report(text, organ))
            return
        if parsed.path == "/api/similar":
            text = str(payload.get("text", ""))
            organ = str(payload.get("organ", "thyroid"))
            limit = int(payload.get("limit", 5) or 5)
            self._send_json({"items": similar_reports(text, organ, limit)})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid json: {exc}") from exc

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
            return
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if mime_type.startswith("text/") else mime_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), UltrasoundNLPHandler)
    print(f"Ultrasound NLP app running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run()
