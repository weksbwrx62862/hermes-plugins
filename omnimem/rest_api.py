from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs


class OmniMemAPIHandler(BaseHTTPRequestHandler):
    _sdk = None

    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

        routes = {
            "/api/memorize": self._handle_memorize,
            "/api/recall": self._handle_recall,
            "/api/reflect": self._handle_reflect,
            "/api/govern": self._handle_govern,
            "/api/compact": self._handle_compact,
            "/api/detail": self._handle_detail,
            "/api/export": self._handle_export,
            "/api/import": self._handle_import,
            "/api/health": self._handle_health,
        }

        handler = routes.get(path)
        if handler:
            try:
                result = handler(body)
                self._send_json(200, result)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": f"Not found: {path}"})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json(200, self._sdk.health_check() if self._sdk else {"status": "not initialized"})
        elif path == "/api/tools":
            self._send_json(200, {"tools": ["memorize", "recall", "reflect", "govern", "compact", "detail", "export", "import", "health"]})
        else:
            self._send_json(404, {"error": f"Not found: {path}"})

    def _handle_memorize(self, body):
        return self._sdk.memorize(**body)

    def _handle_recall(self, body):
        return self._sdk.recall(**body)

    def _handle_reflect(self, body):
        return self._sdk.reflect(**body)

    def _handle_govern(self, body):
        return self._sdk.govern(**body)

    def _handle_compact(self, body):
        return self._sdk.compact(**body)

    def _handle_detail(self, body):
        return self._sdk.detail(**body)

    def _handle_export(self, body):
        return self._sdk.export_memories(**body)

    def _handle_import(self, body):
        return self._sdk.import_memories(**body)

    def _handle_health(self, body):
        return self._sdk.health_check()

    def _send_json(self, code: int, data: Any):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        pass


def run_api(host: str = "0.0.0.0", port: int = 8765, storage_dir: str | None = None, config: dict | None = None):
    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK(storage_dir=storage_dir, config=config)
    OmniMemAPIHandler._sdk = sdk
    server = HTTPServer((host, port), OmniMemAPIHandler)
    print(f"OmniMem REST API running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sdk.close()
        server.server_close()


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_api(port=port)
