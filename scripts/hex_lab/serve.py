"""Serve a prepared MIND hex cache on localhost."""

import argparse
import gzip
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from scripts.hex_lab.analysis import Explorer

WEB = Path(__file__).parent / "web"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, explorer: Explorer, **kwargs: Any) -> None:
        self.explorer = explorer
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            if url.path not in ("/", "/index.html", "/app.js", "/style.css"):
                self.send_error(404)
                return
            super().do_GET()
            return
        try:
            args = {key: value[0] for key, value in parse_qs(url.query).items()}
            if url.path == "/api/meta":
                result = self.explorer.meta
            elif url.path == "/api/select":
                result = self.explorer.select(
                    float(args["lat"]), float(args["lon"]), int(args.get("minimum_km", 3000))
                )
            elif url.path == "/api/hexes":
                bbox = [float(v) for v in args.get("bbox", "-180,-90,180,90").split(",")]
                result = self.explorer.hexes(
                    int(args.get("level", 3)), bbox, args.get("mode", "pca"), args.get("query")
                )
            else:
                self.send_error(404)
                return
            self.send_json(result)
        except (ValueError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, value, status=200) -> None:
        payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        compressed = "gzip" in self.headers.get("Accept-Encoding", "") and len(payload) > 1000
        if compressed:
            payload = gzip.compress(payload, compresslevel=3)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8776)
    args = parser.parse_args()
    explorer = Explorer(args.data_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, explorer=explorer))
    print(f"{explorer.meta['label']} → http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
