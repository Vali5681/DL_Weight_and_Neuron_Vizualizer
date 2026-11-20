"""Local HTTP server exposing the neural network state and static visualization UI."""

import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dl_viz.datasets import DATASETS, get_dataset
from dl_viz.nn import MLP

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class AppState:
    """Thread-safe container for the active dataset, model, and training config."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dataset_name = "xor"
        self.hidden_layers: List[int] = [4, 4]
        self.activation = "tanh"
        self.optimizer = "adam"
        self.lr = 0.05
        self.seed = 42
        self.grid_size = 28
        self.dataset = get_dataset(self.dataset_name, n_samples=160, seed=self.seed)
        self.model = MLP(hidden_layers=self.hidden_layers, activation=self.activation, seed=self.seed)

    def _compute_grid(self) -> List[List[float]]:
        n = self.grid_size
        lo, hi = -1.5, 1.5
        step = (hi - lo) / max(1, n - 1)
        grid: List[List[float]] = []
        for row in range(n):
            y = hi - row * step
            row_vals: List[float] = []
            for col in range(n):
                x = lo + col * step
                row_vals.append(round(self.model.predict([x, y]), 4))
            grid.append(row_vals)
        return grid

    def full_payload(self) -> Dict[str, Any]:
        return {
            "config": {
                "dataset": self.dataset_name,
                "datasets_available": list(DATASETS.keys()),
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "optimizer": self.optimizer,
                "lr": self.lr,
                "seed": self.seed,
            },
            "dataset": [{"x": round(pt[0][0], 4), "y": round(pt[0][1], 4), "label": pt[1]} for pt in self.dataset],
            "model": self.model.snapshot(),
            "grid": self._compute_grid(),
            "grid_bounds": [-1.5, 1.5],
        }

    def step(self, epochs: int = 1) -> Dict[str, Any]:
        with self.lock:
            epochs = max(1, min(epochs, 50))
            for _ in range(epochs):
                self.model.train_epoch(self.dataset, lr=self.lr, optimizer=self.optimizer)
            return self.full_payload()

    def reset(self, new_seed: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            if new_seed is not None:
                self.seed = new_seed
            else:
                self.seed += 1
            self.dataset = get_dataset(self.dataset_name, n_samples=160, seed=self.seed)
            self.model = MLP(
                hidden_layers=self.hidden_layers,
                activation=self.activation,
                seed=self.seed,
            )
            return self.full_payload()

    def update_config(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            rebuild_model = False
            reload_data = False

            if "lr" in body:
                self.lr = max(0.0005, min(float(body["lr"]), 1.0))
            if "optimizer" in body and body["optimizer"] in ("adam", "sgd"):
                self.optimizer = body["optimizer"]

            if "dataset" in body and body["dataset"] in DATASETS and body["dataset"] != self.dataset_name:
                self.dataset_name = body["dataset"]
                reload_data = True
                rebuild_model = True

            if "activation" in body and body["activation"] in ("tanh", "relu", "sigmoid"):
                if body["activation"] != self.activation:
                    self.activation = body["activation"]
                    rebuild_model = True

            if "hidden_layers" in body:
                raw = body["hidden_layers"]
                if isinstance(raw, list) and 1 <= len(raw) <= 4:
                    parsed = [max(1, min(int(v), 10)) for v in raw]
                    if parsed != self.hidden_layers:
                        self.hidden_layers = parsed
                        rebuild_model = True

            if reload_data:
                self.dataset = get_dataset(self.dataset_name, n_samples=160, seed=self.seed)
            if rebuild_model:
                self.model = MLP(
                    hidden_layers=self.hidden_layers,
                    activation=self.activation,
                    seed=self.seed,
                )
            return self.full_payload()


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Keep server output clean unless debugging
            return

        def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path).path
            if parsed == "/" or parsed == "/index.html":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif parsed == "/styles.css":
                self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            elif parsed == "/app.js":
                self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            elif parsed == "/api/state":
                with state.lock:
                    payload = state.full_payload()
                self._send_json(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path).path
            try:
                body = self._read_json_body()
            except (ValueError, UnicodeDecodeError):
                self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed == "/api/step":
                epochs = int(body.get("epochs", 1))
                self._send_json(state.step(epochs=epochs))
            elif parsed == "/api/reset":
                self._send_json(state.reset())
            elif parsed == "/api/config":
                self._send_json(state.update_config(body))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    return Handler


def run_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    state = AppState()
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server
