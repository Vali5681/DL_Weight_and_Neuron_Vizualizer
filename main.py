"""Entry point for the Neural Network Weight & Training Visualizer."""

import argparse
import threading
import webbrowser

from dl_viz.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Network Weight & Training Visualizer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    args = parser.parse_args()

    server = run_server(host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"DL Weight Visualizer running at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
