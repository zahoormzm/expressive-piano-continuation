"""Run the local MIDI upload and piano continuation application."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.web.server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    root = Path(__file__).resolve().parents[1]
    server = make_server(root, args.port)
    print(f"Piano continuation studio: http://127.0.0.1:{args.port}/", flush=True)
    print(
        "Uploads, generation, and audio remain on this computer. Ctrl+C stops the server.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.manager.close()
        server.server_close()


if __name__ == "__main__":
    main()
