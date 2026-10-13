import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.audio.preview import render_preview
from music_project.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Render harmonic diagnostic audio from MIDI")
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    report = render_preview(args.midi, args.wav, args.seconds)
    write_json(args.wav.with_suffix(".json"), report)
    print(report)


if __name__ == "__main__":
    main()
