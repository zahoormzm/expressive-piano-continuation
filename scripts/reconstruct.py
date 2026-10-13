"""Reconstruct one solo-piano MIDI using the pinned published tokenizer."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.io import write_json
from music_project.tokens.aria import AriaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction/custom"))
    args = parser.parse_args()
    tokenizer = AriaTokenizer()
    report = tokenizer.reconstruct(args.midi, args.output)
    write_json(args.output / "tokenizer.json", tokenizer.identity, immutable=True)
    print(report)


if __name__ == "__main__":
    main()
