"""Render bounded grand-piano MIDI with the pinned SoundFont and fixed settings."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.audio.piano import render_piano


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    args = parser.parse_args()
    print(render_piano(args.midi, args.wav))


if __name__ == "__main__":
    main()
