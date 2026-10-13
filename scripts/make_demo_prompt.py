"""Create a project-authored C-major arpeggio prompt for engineering checks."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.generation.midi import flat_midi, save_checked
from music_project.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/prompts/arpeggio.mid"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notes = []
    chords = [
        (48, [60, 64, 67, 72]),
        (53, [60, 65, 69, 72]),
        (45, [60, 64, 69, 72]),
        (43, [59, 62, 67, 71]),
    ]
    for bar, (bass, pitches) in enumerate(chords):
        offset = bar * 2000
        notes.append((bass, 62, offset, offset + 1800))
        for step in range(8):
            onset = offset + step * 250
            notes.append((pitches[step % 4], 70 + 4 * (step % 3), onset, onset + 210))
    _, report = save_checked(flat_midi(notes), args.output)
    report["description"] = (
        "Project-authored eight-second arpeggio; engineering prompt, not a benchmark piece"
    )
    write_json(args.output.with_suffix(".json"), report)
    print(report)


if __name__ == "__main__":
    main()
