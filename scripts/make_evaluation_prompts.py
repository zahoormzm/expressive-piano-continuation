"""Write two authored prompts to complement the existing demo arpeggio."""

from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.generation.midi import flat_midi, save_checked
from music_project.io import sha256_file, write_json


def main():
    root = Path("outputs/prompts")
    root.mkdir(parents=True, exist_ok=True)
    phrases = {}
    # A slower melody and accompanying dyads; no excerpts from the evaluation corpus.
    melody = []
    pitches = [64, 67, 69, 67, 65, 64, 62, 60, 62, 64, 67, 65]
    starts = [0, 700, 1200, 2000, 2700, 3200, 4000, 4700, 5200, 6000, 6700, 7200]
    for index, (pitch, start) in enumerate(zip(pitches, starts)):
        melody.append((pitch, [70, 76, 80, 73][index % 4], start, min(start + 610, 8000)))
    for start, chord in [(0, (48, 55)), (2000, (53, 57)), (4000, (50, 57)), (6000, (43, 55))]:
        melody.extend((pitch, 55, start, start + 1850) for pitch in chord)
    phrases["melody"] = (melody, "Project-authored lyrical melody with held accompaniment")
    # Staccato syncopation tests a second rhythm and minor harmony.
    pulse = []
    for bar, (bass, chord) in enumerate(
        [(45, (60, 64, 69)), (50, (62, 65, 69)), (41, (60, 65, 69)), (40, (59, 64, 68))]
    ):
        start = bar * 2000
        pulse.extend(
            (bass, velocity, start + offset, start + offset + 240)
            for offset, velocity in [(0, 72), (1000, 64)]
        )
        for beat, offset in enumerate([250, 750, 1500]):
            pulse.extend(
                (pitch, 65 + 5 * beat, start + offset, start + offset + 180) for pitch in chord
            )
    phrases["syncopation"] = (pulse, "Project-authored syncopated A-minor chord phrase")
    for name, (notes, description) in phrases.items():
        path = root / f"{name}.mid"
        if path.exists():
            raise FileExistsError(path)
        _, report = save_checked(flat_midi(notes), path)
        report.update(
            description=description, source_sha256=sha256_file(path), intended_prompt_seconds=8
        )
        write_json(path.with_suffix(".json"), report)
        print(f"Created {path}: {len(notes)} notes", flush=True)


if __name__ == "__main__":
    main()
