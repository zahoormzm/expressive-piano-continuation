import csv

import mido

from music_project.data import validate as validator


def test_compositions_crossing_splits_are_rejected(midi_builder, tmp_path, monkeypatch):
    paths = []
    for index, velocity in enumerate((70, 80)):
        paths.append(
            midi_builder(
                [
                    (0, mido.Message("note_on", note=60, velocity=velocity)),
                    (100, mido.Message("note_off", note=60)),
                ],
                name=f"piece-{index}.mid",
            )
        )
    fields = [
        "canonical_composer",
        "canonical_title",
        "split",
        "year",
        "midi_filename",
        "audio_filename",
        "duration",
    ]
    with (tmp_path / "maestro-v3.0.0.csv").open("w", newline="", encoding="utf-8") as source:
        writer = csv.DictWriter(source, fieldnames=fields)
        writer.writeheader()
        for path, split in zip(paths, ("train", "validation")):
            writer.writerow(
                {
                    "canonical_composer": "Composer",
                    "canonical_title": "Same Piece",
                    "split": split,
                    "year": "2004",
                    "midi_filename": path.name,
                    "audio_filename": path.with_suffix(".wav").name,
                    "duration": "0.1",
                }
            )
    monkeypatch.setattr(validator, "EXPECTED_SPLITS", {"train": 1, "validation": 1})
    report = validator.validate(tmp_path, tmp_path / "manifest.jsonl", tmp_path / "report.json")
    assert not report["integrity_passed"]
    assert report["composition_leakage"][0]["splits"] == ["train", "validation"]
    assert not report["cross_split_content_duplicates"]
