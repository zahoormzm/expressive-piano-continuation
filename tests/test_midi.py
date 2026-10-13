import mido
import pytest

from music_project.data.midi import read_performance


def test_tempo_changes_preserve_elapsed_time(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (500, mido.MetaMessage("set_tempo", tempo=1_000_000)),
            (1000, mido.Message("note_off", note=60)),
        ]
    )
    performance = read_performance(path)
    assert performance.notes[0].end_seconds == pytest.approx(1.5)
    assert performance.duration_seconds == pytest.approx(1.5)


def test_unmatched_events_are_logged_without_repair(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("note_off", note=61)),
            (0, mido.Message("note_on", note=60, velocity=80)),
            (100, mido.Message("note_off", note=60)),
            (200, mido.Message("note_on", note=62, velocity=90)),
        ]
    )
    performance = read_performance(path)
    assert performance.issues == {"unmatched_note_off": 1, "unmatched_note_on": 1}
    assert len(performance.notes) == 1


def test_velocity_zero_is_note_off(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (100, mido.Message("note_on", note=60, velocity=0)),
        ]
    )
    performance = read_performance(path)
    assert not performance.issues
    assert performance.notes[0].end_tick == 100
