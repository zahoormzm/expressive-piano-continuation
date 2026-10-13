from pathlib import Path

import mido
import pytest


@pytest.fixture
def midi_builder(tmp_path):
    """Build MIDI from explicitly ordered absolute tick events."""

    def build(events: list[tuple[int, mido.Message]], name="example.mid") -> Path:
        midi = mido.MidiFile(ticks_per_beat=500)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        last = 0
        for tick, event in sorted(events, key=lambda item: item[0]):
            track.append(event.copy(time=tick - last))
            last = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        path = tmp_path / name
        midi.save(path)
        return path

    return build
