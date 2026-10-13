import mido

from music_project.audio.excerpts import piano_excerpt
from music_project.data.midi import read_performance


def test_excerpt_preserves_raw_sustain_and_clips_note_boundaries(midi_builder, tmp_path):
    source = midi_builder(
        [
            (0, mido.Message("control_change", control=64, value=127)),
            (0, mido.Message("note_on", note=60, velocity=83)),
            (3000, mido.Message("note_off", note=60)),
            (4000, mido.Message("control_change", control=64, value=0)),
        ]
    )
    output = tmp_path / "excerpt.mid"
    report = piano_excerpt(source, output, seconds=1)
    assert report["notes"] == 1 and report["notes_clipped_at_boundary"] == 1
    note = read_performance(output).notes[0]
    assert note.velocity == 83 and note.end_seconds == 1
    controls = [
        msg
        for msg in mido.merge_tracks(mido.MidiFile(output).tracks)
        if msg.type == "control_change"
    ]
    assert controls[0].control == 64 and controls[0].value == 127
    assert any(msg.control == 64 and msg.value == 0 for msg in controls[1:])
    assert report["retained_pedal_events"][64] == 1
