"""Piano excerpts for listening, retaining raw pedal controls when requested."""

import math
from pathlib import Path

import mido

from music_project.data.midi import read_performance
from music_project.io import sha256_file


def piano_excerpt(source: Path, destination: Path, seconds=20, retain_pedals=True):
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("Excerpt duration must be in (0, 60] seconds")
    if destination.exists():
        raise FileExistsError(destination)
    performance = read_performance(source)
    if any(note.program != 0 or note.channel == 9 for note in performance.notes):
        raise ValueError("Listening excerpts require acoustic-grand piano MIDI")
    boundary = round(seconds * 1000)
    events = []
    selected = [note for note in performance.notes if note.start_seconds < seconds]
    for note in selected:
        start, end = round(note.start_seconds * 1000), min(boundary, round(note.end_seconds * 1000))
        if end <= start:
            raise ValueError("Millisecond excerpt timing would remove a note")
        events.extend(
            [
                (
                    start,
                    2,
                    mido.Message("note_on", note=note.pitch, velocity=note.velocity, channel=0),
                ),
                (end, 0, mido.Message("note_off", note=note.pitch, velocity=0, channel=0)),
            ]
        )
    pedals = {64: 0, 66: 0, 67: 0}
    if retain_pedals:
        original = mido.MidiFile(source)
        elapsed, tempo = 0.0, 500_000
        for message in mido.merge_tracks(original.tracks):
            elapsed += mido.tick2second(message.time, original.ticks_per_beat, tempo)
            if message.type == "set_tempo":
                tempo = message.tempo
            elif (
                message.type == "control_change" and message.control in pedals and elapsed < seconds
            ):
                events.append((round(elapsed * 1000), 1, message.copy(channel=0, time=0)))
                pedals[message.control] += 1
    for control in pedals:
        events.append(
            (boundary, 1, mido.Message("control_change", channel=0, control=control, value=0))
        )
    output = mido.MidiFile(ticks_per_beat=500)
    track = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.Message("program_change", channel=0, program=0, time=0),
        ]
    )
    output.tracks.append(track)
    previous = 0
    for tick, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)
    actual = read_performance(destination)
    if len(actual.notes) != len(selected) or actual.issues.get("unmatched_note_on"):
        raise ValueError("Excerpt export did not preserve complete note count")
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "midi": str(destination),
        "midi_sha256": sha256_file(destination),
        "seconds": seconds,
        "notes": len(selected),
        "retained_pedal_events": pedals,
        "notes_clipped_at_boundary": sum(note.end_seconds > seconds for note in selected),
        "timing_precision_ms": 1,
        "controls_released_at_boundary": [64, 66, 67],
    }
