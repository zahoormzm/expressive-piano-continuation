"""Build a bounded quantized prompt and export checked full/continuation MIDI files."""

import copy
import math
from pathlib import Path

from ariautils.midi import MidiDict, midi_to_dict

from music_project.data.midi import read_performance
from music_project.io import sha256_file
from music_project.tokens.aria import note_signature


def flat_midi(notes):
    return MidiDict(
        meta_msgs=[],
        tempo_msgs=[{"type": "tempo", "data": 500_000, "tick": 0}],
        pedal_msgs=[],
        instrument_msgs=[{"type": "instrument", "data": 0, "tick": 0, "channel": 0}],
        note_msgs=[
            {
                "type": "note",
                "channel": 0,
                "tick": start,
                "data": {"pitch": pitch, "velocity": velocity, "start": start, "end": end},
            }
            for pitch, velocity, start, end in notes
        ],
        ticks_per_beat=500,
        metadata={},
    )


def build_prompt(path, seconds, tokenizer):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Prompt duration must be finite and positive")
    boundary = math.ceil(seconds * 1000 / 10) * 10
    audit = read_performance(path)
    if audit.issues.get("unmatched_note_on") or audit.issues.get("nonpositive_note_duration"):
        raise ValueError("Prompt contains unresolved or nonpositive notes")
    original = MidiDict.from_midi(path)
    if len(original.note_msgs) != len(audit.notes):
        raise ValueError("Prompt parsers disagree on note count")
    normalized, normalization = tokenizer.normalize(original)
    quantized = tokenizer.expected_notes(normalized)
    clipped = [
        (pitch, velocity, start, min(end, boundary))
        for pitch, velocity, start, end in quantized
        if start < boundary
    ]
    if not clipped:
        raise ValueError("No prompt notes before the requested boundary")
    # MIDI parsers collect notes at note-off, then sort by onset. Reparse the
    # bounded quantized MIDI in memory to retain that simultaneous-note order.
    materialized = flat_midi(clipped).to_midi()
    ids = tokenizer.encode(MidiDict(**midi_to_dict(materialized)))
    if ids[-1] != tokenizer.identity["special_token_ids"]["<E>"]:
        raise ValueError("Prompt tokenizer did not append its real EOS")
    decoded = tokenizer.decode(ids)
    actual = note_signature(decoded)
    if any(start >= boundary or end > boundary for _, _, start, end in actual):
        raise ValueError("Quantized prompt crossed its declared continuation boundary")
    return (
        ids[:-1],
        decoded,
        {
            "source": str(path),
            "source_sha256": sha256_file(path),
            "requested_prompt_seconds": seconds,
            "boundary_ms": boundary,
            "prompt_tokens": len(ids) - 1,
            "prompt_notes": len(actual),
            "normalization": normalization,
            "notes_clipped_at_prompt_boundary": sum(
                end > boundary for _, _, start, end in quantized if start < boundary
            ),
            "preprocessing": "shared scratch normalization; leading silence retained; quantized notes clipped at prompt boundary",
            "simultaneous_note_order": "quantized MIDI parser order",
        },
    )


def save_checked(midi, path):
    serialized = copy.deepcopy(midi)
    before = note_signature(serialized)
    serialized.to_midi().save(path)
    after = note_signature(serialized)
    reloaded = MidiDict.from_midi(path)
    if note_signature(reloaded) != after:
        raise ValueError("Saved MIDI differs from the exporter's note records")
    audit = read_performance(path)
    if any(
        audit.issues.get(key)
        for key in ["unmatched_note_on", "unmatched_note_off", "nonpositive_note_duration"]
    ):
        raise ValueError("Independent MIDI export audit failed")
    return reloaded, {
        "path": str(path),
        "sha256": sha256_file(path),
        "notes": len(after),
        "notes_merged_on_export": len(before) - len(after),
        "note_records_changed_on_export": sum(a != b for a, b in zip(before, after))
        if len(before) == len(after)
        else None,
        "serialization_passed": True,
        "independent_parse_passed": True,
    }


def export_generation(tokenizer, prompt_midi, sequence, boundary, directory: Path):
    eos = tokenizer.identity["special_token_ids"]["<E>"]
    export_ids = sequence if sequence[-1] == eos else sequence + [eos]
    combined = tokenizer.decode(export_ids)
    combined, combined_report = save_checked(combined, directory / "combined.mid")
    if combined_report["notes_merged_on_export"]:
        raise ValueError("MIDI export merged note groups; inspect the sampled sequence")
    before = note_signature(prompt_midi)
    retained = [note for note in note_signature(combined) if note[2] < boundary]
    if retained != before:
        raise ValueError("Generated MIDI changed the supplied quantized prompt")
    continuation = [
        (pitch, velocity, start - boundary, end - boundary)
        for pitch, velocity, start, end in note_signature(combined)
        if start >= boundary
    ]
    if not continuation:
        raise ValueError("No saved continuation notes")
    _, continuation_report = save_checked(flat_midi(continuation), directory / "continuation.mid")
    _, prompt_report = save_checked(prompt_midi, directory / "prompt.mid")
    return {
        "combined": combined_report,
        "continuation": continuation_report,
        "prompt": prompt_report,
        "prompt_preserved": True,
        "note_count_preserved": True,
        "export_eos_appended": sequence[-1] != eos,
        "continuation_first_onset_seconds": min(note[2] for note in continuation) / 1000,
        "continuation_note_end_seconds": max(note[3] for note in continuation) / 1000,
        "continuation_pitch_range": [
            min(note[0] for note in continuation),
            max(note[0] for note in continuation),
        ],
    }
