import copy
import random

import mido
import pytest
from ariautils.midi import MidiDict

from music_project.tokens.aria import AriaTokenizer, note_signature


def test_chords_repeated_notes_rests_and_segment_boundaries(midi_builder, tmp_path):
    events = []
    for start, end, pitch, velocity in [
        (117, 566, 60, 73),
        (117, 566, 64, 85),
        (117, 566, 67, 97),
        (601, 804, 60, 60),
        (4999, 5501, 72, 100),
        (5000, 5200, 76, 81),
        (15_123, 15_569, 48, 32),
        (20_000, 27_000, 65, 77),
    ]:
        events.append((start, mido.Message("note_on", note=pitch, velocity=velocity)))
        events.append((end, mido.Message("note_off", note=pitch)))
    path = midi_builder(events)
    tokenizer = AriaTokenizer()
    report = tokenizer.reconstruct(path, tmp_path / "reconstruction")
    assert report["expected_quantized_note_count"] == 8
    assert report["quantized_reconstruction_passed"]
    assert report["midi_serialization_passed"]
    assert report["duration_clipped_at_5000ms_count"] == 1
    decoded = MidiDict.from_midi(tmp_path / "reconstruction/reconstructed.mid")
    notes = note_signature(decoded)
    assert any(pitch == 72 and start == 4990 for pitch, _, start, _ in notes)
    assert any(pitch == 76 and start == 5000 for pitch, _, start, _ in notes)


def test_sustain_is_applied_once_and_repeated_pitch_overlap_counted(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("control_change", control=64, value=127)),
            (100, mido.Message("note_on", note=60, velocity=80)),
            (200, mido.Message("note_off", note=60)),
            (300, mido.Message("note_on", note=60, velocity=90)),
            (400, mido.Message("note_off", note=60)),
            (500, mido.Message("control_change", control=64, value=0)),
        ]
    )
    tokenizer = AriaTokenizer()
    original = MidiDict.from_midi(path)
    untouched = copy.deepcopy(original.get_msg_dict())
    normalized, report = tokenizer.normalize(original)
    assert original.get_msg_dict() == untouched
    assert report["note_records_changed_by_sustain_or_overlap"] == 2
    decoded = tokenizer.decode(tokenizer.encode(normalized))
    assert note_signature(decoded) == [(60, 80, 100, 300), (60, 90, 300, 500)]
    assert not decoded.pedal_msgs


def test_tokenizer_uses_complete_tempo_map(midi_builder, tmp_path):
    path = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (500, mido.MetaMessage("set_tempo", tempo=1_000_000)),
            (1000, mido.Message("note_off", note=60)),
            (1000, mido.Message("note_on", note=64, velocity=70)),
            (1500, mido.Message("note_off", note=64)),
        ]
    )
    tokenizer = AriaTokenizer()
    report = tokenizer.reconstruct(path, tmp_path / "reconstruction")
    assert report["quantized_reconstruction_passed"]
    decoded = MidiDict.from_midi(tmp_path / "reconstruction/reconstructed.mid")
    assert note_signature(decoded) == [(60, 80, 0, 1500), (64, 70, 1500, 2500)]


def test_encoding_is_deterministic_without_changing_caller_rng(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (100, mido.Message("note_off", note=60)),
        ]
    )
    tokenizer = AriaTokenizer()
    midi, _ = tokenizer.normalize(MidiDict.from_midi(path))
    state = random.getstate()
    first = tokenizer.encode(midi)
    assert state == random.getstate()
    assert first == tokenizer.encode(midi)
    assert tokenizer.decode(first).note_msgs


def test_rejects_drums_and_unknown_ids(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("note_on", channel=9, note=36, velocity=80)),
            (100, mido.Message("note_off", channel=9, note=36)),
        ]
    )
    tokenizer = AriaTokenizer()
    with pytest.raises(ValueError, match="solo piano"):
        tokenizer.normalize(MidiDict.from_midi(path))
    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([tokenizer.tokenizer.vocab_size])
