import mido
import numpy as np
import pytest
import torch
from ariautils.midi import MidiDict

from music_project.data.windows import PerformanceTokens, Window, collate
from music_project.tokens.aria import AriaTokenizer, note_signature


def test_event_crops_preserve_rebased_timing_and_complete_groups(midi_builder):
    events = []
    for start, end, pitch in [
        (120, 300, 60),
        (4900, 5100, 64),
        (5040, 5300, 67),
        (6400, 6500, 72),
        (15100, 15400, 48),
    ]:
        events.extend(
            [
                (start, mido.Message("note_on", note=pitch, velocity=80)),
                (end, mido.Message("note_off", note=pitch)),
            ]
        )
    tokenizer = AriaTokenizer()
    midi, _ = tokenizer.normalize(MidiDict.from_midi(midi_builder(events)))
    ids = tokenizer.encode(midi)
    performance = PerformanceTokens(np.asarray(ids), list(tokenizer.tokenizer.vocab), "piece-one")
    crop = performance.crop(2, 12)
    decoded = note_signature(tokenizer.decode(crop.sequence))
    expected = [
        (pitch, velocity, onset - crop.origin_ms, end - crop.origin_ms)
        for pitch, velocity, onset, end in tokenizer.expected_notes(midi)
        if pitch in (67, 72, 48)
    ]
    assert decoded == sorted(expected)
    assert crop.source_id == "piece-one"
    assert crop.origin_ms == 5000
    assert len(crop.sequence) <= 13
    shortened = performance.crop(0, 8)
    assert shortened.end_event < len(performance.events)
    assert tokenizer.tokenizer.eos_tok not in tokenizer.tokenizer.decode(shortened.sequence)
    assert tokenizer.tokenizer.decode(shortened.sequence)[-1][0] == "dur"


def test_collation_shifts_once_and_ignores_forced_prefix_and_padding():
    window = Window([5, 6, 7, 8, 9], 2, "first", 0, 1, 0, 0, 3)
    other = Window([5, 6, 10, 11], 2, "second", 0, 1, 0, 0, 3)
    batch = collate([window, other], 6, 0)
    assert batch["input_ids"][0].tolist() == [5, 6, 7, 8, 0, 0]
    assert batch["targets"][0].tolist() == [-100, 7, 8, 9, -100, -100]
    assert batch["targets"][1].tolist() == [-100, 10, 11, -100, -100, -100]
    assert batch["valid_tokens"] == 5
    assert batch["source_ids"] == ["first", "second"]
    assert torch.equal(batch["attention_mask"][0], torch.tensor([1, 1, 1, 1, 0, 0]).bool())


def test_malformed_note_group_is_rejected(midi_builder):
    path = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (100, mido.Message("note_off", note=60)),
        ]
    )
    tokenizer = AriaTokenizer()
    midi, _ = tokenizer.normalize(MidiDict.from_midi(path))
    ids = tokenizer.encode(midi)
    # Remove onset token while leaving the end token.
    del ids[3]
    with pytest.raises(ValueError):
        PerformanceTokens(np.asarray(ids), list(tokenizer.tokenizer.vocab), "broken")
