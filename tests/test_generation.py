import math

import mido
import pytest
import torch

from music_project.generation.midi import build_prompt, export_generation
from music_project.generation.sampling import PianoGrammar, sample_sequence, sample_token
from music_project.tokens.aria import AriaTokenizer, note_signature


@pytest.fixture
def vocabulary():
    return [
        "<S>",
        "<E>",
        "<T>",
        ("prefix", "instrument", "piano"),
        ("piano", 60, 80),
        ("piano", 61, 0),
        ("piano", 20, 80),
        ("onset", 0),
        ("onset", 10),
        ("onset", 4990),
        ("dur", 0),
        ("dur", 100),
    ]


def test_grammar_enforces_boundary_and_positive_complete_notes(vocabulary):
    grammar = PianoGrammar(vocabulary, [3, 0, 4, 7, 11], 5000, 10000)
    assert grammar.allowed_ids() == [2]
    grammar.consume(2)
    assert (
        4 in grammar.allowed_ids()
        and 5 not in grammar.allowed_ids()
        and 6 not in grammar.allowed_ids()
    )
    grammar.consume(4)
    assert grammar.allowed_ids() == [7, 8, 9]
    grammar.consume(8)
    assert grammar.allowed_ids() == [11]
    grammar.consume(11)
    grammar.consume(4)
    assert 7 not in grammar.allowed_ids()


def test_seeded_sampling_and_zero_temperature(vocabulary):
    values = torch.linspace(-1, 1, len(vocabulary))
    one = torch.Generator().manual_seed(9)
    two = torch.Generator().manual_seed(9)
    assert [sample_token(values, [4, 8, 11], 0.98, 0.035, one) for _ in range(25)] == [
        sample_token(values, [4, 8, 11], 0.98, 0.035, two) for _ in range(25)
    ]
    assert sample_token(values, [4, 8, 11], 0, 0, one) == 11
    with pytest.raises(ValueError):
        sample_token(values, [4], math.nan, 0, one)


def test_same_pitch_cannot_repeat_at_identical_onset(vocabulary):
    grammar = PianoGrammar(vocabulary, [3, 0, 4, 7, 11], 0, 5000)
    grammar.consume(4)
    assert 7 not in grammar.allowed_ids() and 8 in grammar.allowed_ids()
    grammar.consume(9)
    grammar.consume(11)
    assert 4 not in grammar.allowed_ids()


def test_token_budget_never_leaves_partial_note(vocabulary):
    class Predictor:
        device = torch.device("cpu")

        def next_logits(self, sequence):
            logits = torch.zeros(len(vocabulary))
            logits[4], logits[8], logits[11] = 10, 9, 8
            return logits

    sequence, report = sample_sequence(Predictor(), vocabulary, [3, 0], 0, 1000, 5, 0, 0, 42)
    assert sequence == [3, 0, 4, 8, 11]
    assert report["generated_note_groups"] == 1 and not report["sampled_eos"]


def test_predicted_time_marker_stops_at_horizon(vocabulary):
    class Predictor:
        device = torch.device("cpu")

        def next_logits(self, sequence):
            next_id = {5: 4, 6: 9, 7: 11}.get(len(sequence), 2)
            logits = torch.zeros(len(vocabulary))
            logits[next_id] = 20
            return logits

    sequence, report = sample_sequence(
        Predictor(), vocabulary, [3, 0, 4, 7, 11], 1000, 5000, 20, 0, 0, 42
    )
    assert sequence[-1] == 2
    assert report["stop_reason"] == "time_horizon"
    assert report["generated_note_groups"] == 1


def test_prompt_clips_future_notes_and_export_preserves_prompt(midi_builder, tmp_path):
    path = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (1500, mido.Message("note_off", note=60)),
            (1100, mido.Message("note_on", note=64, velocity=80)),
            (1800, mido.Message("note_off", note=64)),
        ]
    )
    tokenizer = AriaTokenizer()
    ids, prompt, report = build_prompt(path, 1, tokenizer)
    assert note_signature(prompt) == [(60, 80, 0, 1000)]
    assert report["notes_clipped_at_prompt_boundary"] == 1
    lookup = tokenizer.tokenizer.tok_to_id
    sequence = ids + [lookup[("piano", 67, 80)], lookup[("onset", 1000)], lookup[("dur", 500)]]
    exported = export_generation(tokenizer, prompt, sequence, 1000, tmp_path)
    assert exported["prompt_preserved"] and exported["export_eos_appended"]
    assert exported["continuation"]["notes"] == 1
    from ariautils.midi import MidiDict

    assert note_signature(MidiDict.from_midi(tmp_path / "continuation.mid")) == [(67, 80, 0, 500)]


def test_recurrent_inference_matches_full_context():
    from music_project.models.networks import LSTMModel

    torch.manual_seed(3)
    model = LSTMModel(32, 0, width=16, dropout=0).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        _, hidden = model.lstm(model.embedding(tokens[:, :3]))
        output, hidden = model.lstm(model.embedding(tokens[:, 3:4]), hidden)
        cached = model.head(output[:, -1])
        expected = model(tokens[:, :4], last_only=True)[:, -1]
    torch.testing.assert_close(cached, expected, rtol=0, atol=1e-6)


def test_export_rejects_duplicate_note_loss(midi_builder, tmp_path):
    source = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (500, mido.Message("note_off", note=60)),
        ]
    )
    tokenizer = AriaTokenizer()
    ids, prompt, _ = build_prompt(source, 1, tokenizer)
    lookup = tokenizer.tokenizer.tok_to_id
    group = [lookup[("piano", 67, 80)], lookup[("onset", 1000)], lookup[("dur", 500)]]
    with pytest.raises(ValueError, match="merged note groups"):
        export_generation(tokenizer, prompt, ids + group + group, 1000, tmp_path)
