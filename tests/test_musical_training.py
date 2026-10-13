import copy

import numpy as np
import pytest
import torch

from music_project.data.musical_training import (
    MusicalTraining,
    continuation_window,
    read_notes,
    resolve_musical_training,
)
from music_project.data.windows import PerformanceTokens, collate
from music_project.tokens.aria import AriaTokenizer
from music_project.training import engine


@pytest.fixture
def piano():
    vocabulary = list(AriaTokenizer().tokenizer.vocab)
    lookup = {tuple(t) if isinstance(t, list) else t: i for i, t in enumerate(vocabulary)}

    def window(notes, context=64):
        ids = [lookup[("prefix", "instrument", "piano")], lookup["<S>"]]
        segment = 0
        for pitch, onset, duration in notes:
            new_segment, local = divmod(onset, 5000)
            ids.extend([lookup["<T>"]] * (new_segment - segment))
            ids.extend(
                [lookup[("piano", pitch, 80)], lookup[("onset", local)], lookup[("dur", duration)]]
            )
            segment = new_segment
        ids.append(lookup["<E>"])
        performance = PerformanceTokens(np.asarray(ids), vocabulary, "piece")
        return performance.crop(0, context), performance

    return vocabulary, lookup, window


def test_global_transform_preserves_chords_intervals_and_cross_segment_timing(piano):
    vocabulary, lookup, make = piano
    original, performance = make(
        [(60, 4900, 200), (64, 4900, 300), (67, 5040, 100), (60, 15100, 400)]
    )
    before = performance.ids.copy()
    transform = MusicalTraining(vocabulary, 64, 42, {})
    changed, metadata = transform.augment(original, 3, 1.1)
    notes, prefix = read_notes(changed, vocabulary)
    assert prefix == original.prefix_length == 2
    assert [n.pitch for n in notes] == [63, 67, 70, 63]
    assert [n.onset_ms for n in notes] == [4900, 4900, 5050, 16120]
    assert [n.duration_ms for n in notes] == [220, 330, 110, 440]
    assert all(n.velocity == 80 for n in notes)
    assert changed.sequence[-1] == lookup["<E>"]
    assert metadata["tail_notes_dropped"] == 0
    np.testing.assert_array_equal(performance.ids, before)
    assert original.sequence != changed.sequence
    with pytest.raises(ValueError, match="piano range"):
        transform.augment(original, 50, 1)


def test_transform_never_invents_eos_or_partial_notes_when_context_fills(piano):
    vocabulary, lookup, make = piano
    original, _ = make([(60, 4900, 5000), (64, 14900, 100), (67, 24900, 100)])
    transform = MusicalTraining(vocabulary, 10, 42, {})
    changed, metadata = transform.augment(original, 0, 1.1)
    notes, _ = read_notes(changed, vocabulary)
    assert len(changed.sequence) <= 11
    assert changed.sequence[-1] != lookup["<E>"]
    assert vocabulary[changed.sequence[-1]][0] == "dur"
    assert len(notes) == 2
    assert metadata["tail_notes_dropped"] == 1
    assert metadata["duration_clamps"] == 1
    assert changed.source_token_end == original.source_token_start + 10 - 2
    assert changed.end_event == 2


def test_quantization_collision_keeps_original_timing(piano):
    vocabulary, _, make = piano
    original, _ = make([(60, 0, 100), (60, 40, 100), (60, 50, 100)])
    changed, metadata = MusicalTraining(vocabulary, 64, 1, {}).augment(original, 2, 0.9)
    assert [n.onset_ms for n in read_notes(changed, vocabulary)[0]] == [0, 40, 50]
    assert metadata["time_scale_rejected"] == 1
    assert metadata["time_scale"] == 1


def test_prompt_mask_keeps_full_chords_and_continuation_targets(piano):
    vocabulary, lookup, make = piano
    original, _ = make(
        [
            (60, 4900, 100),
            (64, 4900, 100),
            (67, 6000, 100),
            (60, 7000, 100),
            (64, 7000, 100),
            (67, 8000, 100),
        ]
    )
    focused, count, milliseconds = continuation_window(original, vocabulary, 5000)
    # Half-note cap shortens the requested prompt at a complete onset boundary.
    assert count == 3 and milliseconds == 2100
    assert focused.prefix_length == 12
    batch = collate([focused], 64, lookup["<P>"])
    assert torch.all(batch["targets"][0, :11] == -100)
    assert batch["targets"][0, 11] == lookup[("piano", 60, 80)]
    assert batch["valid_tokens"] == len(original.sequence) - focused.prefix_length
    assert focused.sequence == original.sequence and original.prefix_length == 2
    short, _ = make([(60, 0, 100), (64, 0, 100)])
    assert continuation_window(short, vocabulary, 2000) == (short, 0, 0)


@pytest.mark.parametrize(
    "config",
    [
        {"unknown": 1},
        {"continuation_probability": 2},
        {"semitone_min": -1.5},
        {"time_scale_min": float("nan")},
        {"prompt_seconds_min": 0},
    ],
)
def test_invalid_training_transforms_rejected(config):
    with pytest.raises(ValueError):
        resolve_musical_training(config)


def test_training_stream_resume_replays_crops_masks_rng_and_weights(piano, monkeypatch, tmp_path):
    vocabulary, lookup, make = piano
    example, performance = make([(60 + i % 8, i * 1000, 200) for i in range(12)])

    class Corpus:
        def __init__(self, cache, index, manifest, split):
            self.identity = {
                "sha256": "tokens",
                "vocab_size": len(vocabulary),
                "vocabulary": vocabulary,
                "special_token_ids": {"<P>": lookup["<P>"]},
            }
            self.manifest_sha256, self.index_sha256 = "manifest", "index"
            self.performances = [performance]

        def fixed_windows(self, context, count):
            return [copy.deepcopy(example)] * count

    monkeypatch.setattr(engine, "TokenCorpus", Corpus)
    monkeypatch.setattr(engine, "source_identity", lambda: "musical-source")
    config = {
        "name": "tiny",
        "seed": 42,
        "device": "cpu",
        "precision": "fp32",
        "deterministic": True,
        "data": {
            "cache": "cache",
            "index": "index",
            "manifest": "manifest",
            "tokenizer_sha256": "tokens",
            "manifest_sha256": "manifest",
            "limited_pilot": True,
        },
        "model": {
            "kind": "transformer",
            "width": 16,
            "layers": 1,
            "heads": 4,
            "feedforward": 32,
            "dropout": 0.1,
            "tied_embeddings": True,
        },
        "training": {
            "context_length": 48,
            "microbatch": 2,
            "accumulation": 1,
            "overfit_windows": 1,
            "validation_windows_per_piece": 1,
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip": 1,
            "warmup_fraction": 0,
            "minimum_lr_fraction": 0.5,
            "token_budget": 1000,
            "validation_every": 2,
            "checkpoint_every": 2,
        },
    }
    parent = engine.Trainer(config)
    parent.update()
    saved = tmp_path / "parent.pt"
    parent.save(saved)
    config["training"]["musical_training"] = {
        "augmentation_probability": 1,
        "continuation_probability": 1,
    }
    musical = engine.Trainer(config)
    musical.initialize_from(saved)
    assert musical.identity["initialization"]["musical_training_stream"] == "new_independent_stream"
    assert musical.sampler.rng.bit_generator.state == parent.sampler.rng.bit_generator.state
    assert all(w.prefix_length == 2 for w in musical.validation_windows)
    assert (
        musical.identity["validation_windows_sha256"]
        == parent.identity["validation_windows_sha256"]
    )
    musical.update()
    saved = tmp_path / "musical.pt"
    musical.save(saved)
    expected_loss = musical.update()["loss"]
    expected_weights = copy.deepcopy(musical.model.state_dict())
    expected_sampler = copy.deepcopy(musical.sampler.state_dict())
    replay = engine.Trainer(config)
    replay.load(saved)
    assert replay.update()["loss"] == expected_loss
    assert replay.sampler.state_dict() == expected_sampler
    for key, value in replay.model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
    assert expected_sampler["musical_training"]["stats"]["continuation_windows"] == 4
    bad = torch.load(saved, weights_only=True)
    del bad["sampler"]["musical_training"]
    torch.save(bad, saved)
    with pytest.raises(ValueError, match="RNG state"):
        engine.Trainer(config).load(saved)
