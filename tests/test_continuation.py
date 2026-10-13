import copy
from types import SimpleNamespace

import pytest
import torch

from music_project.data.windows import Window
from music_project.io import sha256_file
from music_project.training import engine


@pytest.fixture
def tiny_config(monkeypatch):
    class Corpus:
        def __init__(self, cache, index, manifest, split):
            self.split = split
            self.identity = {"sha256": "tokens", "vocab_size": 32, "special_token_ids": {"<P>": 0}}
            self.manifest_sha256 = "manifest"
            self.index_sha256 = "index"
            self.performances = [SimpleNamespace(events=[1])]

        def fixed_windows(self, context, count):
            return [Window([1, 2, 3, 4, 5], 1, self.split, 0, 1, 0, 0, 4)] * count

    monkeypatch.setattr(engine, "TokenCorpus", Corpus)
    monkeypatch.setattr(engine, "source_identity", lambda: "current-source")
    return {
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
            "context_length": 8,
            "microbatch": 1,
            "accumulation": 1,
            "overfit_windows": 1,
            "validation_windows_per_piece": 1,
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "warmup_fraction": 0,
            "minimum_lr_fraction": 0.5,
            "token_budget": 16,
            "validation_every": 2,
            "checkpoint_every": 2,
        },
    }


def test_continuation_preserves_adam_rng_and_sampler_then_resumes_exactly(tiny_config, tmp_path):
    parent = engine.Trainer(tiny_config)
    parent.update()
    parent.update()
    checkpoint = tmp_path / "parent.pt"
    parent.save(checkpoint)
    parent_hash = sha256_file(checkpoint)
    expected_sampler = copy.deepcopy(parent.sampler.state_dict())
    config = copy.deepcopy(tiny_config)
    config["name"] = "continued"
    config["training"]["learning_rate"] = 0.0001
    config["training"]["token_budget"] = 32
    continued = engine.Trainer(config)
    continued.initialize_from(checkpoint)
    assert continued.state["valid_tokens_seen"] == continued.state["update"] == 0
    assert continued.sampler.state_dict() == expected_sampler
    for before, after in zip(parent.model.parameters(), continued.model.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(parent.optimizer.state.values(), continued.optimizer.state.values()):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    continued.update()
    saved = tmp_path / "continued.pt"
    continued.save(saved)
    expected_loss = continued.update()["loss"]
    expected_weights = copy.deepcopy(continued.model.state_dict())
    replay = engine.Trainer(config)
    replay.load(saved)
    assert replay.update()["loss"] == expected_loss
    for key, value in replay.model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
    assert replay.identity["initialization"]["parent_valid_tokens_seen"] == 8
    assert replay.identity["initialization"]["checkpoint_sha256"] == parent_hash
    assert sha256_file(checkpoint) == parent_hash


@pytest.mark.parametrize("change", ["architecture", "data", "optimizer", "context"])
def test_continuation_rejects_incompatible_parent(tiny_config, tmp_path, change):
    parent = engine.Trainer(tiny_config)
    path = tmp_path / "parent.pt"
    parent.save(path)
    payload = torch.load(path, weights_only=True)
    if change == "architecture":
        payload["config"]["model"]["layers"] = 2
    elif change == "data":
        payload["identity"]["index_sha256"] = "other"
    elif change == "optimizer":
        payload["config"]["training"]["betas"] = [0.9, 0.99]
    else:
        payload["config"]["training"]["context_length"] = 16
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Initialization requires"):
        engine.Trainer(tiny_config).initialize_from(path)


def test_new_run_accepts_previous_source_but_exact_resume_rejects_it(tiny_config, tmp_path):
    parent = engine.Trainer(tiny_config)
    parent.identity["source_sha256"] = "old-source"
    path = tmp_path / "parent.pt"
    parent.save(path)
    fresh = engine.Trainer(tiny_config)
    with pytest.raises(ValueError, match="Checkpoint code"):
        fresh.load(path)
    fresh.initialize_from(path)
    assert fresh.identity["source_sha256"] == "current-source"
    assert fresh.identity["initialization"]["parent_identity"]["source_sha256"] == "old-source"


def test_new_run_records_cumulative_tokens_without_extending_old_run(tiny_config, tmp_path):
    parent = engine.Trainer(tiny_config)
    parent.update()
    path = tmp_path / "parent.pt"
    parent.save(path)
    run = tmp_path / "run"
    summary = engine.train(tiny_config, run, stop_after_updates=1, initialize_from=path)
    assert summary["valid_tokens_seen"] == 4
    assert summary["cumulative_valid_tokens_seen"] == 8
    assert summary["cumulative_updates"] == 2
    assert summary["initial_validation_loss"] == parent.evaluate()["loss"]
