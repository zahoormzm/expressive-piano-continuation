"""Resolve project configuration and validate data identities before a run."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from music_project.data.musical_training import resolve_musical_training


def source_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    files = sorted(root.glob("src/**/*.py")) + sorted(root.glob("scripts/*.py"))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or set(config) - {
        "name",
        "seed",
        "device",
        "precision",
        "deterministic",
        "data",
        "model",
        "training",
        "resolved",
    }:
        raise ValueError("Invalid or unknown top-level configuration field")
    config = copy.deepcopy(config)
    config.setdefault("seed", 42)
    config.setdefault("device", "cuda")
    config.setdefault("precision", "bf16")
    config.setdefault("deterministic", False)
    if config["device"] not in ("cpu", "cuda") or config["precision"] not in ("fp32", "bf16"):
        raise ValueError("Supported devices: cpu/cuda; precisions: fp32/bf16")
    training_defaults = {
        "context_length": 2048,
        "microbatch": 1,
        "accumulation": 16,
        "token_budget": 5_000_000,
        "learning_rate": 0.0003,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "warmup_fraction": 0.03,
        "minimum_lr_fraction": 0.1,
        "validation_every": 20,
        "checkpoint_every": 20,
        "validation_windows_per_piece": 4,
        "overfit_windows": 0,
        "musical_training": None,
    }
    if set(config["training"]) - set(training_defaults):
        raise ValueError("Unknown training configuration field")
    config["training"] = training_defaults | config["training"]
    training = config["training"]
    training["musical_training"] = resolve_musical_training(training["musical_training"])
    for key in (
        "context_length",
        "microbatch",
        "accumulation",
        "token_budget",
        "validation_every",
        "checkpoint_every",
        "validation_windows_per_piece",
    ):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if training["context_length"] < 8 or training["overfit_windows"] < 0:
        raise ValueError("Context must be at least 8; overfit window count must be nonnegative")
    if not 0 <= training["warmup_fraction"] < 1 or not 0 <= training["minimum_lr_fraction"] <= 1:
        raise ValueError("Invalid learning-rate schedule fractions")
    if training["learning_rate"] <= 0 or training["gradient_clip"] <= 0:
        raise ValueError("Learning rate and gradient clipping must be positive")
    summary_path = Path(config["data"].get("token_summary", "outputs/tokenization-summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config["data"].update(
        {
            "cache": summary["cache_directory"],
            "index": summary["index_file"],
            "manifest_sha256": summary["manifest_sha256"],
            "tokenizer_sha256": summary["tokenizer_sha256"],
            "limited_pilot": summary["limited_pilot"],
        }
    )
    config["data"].setdefault("manifest", "data/manifests/maestro-v3-clean.jsonl")
    return config
