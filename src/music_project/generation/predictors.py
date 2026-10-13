"""Inference adapters; saved scratch weights and pinned native Aria weights remain unchanged."""

import json
import subprocess
import sys
from pathlib import Path

import torch

from music_project.io import sha256_file
from music_project.models import build_model
from music_project.training.engine import precision_context


class ScratchPredictor:
    def __init__(self, checkpoint, tokenizer, device, precision):
        self.device, self.precision = torch.device(device), precision
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or (
            json.dumps(payload["tokenizer"], sort_keys=True)
            != json.dumps(tokenizer.identity, sort_keys=True)
        ):
            raise ValueError(
                "Scratch checkpoint format/tokenizer disagrees with the current pipeline"
            )
        config = payload["config"]
        self.context_length = config["training"]["context_length"]
        self.kind = config["model"]["kind"]
        self.model = build_model(
            config["model"],
            tokenizer.identity["vocab_size"],
            tokenizer.identity["special_token_ids"]["<P>"],
            self.context_length,
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.hidden, self.processed = None, 0
        self.identity = {
            "backend": self.kind,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_source_sha256": payload["identity"]["source_sha256"],
            "training_valid_tokens": payload["state"]["valid_tokens_seen"]
            + payload["identity"].get("initialization", {}).get("parent_valid_tokens_seen", 0),
            "run_training_valid_tokens": payload["state"]["valid_tokens_seen"],
            "training_initialization": payload["identity"].get("initialization"),
            "parameter_count": payload["parameter_count"],
            "weights_modified": False,
            "inference": "recurrent cache" if self.kind == "lstm" else "full context recomputation",
        }

    def next_logits(self, sequence):
        if len(sequence) > self.context_length:
            raise ValueError("Generation exceeds the scratch checkpoint context")
        tokens = sequence[self.processed :] if self.kind == "lstm" else sequence
        inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with precision_context(self.device, self.precision):
            if self.kind == "lstm":
                embedded = self.model.dropout(self.model.embedding(inputs))
                hidden, self.hidden = self.model.lstm(embedded, self.hidden)
                logits = self.model.head(self.model.dropout(hidden[:, -1]))[0]
                self.processed = len(sequence)
            else:
                logits = self.model(inputs, last_only=True)[0, -1]
        return logits


class AriaPredictor:
    def __init__(
        self, tokenizer, device, precision, total_tokens, asset_registry=Path("configs/assets.json")
    ):
        from safetensors.torch import load_file

        self.device, self.precision = torch.device(device), precision
        if self.device.type != "cuda" or precision != "bf16":
            raise ValueError("This verified native Aria adapter requires CUDA and BF16")
        assets = json.loads(asset_registry.read_text())
        asset = assets["pretrained"]
        checkpoint, source = Path(asset["path"]), Path(asset["source_directory"])
        if sha256_file(checkpoint) != asset["sha256"]:
            raise ValueError("Pretrained checkpoint checksum mismatch")
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
        if revision != asset["source_revision"] or dirty:
            raise ValueError("Native Aria source must be the pinned unmodified revision")
        sys.path.insert(0, str(source.resolve()))
        from aria.inference.model_cuda import TransformerLM
        from aria.model import ModelConfig

        config_values = json.loads((source / asset["model_config"]).read_text())
        self.context_length = config_values["max_seq_len"]
        if (
            config_values["vocab_size"] != tokenizer.identity["vocab_size"]
            or total_tokens > self.context_length
        ):
            raise ValueError("Aria vocabulary size or generation context mismatch")
        config = ModelConfig(**config_values)
        state = load_file(str(checkpoint), device="cpu")
        with torch.device("meta"):
            self.model = TransformerLM(config)
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.to(device=self.device, dtype=torch.bfloat16).eval()
        del state
        self.model.setup_cache(batch_size=1, max_seq_len=total_tokens, dtype=torch.bfloat16)
        self.processed = 0
        self.identity = {
            "backend": "aria",
            "repository": asset["repository"],
            "revision": asset["revision"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": asset["sha256"],
            "source_revision": revision,
            "model_config": config_values,
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "weights_modified": False,
            "inference": "native PyTorch KV cache, no compilation",
        }

    def next_logits(self, sequence):
        tokens = sequence[self.processed :]
        inputs = torch.tensor([tokens], device=self.device, dtype=torch.long)
        positions = torch.arange(self.processed, len(sequence), device=self.device)
        with precision_context(self.device, self.precision):
            logits = self.model(idxs=inputs, input_pos=positions)[0, -1]
        self.processed = len(sequence)
        return logits
