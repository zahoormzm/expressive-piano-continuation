"""Training state restored at optimizer-update boundaries, including crop and global RNGs."""

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from music_project.data.windows import TokenCorpus, WindowSampler, collate
from music_project.io import sha256_file, write_json
from music_project.models import build_model
from music_project.training.config import source_identity


def seed_everything(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def rng_state() -> dict:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {
            "kind": numpy[0],
            "keys": numpy[1].tolist(),
            "position": numpy[2],
            "gaussian": numpy[3],
            "cached": numpy[4],
        },
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    numpy = state["numpy"]
    np.random.set_state(
        (
            numpy["kind"],
            np.asarray(numpy["keys"], dtype=np.uint32),
            numpy["position"],
            numpy["gaussian"],
            numpy["cached"],
        )
    )


def precision_context(device: torch.device, precision: str):
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16"
        else contextlib.nullcontext()
    )


def loss_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def make_optimizer(model, training):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": training["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
    )


def optimize(
    model,
    optimizer,
    batches: list[dict],
    device: torch.device,
    precision: str,
    gradient_clip: float,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    valid = sum(batch["valid_tokens"] for batch in batches)
    if valid <= 0:
        raise ValueError("Optimizer update has no valid target tokens")
    total_loss = torch.zeros((), device=device)
    for batch in batches:
        inputs, targets, mask = (
            batch[key].to(device) for key in ("input_ids", "targets", "attention_mask")
        )
        with precision_context(device, precision):
            subtotal = loss_sum(model(inputs, mask), targets)
        if not torch.isfinite(subtotal).item():
            raise FloatingPointError("Nonfinite training loss")
        (subtotal / valid).backward()
        total_loss += subtotal.detach()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "loss": float(total_loss.item() / valid),
        "valid_tokens": valid,
        "gradient_norm": float(norm.item()),
        "clipped": bool(norm.item() > gradient_clip),
    }


class Trainer:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        seed_everything(config["seed"], config["deterministic"])
        self.device = torch.device(config["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly choose CPU for diagnostic runs")
        if (
            self.device.type == "cuda"
            and config["precision"] == "bf16"
            and not (torch.cuda.is_bf16_supported())
        ):
            raise RuntimeError("BF16 is not supported by this GPU")
        data, training = config["data"], config["training"]
        self.train_data = TokenCorpus(
            Path(data["cache"]), data["index"], Path(data["manifest"]), "train"
        )
        self.validation_data = TokenCorpus(
            Path(data["cache"]), data["index"], Path(data["manifest"]), "validation"
        )
        if self.train_data.identity["sha256"] != data["tokenizer_sha256"] or (
            self.train_data.manifest_sha256 != data["manifest_sha256"]
        ):
            raise ValueError("Configured data/tokenizer hashes do not match the current artifacts")
        context = training["context_length"]
        fixed = None
        if training["overfit_windows"]:
            fixed = self.train_data.fixed_windows(context, 1)[: training["overfit_windows"]]
        self.sampler = WindowSampler(
            self.train_data, context, config["seed"], fixed, training.get("musical_training")
        )
        self.validation_windows = self.validation_data.fixed_windows(
            context, training["validation_windows_per_piece"]
        )
        validation_identity = [
            {"source": window.source_id, "sequence": window.sequence, "origin_ms": window.origin_ms}
            for window in self.validation_windows
        ]
        self.identity = {
            "manifest_sha256": data["manifest_sha256"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "index_sha256": self.train_data.index_sha256,
            "validation_windows_sha256": hashlib.sha256(
                json.dumps(validation_identity, sort_keys=True).encode()
            ).hexdigest(),
            "source_sha256": source_identity(),
        }
        self.model = build_model(
            config["model"],
            self.train_data.identity["vocab_size"],
            self.train_data.identity["special_token_ids"]["<P>"],
            context,
        ).to(self.device)
        self.optimizer = make_optimizer(self.model, training)
        self.state = {
            "update": 0,
            "valid_tokens_seen": 0,
            "best_validation_loss": None,
            "training_seconds": 0.0,
            "schedule_tokens_seen": 0,
        }
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def learning_rate(self, tokens: int) -> float:
        settings = self.config["training"]
        fraction = min(1.0, tokens / settings["token_budget"])
        warmup = settings["warmup_fraction"]
        if warmup and fraction < warmup:
            return settings["learning_rate"] * fraction / warmup
        cosine_fraction = (fraction - warmup) / (1 - warmup)
        minimum = settings["minimum_lr_fraction"]
        return settings["learning_rate"] * (
            minimum + (1 - minimum) * (1 + math.cos(math.pi * cosine_fraction)) / 2
        )

    def update(self) -> dict:
        settings = self.config["training"]
        batches = [
            self.sampler.batch(settings["microbatch"]) for _ in range(settings["accumulation"])
        ]
        tokens = sum(batch["valid_tokens"] for batch in batches)
        next_tokens = self.state["valid_tokens_seen"] + tokens
        learning_rate = self.learning_rate(next_tokens)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            started = time.perf_counter()
        result = optimize(
            self.model,
            self.optimizer,
            batches,
            self.device,
            self.config["precision"],
            settings["gradient_clip"],
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        self.state["update"] += 1
        self.state["valid_tokens_seen"] = self.state["schedule_tokens_seen"] = next_tokens
        self.state["training_seconds"] += seconds
        return result | {
            "update": self.state["update"],
            "valid_tokens_seen": next_tokens,
            "learning_rate": learning_rate,
            "seconds": seconds,
            "tokens_per_second": result["valid_tokens"] / seconds,
        }

    @torch.no_grad()
    def evaluate(self, windows=None) -> dict:
        self.model.eval()
        windows = self.validation_windows if windows is None else windows
        total_loss, valid = 0.0, 0
        settings = self.config["training"]
        for start in range(0, len(windows), settings["microbatch"]):
            batch = collate(
                windows[start : start + settings["microbatch"]],
                settings["context_length"],
                self.train_data.identity["special_token_ids"]["<P>"],
            )
            inputs, targets, mask = (
                batch[key].to(self.device) for key in ("input_ids", "targets", "attention_mask")
            )
            with precision_context(self.device, self.config["precision"]):
                subtotal = loss_sum(self.model(inputs, mask), targets)
            total_loss += subtotal.item()
            valid += batch["valid_tokens"]
        if not valid or not math.isfinite(total_loss):
            raise FloatingPointError("Invalid validation loss or empty evaluation")
        return {"loss": total_loss / valid, "valid_tokens": valid, "windows": len(windows)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        payload = {
            "format_version": 1,
            "config": self.config,
            "identity": self.identity,
            "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "state": self.state,
            "sampler": self.sampler.state_dict(),
            "rng": rng_state(),
            "tokenizer": self.train_data.identity,
            "parameter_count": self.parameter_count,
        }
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        identity = {
            key: value for key, value in payload["identity"].items() if key != "initialization"
        }
        expected = {key: value for key, value in self.identity.items() if key != "initialization"}
        if payload["format_version"] != 1 or identity != expected:
            raise ValueError(
                "Checkpoint code, data, tokenizer, or fixed evaluation windows disagree"
            )
        if payload["config"] != self.config:
            raise ValueError("Resume requires the exact saved training configuration")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.state = payload["state"]
        self.sampler.load_state_dict(payload["sampler"])
        restore_rng(payload["rng"])
        self.identity = copy.deepcopy(payload["identity"])

    def initialize_from(self, path: Path) -> None:
        """Continue compatible weights/Adam/RNGs in a new run with a new local schedule.

        Unlike exact resume, the old source and schedule may differ intentionally.
        Architecture, corpus, tokenizer, evaluation and optimizer settings must match.
        The parent is never modified and its content hash is retained in every checkpoint.
        """
        if self.state["update"] or "initialization" in self.identity:
            raise ValueError("Initialization requires a fresh trainer")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ignored = {"source_sha256", "initialization"}
        parent_identity = {k: v for k, v in payload["identity"].items() if k not in ignored}
        expected = {k: v for k, v in self.identity.items() if k not in ignored}
        if payload["format_version"] != 1 or parent_identity != expected:
            raise ValueError(
                "Initialization requires identical data, tokenizer and validation windows"
            )
        previous = payload["config"]
        if previous["model"] != self.config["model"]:
            raise ValueError("Initialization requires an identical model architecture")
        for key in (
            "context_length",
            "microbatch",
            "accumulation",
            "betas",
            "weight_decay",
            "gradient_clip",
            "overfit_windows",
            "validation_windows_per_piece",
        ):
            if previous["training"][key] != self.config["training"][key]:
                raise ValueError(f"Initialization requires identical {key}")
        if (
            previous["seed"] != self.config["seed"]
            or payload["tokenizer"] != self.train_data.identity
        ):
            raise ValueError("Initialization requires the same sampler seed and tokenizer")
        if (
            previous["device"] != self.config["device"]
            or previous["precision"] != self.config["precision"]
        ):
            raise ValueError("Initialization requires the same device and precision")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        transform_mode = self.sampler.load_state_dict(payload["sampler"], initialize=True)
        restore_rng(payload["rng"])
        ancestor = payload["identity"].get("initialization", {})
        self.identity["initialization"] = {
            "mode": "continue_weights_optimizer_sampler_rng_reset_local_schedule",
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "parent_identity": payload["identity"],
            "parent_config": previous,
            "musical_training_stream": transform_mode,
            "parent_valid_tokens_seen": payload["state"]["valid_tokens_seen"]
            + ancestor.get("parent_valid_tokens_seen", 0),
            "parent_updates": payload["state"]["update"] + ancestor.get("parent_updates", 0),
            "parent_training_seconds": payload["state"]["training_seconds"]
            + ancestor.get("parent_training_seconds", 0),
        }


def train(
    config: dict,
    run: Path,
    resume: Path | None = None,
    stop_after_updates: int | None = None,
    initialize_from: Path | None = None,
):
    if resume is not None and initialize_from is not None:
        raise ValueError("Choose exact resume or new-run initialization")
    trainer = Trainer(config)
    if resume:
        trainer.load(resume)
    elif initialize_from:
        trainer.initialize_from(initialize_from)
    if resume is None:
        run.mkdir(parents=True, exist_ok=False)
    settings = config["training"]
    write_json(run / "resolved-config.json", config, immutable=True)
    write_json(run / "identity.json", trainer.identity, immutable=True)
    write_json(
        run / "environment.json",
        {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None,
            "precision": config["precision"],
            "parameter_count": trainer.parameter_count,
            "packages": {
                package.metadata["Name"]: package.version
                for package in importlib.metadata.distributions()
            },
            "data_loading_workers": 0,
            "limited_pilot_data": config["data"]["limited_pilot"],
            "train_performances": len(trainer.train_data.performances),
            "validation_performances": len(trainer.validation_data.performances),
            "test_data_used": False,
            "source_commit": None,
        },
    )
    print(
        f"{config['name']}: {trainer.parameter_count:,} parameters, "
        f"context={settings['context_length']}, precision={config['precision']}",
        flush=True,
    )
    metrics_path = run / "metrics.jsonl"
    if resume and metrics_path.exists():
        # Remove log entries newer than the restored atomic checkpoint. This
        # handles a crash between logging an update and saving its checkpoint.
        retained = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if json.loads(line)["update"] <= trainer.state["update"]
        ]
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8"
        )
    initial = trainer.evaluate()
    if trainer.state["best_validation_loss"] is None:
        trainer.state["best_validation_loss"] = initial["loss"]
        write_json(run / "initial-validation.json", initial, immutable=True)
        trainer.save(run / "best.pt")
        trainer.save(run / "latest.pt")
    try:
        while trainer.state["valid_tokens_seen"] < settings["token_budget"]:
            result = trainer.update()
            finished = result["valid_tokens_seen"] >= settings["token_budget"]
            stopping = stop_after_updates is not None and result["update"] >= stop_after_updates
            if result["update"] % settings["validation_every"] == 0 or finished or stopping:
                validation = trainer.evaluate()
                result["validation_loss"] = validation["loss"]
                result["validation_tokens"] = validation["valid_tokens"]
                if validation["loss"] < trainer.state["best_validation_loss"]:
                    trainer.state["best_validation_loss"] = validation["loss"]
                    trainer.save(run / "best.pt")
            with metrics_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(result) + "\n")
                log.flush()
            if result["update"] % settings["checkpoint_every"] == 0 or finished or stopping:
                trainer.save(run / "latest.pt")
            if result["update"] == 1 or result["update"] % 5 == 0 or finished or stopping:
                print(json.dumps(result), flush=True)
            if stopping:
                break
    except KeyboardInterrupt:
        print(
            f"Interrupted; resume the last complete atomic checkpoint at {run / 'latest.pt'}",
            flush=True,
        )
        raise
    parent = trainer.identity.get("initialization", {})
    summary = trainer.state | {
        "cumulative_valid_tokens_seen": trainer.state["valid_tokens_seen"]
        + parent.get("parent_valid_tokens_seen", 0),
        "cumulative_updates": trainer.state["update"] + parent.get("parent_updates", 0),
        "cumulative_training_seconds": trainer.state["training_seconds"]
        + parent.get("parent_training_seconds", 0),
        "completed_budget": trainer.state["valid_tokens_seen"] >= settings["token_budget"],
        "parameter_count": trainer.parameter_count,
        "musical_training_statistics": dict(trainer.sampler.musical_training.stats)
        if trainer.sampler.musical_training
        else None,
        "initial_validation_loss": json.loads((run / "initial-validation.json").read_text())[
            "loss"
        ],
        "final_validation_loss": trainer.evaluate()["loss"],
        "identity": trainer.identity,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
        if trainer.device.type == "cuda"
        else None,
    }
    write_json(run / "summary.json", summary)
    return summary
