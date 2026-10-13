"""Measure training updates with actual performance windows and optimizer state."""

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from music_project.io import write_json
from music_project.resources import gpu_slot
from music_project.training.config import resolve_config
from music_project.training.engine import Trainer, optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20, help="Warmup microbatches")
    parser.add_argument("--steps", type=int, default=100, help="Measured microbatches")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1:
        parser.error("warmup and steps must be positive")
    config = resolve_config(args.config)
    with gpu_slot("benchmarking", enabled=config["device"] == "cuda"):
        benchmark(args, config)


def benchmark(args, config):
    trainer = Trainer(config)
    settings = config["training"]

    def measure(microbatches):
        total_tokens, updates = 0, 0
        remaining = microbatches
        started = time.perf_counter()
        while remaining:
            count = min(settings["accumulation"], remaining)
            batches = [trainer.sampler.batch(settings["microbatch"]) for _ in range(count)]
            result = optimize(
                trainer.model,
                trainer.optimizer,
                batches,
                trainer.device,
                config["precision"],
                settings["gradient_clip"],
            )
            total_tokens += result["valid_tokens"]
            remaining -= count
            updates += 1
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        return {
            "microbatches": microbatches,
            "optimizer_updates": updates,
            "valid_tokens": total_tokens,
            "seconds": time.perf_counter() - started,
        }

    print(f"Benchmarking {config['name']}: {trainer.parameter_count:,} parameters", flush=True)
    warmup = measure(args.warmup)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = measure(args.steps)
    result.update(
        {
            "config": config,
            "identity": trainer.identity,
            "warmup": warmup,
            "parameter_count": trainer.parameter_count,
            "tokens_per_second": result["valid_tokens"] / result["seconds"],
            "data_loading_included": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if trainer.device.type == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if trainer.device.type == "cuda"
            else None,
            "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None,
            "torch": str(torch.__version__),
            "attention": "PyTorch scaled_dot_product_attention"
            if config["model"]["kind"] == "transformer"
            else "cuDNN LSTM",
            "partial_accumulation_note": "Last group may use fewer microbatches; losses remain token-weighted.",
            "planned_pilot_compute_seconds": settings["token_budget"]
            / (result["valid_tokens"] / result["seconds"]),
            "includes_validation_and_checkpointing": False,
        }
    )
    output = args.output or Path("outputs/benchmarks") / f"{config['name']}.json"
    write_json(output, result)
    print(
        {
            key: result[key]
            for key in (
                "parameter_count",
                "tokens_per_second",
                "peak_allocated_bytes",
                "planned_pilot_compute_seconds",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
