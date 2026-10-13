"""Continue a MIDI prompt with scratch or unmodified pretrained Aria weights."""

import argparse
import json
import math
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from music_project.audio.piano import render_piano
from music_project.generation.limits import parse_token_budget, resolve_token_budget
from music_project.generation.midi import build_prompt, export_generation
from music_project.generation.predictors import AriaPredictor, ScratchPredictor
from music_project.generation.sampling import sample_sequence
from music_project.io import write_json
from music_project.resources import gpu_slot
from music_project.tokens.aria import AriaTokenizer
from music_project.training.config import source_identity
from music_project.training.engine import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["scratch", "aria"], default="scratch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--prompt-seconds", type=float, default=8)
    parser.add_argument(
        "--new-seconds",
        type=float,
        default=12,
        help="Bound new note onsets; note releases may extend later",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=parse_token_budget,
        default=384,
        help="New-token ceiling, or 'auto' to use the context left after the prompt",
    )
    parser.add_argument("--temperature", type=float, default=0.98)
    parser.add_argument("--min-p", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.backend == "scratch" and args.checkpoint is None:
        parser.error("Scratch inference requires --checkpoint")
    if args.backend == "aria" and args.checkpoint is not None:
        parser.error("Aria inference uses the checksum-pinned asset registry")
    if not math.isfinite(args.new_seconds) or args.new_seconds <= 0:
        parser.error("new-seconds must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    with gpu_slot("generation", enabled=args.device == "cuda"):
        generate(args)


def generate(args):
    seed_everything(args.seed, deterministic=False)
    tokenizer = AriaTokenizer()
    prompt, prompt_midi, prompt_report = build_prompt(args.prompt, args.prompt_seconds, tokenizer)
    print("PROGRESS\t" + json.dumps({"stage": "loading"}), flush=True)
    if args.backend == "aria":
        asset = json.loads(Path("configs/assets.json").read_text(encoding="utf-8"))["pretrained"]
        config = json.loads(
            (Path(asset["source_directory"]) / asset["model_config"]).read_text(encoding="utf-8")
        )
        budget = resolve_token_budget(args.max_new_tokens, len(prompt), config["max_seq_len"])
        total = len(prompt) + budget
        predictor = AriaPredictor(tokenizer, args.device, args.precision, total)
    else:
        predictor = ScratchPredictor(args.checkpoint, tokenizer, args.device, args.precision)
        budget = resolve_token_budget(args.max_new_tokens, len(prompt), predictor.context_length)
    args.output.mkdir(parents=True, exist_ok=False)
    start_ms = prompt_report["boundary_ms"]
    end_ms = start_ms + math.ceil(args.new_seconds * 1000 / 10) * 10
    if args.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    print("PROGRESS\t" + json.dumps({"stage": "generating"}), flush=True)
    sequence, sampling_report = sample_sequence(
        predictor,
        tokenizer.identity["vocabulary"],
        prompt,
        start_ms,
        end_ms,
        budget,
        args.temperature,
        args.min_p,
        args.seed,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    sampling_seconds = time.perf_counter() - started
    np.save(
        args.output / "sampled-tokens.npy", np.asarray(sequence, dtype=np.int32), allow_pickle=False
    )
    export = export_generation(tokenizer, prompt_midi, sequence, start_ms, args.output)
    report = {
        "model": predictor.identity,
        "tokenizer_sha256": tokenizer.identity["sha256"],
        "generation_source_sha256": source_identity(),
        "prompt": prompt_report,
        "seed": args.seed,
        "temperature": args.temperature,
        "min_p": args.min_p,
        "precision": args.precision,
        "device": args.device,
        "sampling_seconds": sampling_seconds,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "sampling": sampling_report,
        "generation_limits": {
            "context_tokens": predictor.context_length,
            "prompt_tokens": len(prompt),
            "max_new_tokens": budget,
            "budget_mode": "remaining_context" if args.max_new_tokens == "auto" else "fixed",
            "requested_new_seconds": args.new_seconds,
        },
        "export": export,
        "quality_evaluated": False,
        "protocol": "shared constrained decoding; native upstream sampler is not used",
    }
    if args.render:
        print("PROGRESS\t" + json.dumps({"stage": "rendering"}), flush=True)
        report["audio"] = {
            name: render_piano(args.output / f"{name}.mid", args.output / f"{name}.wav")
            for name in ["prompt", "combined", "continuation"]
        }
    write_json(args.output / "report.json", report)
    print(
        {
            "output": str(args.output),
            "backend": predictor.identity["backend"],
            "generated_notes": export["continuation"]["notes"],
            "seconds": sampling_seconds,
            "stop": sampling_report["stop_reason"],
            "prompt_preserved": export["prompt_preserved"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
