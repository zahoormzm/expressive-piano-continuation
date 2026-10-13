"""Verify the pinned native Aria weights, prompt compatibility, and KV-cache logits."""

import argparse
import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from ariautils.midi import MidiDict

from music_project.generation.midi import build_prompt
from music_project.generation.predictors import AriaPredictor
from music_project.io import write_json
from music_project.tokens.aria import AriaTokenizer
from music_project.training.engine import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, default=Path("outputs/prompts/arpeggio.mid"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pretrained-verification.json"))
    args = parser.parse_args()
    seed_everything(42, deterministic=False)
    tokenizer = AriaTokenizer()
    prompt, _, prompt_report = build_prompt(args.prompt, 8, tokenizer)
    predictor = AriaPredictor(tokenizer, "cuda", "bf16", len(prompt) + 8)
    from aria.inference import get_inference_prompt

    native_tokens = get_inference_prompt(
        copy.deepcopy(MidiDict.from_midi(args.prompt)), tokenizer.tokenizer, 8000
    )
    native_ids = tokenizer.tokenizer.encode(native_tokens)
    if native_ids != prompt:
        raise ValueError("The engineering prompt differs from upstream native preprocessing")
    extra = tokenizer.tokenizer.tok_to_id[("piano", 60, 80)]
    extended = prompt + [extra]
    with torch.inference_mode():
        predictor.next_logits(prompt)
        cached = predictor.next_logits(extended).float().clone()
        predictor.model.setup_cache(1, max_seq_len=len(prompt) + 8, dtype=torch.bfloat16)
        predictor.processed = 0
        complete = predictor.next_logits(extended).float()
    if not torch.isfinite(cached).all() or not torch.isfinite(complete).all():
        raise FloatingPointError("Pretrained inference produced nonfinite logits")
    # BF16 matrix kernels differ for prefill and one-token decoding.
    torch.testing.assert_close(cached, complete, rtol=0.03, atol=0.25)
    config = json.loads(Path("data/assets/aria-metadata/config.json").read_text())
    if (
        config["vocab_size"] != tokenizer.identity["vocab_size"]
        or config["eos_token_id"] != tokenizer.identity["special_token_ids"]["<E>"]
        or config["pad_token_id"] != tokenizer.identity["special_token_ids"]["<P>"]
    ):
        raise ValueError("Published vocabulary size or special IDs disagree")
    report = {
        "model": predictor.identity,
        "strict_weight_load_passed": True,
        "vocab_size": tokenizer.identity["vocab_size"],
        "published_special_ids_match": True,
        "tokenizer_sha256": tokenizer.identity["sha256"],
        "synthetic_prompt_matches_upstream_ids": True,
        "prompt": prompt_report,
        "cached_prefill_max_logit_delta": float((cached - complete).abs().max()),
        "cached_prefill_logit_rmse": float((cached - complete).square().mean().sqrt()),
        "cache_check_tolerance": {"rtol": 0.03, "atol": 0.25},
        "cache_check_passed": True,
        "finite_logits": True,
        "preprocessing_scope": "Agreement checked on this project-authored prompt; shared wrapper remains explicit for other prompts.",
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(args.output, report)
    print(
        {
            key: report[key]
            for key in [
                "strict_weight_load_passed",
                "synthetic_prompt_matches_upstream_ids",
                "cached_prefill_max_logit_delta",
                "cache_check_passed",
            ]
        }
    )


if __name__ == "__main__":
    main()
