"""Run every predeclared prompt/seed/model combination and retain failures."""

import argparse
import contextlib
import gc
import json
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from generate import generate

from music_project.io import sha256_file, write_json
from music_project.resources import gpu_slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generation_suite.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full-generation"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("The comparison suite requires CUDA")
    with gpu_slot("comparison generation"):
        run_suite(args)


def run_suite(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": config,
        "config_sha256": sha256_file(args.config),
        "attempts": [],
        "human_assessed": False,
        "complete": False,
    }
    result["prompt_hashes"] = {p["id"]: sha256_file(Path(p["path"])) for p in config["prompts"]}
    result["checkpoint_hashes"] = {
        m["id"]: sha256_file(Path(m["checkpoint"])) for m in config["models"] if m["checkpoint"]
    }
    write_json(args.output / "suite.json", result)
    for prompt in config["prompts"]:
        for seed in config["seeds"]:
            for model in config["models"]:
                sample = args.output / f"{prompt['id']}-{model['id']}-seed{seed}"
                settings = argparse.Namespace(
                    backend="scratch" if model["checkpoint"] else "aria",
                    checkpoint=Path(model["checkpoint"]) if model["checkpoint"] else None,
                    prompt=Path(prompt["path"]),
                    seed=seed,
                    output=sample,
                    render=True,
                    device="cuda",
                    precision="bf16",
                    **{
                        key: config[key]
                        for key in [
                            "prompt_seconds",
                            "new_seconds",
                            "max_new_tokens",
                            "temperature",
                            "min_p",
                        ]
                    },
                )
                started = time.perf_counter()
                log_path = args.output / f"{sample.name}.log"
                success = False
                torch.cuda.reset_peak_memory_stats()
                with log_path.open("w", encoding="utf-8") as log:
                    try:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            generate(settings)
                        success = (sample / "report.json").exists()
                    except Exception:  # noqa: BLE001 -- retain each failed declared attempt
                        traceback.print_exc(file=log)
                gc.collect()
                torch.cuda.empty_cache()
                attempt = {
                    "prompt": prompt["id"],
                    "description": prompt["description"],
                    "model": model["id"],
                    "label": model["label"],
                    "seed": seed,
                    "directory": str(sample),
                    "log": str(log_path),
                    "wall_seconds": time.perf_counter() - started,
                    "success": success,
                }
                if not attempt["success"]:
                    attempt["failure"] = log_path.read_text(encoding="utf-8").splitlines()[-1:]
                result["attempts"].append(attempt)
                write_json(args.output / "suite.json", result)
                print(
                    f"{len(result['attempts'])}/{len(config['prompts']) * len(config['seeds']) * len(config['models'])} {sample.name}: {'complete' if attempt['success'] else 'FAILED'}",
                    flush=True,
                )
    result.update(
        complete=True,
        successes=sum(a["success"] for a in result["attempts"]),
        failures=sum(not a["success"] for a in result["attempts"]),
    )
    write_json(args.output / "suite.json", result)
    print(
        f"Suite complete: {result['successes']} successes, {result['failures']} failures",
        flush=True,
    )


if __name__ == "__main__":
    main()
