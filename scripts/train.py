"""Train, exactly resume, or continue a compatible checkpoint in a new recorded run."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from music_project.resources import gpu_slot
from music_project.training.config import resolve_config
from music_project.training.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--config", type=Path)
    choice.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path, help="Continue a checkpoint in a new run")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--stop-after-updates", type=int, help="Stop at an update boundary for recovery checks"
    )
    args = parser.parse_args()
    if args.initialize_from and args.resume:
        parser.error("--initialize-from requires --config and a new run directory")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        parser.error("stop-after-updates must be positive")
    if args.resume:
        config = torch.load(args.resume, map_location="cpu", weights_only=True)["config"]
        run = args.resume.parent
        if args.run_dir is not None and args.run_dir.resolve() != run.resolve():
            parser.error("Resume keeps the checkpoint's run directory")
    else:
        config = resolve_config(args.config)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run = args.run_dir or Path("runs") / f"{config['name']}-{stamp}"
    with gpu_slot("training", enabled=config["device"] == "cuda"):
        print(
            json.dumps(
                train(config, run, args.resume, args.stop_after_updates, args.initialize_from),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
