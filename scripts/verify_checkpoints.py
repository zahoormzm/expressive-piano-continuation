"""Strict-load saved scratch checkpoints and replay their fixed validation losses."""

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from music_project.io import sha256_file, write_json
from music_project.resources import gpu_slot
from music_project.training.engine import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/full-checkpoint-verification.json")
    )
    args = parser.parse_args()
    results = []
    with gpu_slot("checkpoint validation"):
        for run in args.runs:
            payload = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
            trainer = Trainer(payload["config"])
            for name in ["best", "latest"]:
                checkpoint = run / f"{name}.pt"
                trainer.load(checkpoint)
                actual = trainer.evaluate()
                summary = json.loads((run / "summary.json").read_text())
                expected = (
                    trainer.state["best_validation_loss"]
                    if name == "best"
                    else summary["final_validation_loss"]
                )
                if not math.isclose(actual["loss"], expected, rel_tol=1e-7, abs_tol=1e-6):
                    raise ValueError("Validation replay differs from the recorded checkpoint loss")
                result = {
                    "run": str(run),
                    "checkpoint": name,
                    "sha256": sha256_file(checkpoint),
                    "strict_load_passed": True,
                    "validation": actual,
                    "expected_loss": expected,
                    "loss_delta": actual["loss"] - expected,
                    "valid_tokens_seen": trainer.state["valid_tokens_seen"],
                    "update": trainer.state["update"],
                    "identity": trainer.identity,
                }
                results.append(result)
                print(
                    f"Verified {run.name}/{name}: loss={actual['loss']:.6f}, delta={result['loss_delta']:.9f}",
                    flush=True,
                )
            del trainer, payload
            torch.cuda.empty_cache()
    write_json(args.output, results)


if __name__ == "__main__":
    main()
