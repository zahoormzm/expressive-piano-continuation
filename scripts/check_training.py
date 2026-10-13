"""Actual-GPU tiny-data learning and interrupted-versus-continuous recovery checks."""

import argparse
import copy
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from music_project.io import write_json
from music_project.training.config import resolve_config
from music_project.training.engine import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("outputs/training-checks"))
    args = parser.parse_args()
    reports = []
    for kind, config_file in (
        ("lstm", "configs/lstm.yaml"),
        ("transformer", "configs/transformer_small.yaml"),
    ):
        config = resolve_config(Path(config_file))
        config["name"] = f"tiny-{kind}-diagnostic"
        config["precision"] = "fp32"
        config["deterministic"] = True
        config["model"] = (
            {"kind": "lstm", "width": 64, "layers": 1, "dropout": 0.0}
            if kind == "lstm"
            else {
                "kind": "transformer",
                "width": 64,
                "layers": 2,
                "heads": 4,
                "feedforward": 192,
                "dropout": 0.0,
                "tied_embeddings": True,
            }
        )
        config["training"].update(
            {
                "context_length": 64,
                "microbatch": 1,
                "accumulation": 1,
                "overfit_windows": 1,
                "learning_rate": 0.003,
                "weight_decay": 0,
                "token_budget": args.updates * 64,
                "validation_windows_per_piece": 1,
            }
        )
        trainer = Trainer(config)
        initial = trainer.evaluate(trainer.sampler.fixed)["loss"]
        for _ in range(args.updates):
            result = trainer.update()
            if result["update"] % 50 == 0:
                print(
                    f"{kind} tiny update {result['update']}: loss={result['loss']:.4f}", flush=True
                )
        final = trainer.evaluate(trainer.sampler.fixed)["loss"]
        learned = final < initial * 0.35
        report = {
            "kind": kind,
            "purpose": "tiny-data diagnostic; not a musical quality comparison",
            "config": config,
            "initial_fixed_training_loss": initial,
            "final_fixed_training_loss": final,
            "learning_passed": learned,
            "updates": args.updates,
            "parameter_count": trainer.parameter_count,
        }
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        # Dropout is enabled for replay to exercise random-state restoration.
        recovery_config = copy.deepcopy(config)
        recovery_config["training"]["overfit_windows"] = 0
        recovery_config["model"]["dropout"] = 0.15
        continuous = Trainer(recovery_config)
        continuous.update()
        continuous.update()
        checkpoint = args.output / kind / "recovery.pt"
        continuous.save(checkpoint)
        reference = [continuous.update(), continuous.update()]
        expected = {
            key: value.detach().cpu().clone()
            for key, value in continuous.model.state_dict().items()
        }
        expected_sampler = copy.deepcopy(continuous.sampler.state_dict())
        del continuous
        gc.collect()
        torch.cuda.empty_cache()
        recovered = Trainer(recovery_config)
        recovered.load(checkpoint)
        replay = [recovered.update(), recovered.update()]
        max_delta = max(
            (expected[key] - value.detach().cpu()).abs().max().item()
            for key, value in recovered.model.state_dict().items()
        )
        loss_delta = max(
            abs(before["loss"] - after["loss"]) for before, after in zip(reference, replay)
        )
        same_sampler = recovered.sampler.state_dict() == expected_sampler
        report.update(
            {
                "checkpoint_recovery_max_weight_delta": max_delta,
                "checkpoint_recovery_max_loss_delta": loss_delta,
                "sampler_replay_passed": same_sampler,
                "checkpoint_recovery_passed": max_delta <= 1e-6
                and loss_delta <= 1e-6
                and same_sampler,
            }
        )
        write_json(args.output / kind / "report.json", report)
        reports.append(report)
        print(
            json.dumps({key: value for key, value in report.items() if key != "config"}), flush=True
        )
        del recovered
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "summary.json", reports)
    if not all(
        report["learning_passed"] and report["checkpoint_recovery_passed"] for report in reports
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
