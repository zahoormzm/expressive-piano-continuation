"""Audit completed full-data runs and the predeclared generation suite from disk."""

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from ariautils.midi import MidiDict

from music_project.io import sha256_file, write_json
from music_project.tokens.aria import AriaTokenizer, note_signature


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("outputs/full-generation/suite.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/full-milestone-verification.json")
    )
    args = parser.parse_args()
    source = read("outputs/full-training-source.json")
    require(
        sha256_file(Path(source["snapshot"])) == source["snapshot_sha256"],
        "Training source archive changed",
    )
    replay = read("outputs/full-checkpoint-verification.json")
    trajectories, runs = [], []
    for run_name in source["runs"]:
        run = Path(run_name)
        config, summary = read(run / "resolved-config.json"), read(run / "summary.json")
        initial = read(run / "initial-validation.json")
        environment = read(run / "environment.json")
        require(
            config["seed"] == 42
            and config["precision"] == "bf16"
            and config["training"]["context_length"] == 2048,
            "Training seed, precision or context differs from the declared protocol",
        )
        require(
            config["training"]["token_budget"]
            == source["valid_token_budget_per_model"]
            == 100_000_000,
            "Training budget differs from the declared protocol",
        )
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        require(
            summary["completed_budget"] and summary["valid_tokens_seen"] >= 100_000_000,
            "Training budget incomplete",
        )
        require(
            not config["data"]["limited_pilot"] and not environment["test_data_used"],
            "Wrong training corpus",
        )
        require(
            environment["train_performances"] == 957
            and environment["validation_performances"] == 134,
            "Unexpected performance counts",
        )
        require(
            config["training"]["microbatch"] == 4 and config["training"]["accumulation"] == 4,
            "Batching differs from the recorded protocol",
        )
        require(
            initial["windows"] == 536 and initial["valid_tokens"] == 1_093_516,
            "Validation crop counts changed",
        )
        require(
            [r["update"] for r in rows] == list(range(1, summary["update"] + 1)),
            "Missing/duplicate optimizer updates",
        )
        total = 0
        for row in rows:
            total += row["valid_tokens"]
            require(total == row["valid_tokens_seen"], "Invalid token accounting")
            require(
                all(
                    math.isfinite(row[key])
                    for key in ["loss", "gradient_norm", "seconds", "learning_rate"]
                ),
                "Non-finite training metrics",
            )
            if "validation_loss" in row:
                require(
                    math.isfinite(row["validation_loss"])
                    and row["validation_tokens"] == initial["valid_tokens"],
                    "Invalid validation result",
                )
        require(
            total == summary["valid_tokens_seen"] == summary["schedule_tokens_seen"],
            "Summary/schedule tokens disagree",
        )
        require(
            total - config["training"]["token_budget"] < rows[-1]["valid_tokens"],
            "Budget exceeded by more than a final complete update",
        )
        require(
            math.isclose(
                sum(r["seconds"] for r in rows), summary["training_seconds"], rel_tol=1e-9
            ),
            "Optimizer time accounting disagrees",
        )
        require(
            summary["identity"]["source_sha256"] == source["source_sha256"],
            "Unexpected training code",
        )
        checked = [r for r in replay if Path(r["run"]) == run]
        require(
            {r["checkpoint"] for r in checked} == {"best", "latest"},
            "Missing saved-checkpoint validation",
        )
        for result in checked:
            require(
                result["strict_load_passed"] and abs(result["loss_delta"]) <= 1e-6,
                "Checkpoint replay failed",
            )
            require(
                sha256_file(run / f"{result['checkpoint']}.pt") == result["sha256"],
                "Validated checkpoint changed",
            )
            require(result["identity"] == summary["identity"], "Saved identity mismatch")
        trajectories.append([r["valid_tokens_seen"] for r in rows])
        validations = [r for r in rows if "validation_loss" in r]
        runs.append(
            {
                "directory": run_name,
                "model": config["model"],
                "summary": summary,
                "initial_validation": initial,
                "validation_history": validations,
                "checkpoints": checked,
                "config_sha256": sha256_file(run / "resolved-config.json"),
                "metrics_sha256": sha256_file(run / "metrics.jsonl"),
                "optimizer_updates_contiguous": True,
                "all_logged_metrics_finite": True,
                "test_data_used": False,
            }
        )
    require(
        trajectories[0] == trajectories[1], "Models sampled different training-window token counts"
    )
    require(
        runs[0]["summary"]["identity"] == runs[1]["summary"]["identity"],
        "Model data/code/validation identities differ",
    )
    suite = read(args.suite)
    generation_source = read("outputs/full-generation-source.json")
    require(
        sha256_file(Path(generation_source["snapshot"])) == generation_source["snapshot_sha256"],
        "Generation source archive changed",
    )
    prepared_prompts = read("outputs/evaluation-prompts/report.json")
    require(
        prepared_prompts["config_sha256"] == suite["config_sha256"],
        "Prompt preparation used a different protocol",
    )
    prepared_audio = {
        p["declared"]["id"]: p["audio"]["wav_sha256"] for p in prepared_prompts["prompts"]
    }
    protocol = suite["protocol"]
    expected = {
        (p["id"], seed, m["id"])
        for p in protocol["prompts"]
        for seed in protocol["seeds"]
        for m in protocol["models"]
    }
    attempted = [(a["prompt"], a["seed"], a["model"]) for a in suite["attempts"]]
    require(
        suite["complete"] and set(attempted) == expected and len(attempted) == len(expected),
        "Suite omitted or repeated declared attempts",
    )
    groups, samples = {}, []
    tokenizer = AriaTokenizer()
    for attempt in suite["attempts"]:
        if not attempt["success"]:
            samples.append(attempt)
            continue
        directory = Path(attempt["directory"])
        report = read(directory / "report.json")
        require(
            report["generation_source_sha256"] == generation_source["source_sha256"],
            "Generation source differs between samples",
        )
        require(
            report["audio"]["prompt"]["wav_sha256"] == prepared_audio[attempt["prompt"]],
            "Generation differs from the prepared prompt audio",
        )
        require(
            report["seed"] == attempt["seed"] and report["model"]["backend"] == attempt["model"],
            "Sample model/seed differs",
        )
        require(
            report["prompt"]["source_sha256"] == suite["prompt_hashes"][attempt["prompt"]],
            "Sample prompt differs",
        )
        for setting in ["temperature", "min_p"]:
            require(report[setting] == protocol[setting], "Sampler settings differ")
        require(
            report["sampling"]["new_tokens"] <= protocol["max_new_tokens"],
            "Generation exceeds token budget",
        )
        require(
            report["export"]["prompt_preserved"] and report["export"]["note_count_preserved"],
            "MIDI integrity failed",
        )
        for kind in ["prompt", "combined", "continuation"]:
            midi, audio = report["export"][kind], report["audio"][kind]
            require(
                midi["notes_merged_on_export"] == 0 and midi["independent_parse_passed"],
                "MIDI export changed note counts",
            )
            require(
                sha256_file(directory / f"{kind}.mid") == midi["sha256"],
                "MIDI changed after validation",
            )
            require(
                sha256_file(directory / f"{kind}.wav") == audio["wav_sha256"],
                "WAV changed after validation",
            )
            require(audio["clipped_samples"] == 0 and audio["rms"] > 0, "Clipped or empty audio")
            require(
                audio["gain"] == 0.25 and audio["sample_rate"] == 44100 and audio["channels"] == 2,
                "Renderer settings differ",
            )
        group = (attempt["prompt"], attempt["seed"])
        groups.setdefault(group, set()).add(report["audio"]["prompt"]["wav_sha256"])
        if attempt["model"] != "aria":
            require(
                report["model"]["checkpoint_sha256"]
                == suite["checkpoint_hashes"][attempt["model"]],
                "Sample checkpoint differs",
            )
        ids = np.load(directory / "sampled-tokens.npy", allow_pickle=False).tolist()
        eos = tokenizer.identity["special_token_ids"]["<E>"]
        decoded = note_signature(tokenizer.decode(ids if ids[-1] == eos else ids + [eos]))
        require(
            len(decoded)
            == report["prompt"]["prompt_notes"] + report["sampling"]["generated_note_groups"],
            "Raw sampled note groups disagree with the report",
        )
        saved = note_signature(MidiDict.from_midi(directory / "combined.mid"))
        expected_notes = {note[:3]: note[3] for note in decoded}
        actual_notes = {note[:3]: note[3] for note in saved}
        require(
            len(expected_notes) == len(decoded) and len(actual_notes) == len(saved),
            "Duplicate note identity in declared sample",
        )
        require(
            expected_notes.keys() == actual_notes.keys(), "Export changed pitch, velocity or onset"
        )
        end_adjustments = [actual_notes[key] - end for key, end in expected_notes.items()]
        samples.append(
            attempt
            | {
                "report_sha256": sha256_file(directory / "report.json"),
                "model_identity": report["model"],
                "generated_notes": report["export"]["continuation"]["notes"],
                "stop_reason": report["sampling"]["stop_reason"],
                "sampling_seconds": report["sampling_seconds"],
                "continuation_pitch_range": report["export"]["continuation_pitch_range"],
                "continuation_note_end_seconds": report["export"]["continuation_note_end_seconds"],
                "peak_gpu_allocated_bytes": report["peak_gpu_allocated_bytes"],
                "sampled_tokens_sha256": sha256_file(directory / "sampled-tokens.npy"),
                "note_ends_changed_on_export": sum(delta != 0 for delta in end_adjustments),
                "max_note_end_adjustment_ms": max(map(abs, end_adjustments), default=0),
            }
        )
    require(
        all(len(hashes) == 1 for hashes in groups.values()),
        "Prompt audio differs within a comparison group",
    )
    reconstruction = read("outputs/reconstruction-listening/report.json")
    require(len(reconstruction["pairs"]) == 12, "Missing reconstruction pairs")
    for pair in reconstruction["pairs"]:
        require(
            pair["source"]["notes"] == pair["reconstruction"]["notes"],
            "Reconstruction excerpt note counts differ",
        )
        for kind in ["source", "reconstruction"]:
            excerpt = pair[kind]
            require(
                sha256_file(Path(excerpt["midi"])) == excerpt["midi_sha256"],
                "Reconstruction excerpt MIDI changed",
            )
            require(
                sha256_file(Path(excerpt["midi"]).with_suffix(".wav"))
                == excerpt["audio"]["wav_sha256"],
                "Reconstruction excerpt audio changed",
            )
            require(
                excerpt["audio"]["clipped_samples"] == 0 and excerpt["audio"]["rms"] > 0,
                "Clipped/empty reconstruction audio",
            )
    result = {
        "date": "2026-10-06",
        "training_source": source,
        "generation_source": generation_source,
        "reconstruction_listening": {
            "pairs": 12,
            "audio_files": 24,
            "all_artifact_hashes_verified": True,
            "note_counts_match": True,
            "all_audio_nonzero_and_unclipped": True,
            "human_assessed": False,
        },
        "runs": runs,
        "identical_training_token_trajectories": True,
        "identical_data_and_validation_identities": True,
        "generation_suite": {
            "path": str(args.suite),
            "sha256": sha256_file(args.suite),
            "protocol": protocol,
            "attempts": samples,
            "successes": suite["successes"],
            "failures": suite["failures"],
        },
        "human_assessed": False,
        "musical_quality_ranking": None,
    }
    write_json(args.output, result)
    print(f"Verified two full runs and all {len(samples)} generation attempts", flush=True)


if __name__ == "__main__":
    main()
