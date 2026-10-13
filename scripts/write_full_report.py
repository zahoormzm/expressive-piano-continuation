"""Write a readable milestone report from the audited full-run evidence."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.io import write_json


def main():
    evidence = json.loads(
        Path("outputs/full-milestone-verification.json").read_text(encoding="utf-8")
    )
    runs = evidence["runs"]
    lines = [
        "# Full-corpus training and continuation — 6 October 2026",
        "",
        "Both scratch runs completed their 100-million-valid-target budgets on the retained MAESTRO corpus. All four best/latest checkpoints passed strict identity/configuration loading and reproduced their recorded fixed validation losses. The app now exposes the selected full-corpus checkpoints alongside the unmodified pretrained Aria baseline.",
        "",
        "Loss is next-token cross-entropy in nats on the same 536 fixed crops from 134 validation performances (1,093,516 valid targets). Training uses 957 performances. Test data was not used for fitting, selection, or model evaluation.",
        "",
        "| Model | Parameters | Valid training targets | Updates | Initial validation | Best validation | Final validation | Optimizer computation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        s = run["summary"]
        lines.append(
            f"| {run['model']['kind'].upper()} | {s['parameter_count']:,} | {s['valid_tokens_seen']:,} | {s['update']:,} | {s['initial_validation_loss']:.4f} | {s['best_validation_loss']:.4f} | {s['final_validation_loss']:.4f} | {s['training_seconds']:.2f} s |"
        )
    lines.extend(
        [
            "",
            "The models use seed 42, BF16, 2,048-token context, microbatch four and accumulation four, AdamW, clipping at 1.0, and token-based warmup/cosine scheduling. They start fresh rather than extending the earlier pilots. Architectures remain a one-layer width-256 LSTM and an eight-layer width-512 Transformer with eight heads, SwiGLU width 1,536, pre-LayerNorm, rotary positions and tied embedding/output weights.",
            "",
            "Both runs have identical sampled token counts at every optimizer update and matching source, manifest, token index, tokenizer, and validation-window identities. Every update is present and all logged losses/gradient norms are finite. Complete-update overshoot is included in the table. Crops are sampled within individual performances; 100 million targets should not be described as a deterministic number of dataset epochs. Supplied prefixes and padding are excluded from loss. Optimizer time excludes crop preparation, validation, checkpoint writes and initialization.",
            "",
            "Separate 20-warmup/100-measured-microbatch benchmarks, including crop sampling and optimizer work, measured 71,939 valid targets/s for LSTM and 55,731/s for Transformer, with peak allocations of 1.88 and 4.18 GiB. Benchmark extrapolations exclude validation/checkpoint overhead. Full-run peak allocations are recorded in the machine-readable evidence.",
            "",
            "These are single-seed training experiments with different parameter counts. A lower validation loss here does not establish a musical-quality ranking. Exact recovery was previously established for the deterministic FP32 diagnostics; these BF16 runs have separate saved-checkpoint loss replay, without a claim of exact interrupted-versus-continuous BF16 training replay.",
            "",
            "## Controlled generation",
            "",
            "The fixed protocol was declared before these samples were generated: three authored eight-second prompts (arpeggio, melody with held accompaniment, syncopated minor chords), seeds 42/43, all three models, temperature 0.98, min-p 0.035, 384-new-token ceiling, and a 12-second new-onset horizon. Authored prompts are engineering fixtures; no test performance was selected. Every declared attempt is retained, including failures.",
            "",
            "| Prompt | Seed | LSTM new notes / stop | Transformer new notes / stop | Aria new notes / stop |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    suite = evidence["generation_suite"]
    attempts = suite["attempts"]
    for prompt in suite["protocol"]["prompts"]:
        for seed in suite["protocol"]["seeds"]:
            row = [prompt["id"], str(seed)]
            for model in suite["protocol"]["models"]:
                attempt = next(
                    a
                    for a in attempts
                    if a["prompt"] == prompt["id"]
                    and a["seed"] == seed
                    and a["model"] == model["id"]
                )
                row.append(
                    f"{attempt['generated_notes']} / {attempt['stop_reason'].replace('_', ' ')}"
                    if attempt["success"]
                    else "FAILED (retained)"
                )
            lines.append("| " + " | ".join(row) + " |")
    adjusted_samples = [a for a in attempts if a["success"] and a["note_ends_changed_on_export"]]
    lines.extend(
        [
            "",
            f"MIDI serialization changed note endings in {len(adjusted_samples)} successful samples. The largest recorded adjustment is {max((a['max_note_end_adjustment_ms'] for a in adjusted_samples), default=0)} ms. The audit checks unchanged pitch, velocity, onset and note counts against the raw sampled tokens, and records changed endings separately. This exporter behavior can affect repeated-note articulation; zero merged notes does not mean every sampled duration is unchanged.",
        ]
    )
    lines.extend(
        [
            "",
            f"The suite contains **{suite['successes']} successful attempts and {suite['failures']} failures**. Successful exports preserve all quantized prompt notes and generated note groups with zero merges; matching prompt WAV hashes are checked within each group. Every successful sample's MIDI/WAV hashes were rechecked from disk. Piano audio is nonzero and unclipped at fixed gain 0.25, using the same checksum-pinned renderer/SoundFont and settings as the reconstruction pairs.",
            "",
            "All models use shared grammar/range/density constraints. This differs from Aria's native sampler and does not measure each model's unconstrained syntax validity. Note releases and renderer tails may extend beyond the horizon. Cards report the selected checkpoint's actual training-token count; a best checkpoint can precede the final update. No samples receive an invented human score, and model labels remain visible on the comparison page.",
            "",
            "## Listening and application",
            "",
            "[Reconstruction pairs](http://127.0.0.1:8766/reconstruction-listening/) compare the opening 20 seconds of 12 training performances using the same piano. All 24 audio files pass the automated checks; human timing/sustain assessment is pending. Representation limits and browser-check scope are described in [reconstruction listening notes](reconstruction-listening.md). The user's authorization to run training is separate from an auditory assessment.",
            "",
            "[Full-model comparisons](http://127.0.0.1:8766/full-listening/) contain every fixed prompt/seed group. [Piano continuation studio](http://127.0.0.1:8767/) accepts uploaded piano MIDI or the demo and exposes the full-corpus LSTM/Transformer plus pretrained Aria. Historical pilot jobs keep their original labels. The cooperative GPU lock prevents project training, generation, benchmark and checkpoint-validation commands from overlapping; the app displays when the GPU is busy. It does not control unrelated GPU applications.",
            "",
            "## Artifacts and reproduction",
            "",
            "- [Loss curves](../outputs/full-training-curves.png).",
            "- [Complete audited evidence](full-training-results.json).",
            "- [Checkpoint loss replay](../outputs/full-checkpoint-verification.json).",
            "- [All declared sample attempts](../outputs/full-generation/suite.json).",
            "- [Fixed sample protocol](../configs/generation_suite.json).",
        ]
    )
    for run in runs:
        lines.append(
            f"- [{run['model']['kind'].upper()} summary](../{run['directory']}/summary.json) and [selected checkpoint](../{run['directory']}/best.pt)."
        )
    source = evidence["training_source"]
    generation_source = evidence["generation_source"]
    lines.extend(
        [
            "",
            f"Exact training source: `{source['source_sha256']}`. The archive `{source['snapshot']}` has SHA256 `{source['snapshot_sha256']}`. The extracted directory is `{source['frozen_directory']}`. No Git commit is recorded yet. Data, weights, outputs and frozen checkouts are ignored artifacts; preserve them separately when moving the project.",
            "",
            f"Exact app/generation source: `{generation_source['source_sha256']}`. Archive `{generation_source['snapshot']}` has SHA256 `{generation_source['snapshot_sha256']}`. Every successful fixed-suite sample records this generation identity, separately from its checkpoint's original training identity.",
            "",
            "From the main workspace, use the environment executable and the archived scripts for strict load/replay:",
            "",
            "```powershell",
            "$pythonExe = '.\\.venv\\Scripts\\python.exe'",
            f"& $pythonExe {source['frozen_directory']}/scripts/verify_checkpoints.py --runs {' '.join(source['runs'])}",
            "& $pythonExe scripts/build_listening_page.py --suite outputs/full-generation/suite.json --output outputs/full-listening/index.html",
            "& $pythonExe scripts/serve_app.py",
            "& $pythonExe -m http.server 8766 --bind 127.0.0.1 --directory outputs",
            "```",
            "",
            "Resuming uses the archived `train.py --resume runs/SELECTED_RUN/latest.pt` with the main workspace as the current directory. These runs have completed their budgets; resume validates the completed state and does not extend training. For a fresh experiment, use a new run directory and preserve its own source snapshot. The source for subsequent app/generation work is recorded separately.",
            "",
            "Next: record human listening judgments, decide whether more training or optional adaptation is worthwhile, rehearse the demo, and finish the write-up/presentation before 13 October. Full-corpus fitting is complete for this milestone; final project evaluation and submission are still outstanding.",
        ]
    )
    Path("docs/full-training-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(Path("docs/full-training-results.json"), evidence)
    print("Wrote docs/full-training-results.md and .json", flush=True)


if __name__ == "__main__":
    main()
