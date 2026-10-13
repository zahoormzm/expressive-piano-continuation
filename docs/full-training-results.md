# Full-corpus training and continuation — 6 October 2026

Both scratch runs completed their 100-million-valid-target budgets on the retained MAESTRO corpus. All four best/latest checkpoints passed strict identity/configuration loading and reproduced their recorded fixed validation losses. The app now exposes the selected full-corpus checkpoints alongside the unmodified pretrained Aria baseline.

This report preserves the original full-run and 18-sample milestone. A subsequent [app duration-budget fix](duration-budget-fix.md) uses remaining context for live generation and passes 46 tests; its new samples and source identity are recorded separately.

Loss is next-token cross-entropy in nats on the same 536 fixed crops from 134 validation performances (1,093,516 valid targets). Training uses 957 performances. Test data was not used for fitting, selection, or model evaluation.

| Model | Parameters | Valid training targets | Updates | Initial validation | Best validation | Final validation | Optimizer computation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 9,620,287 | 100,003,399 | 3,239 | 9.7861 | 4.3161 | 4.3161 | 1336.07 s |
| TRANSFORMER | 36,356,608 | 100,003,399 | 3,239 | 9.8901 | 3.2758 | 3.2758 | 1742.14 s |

The models use seed 42, BF16, 2,048-token context, microbatch four and accumulation four, AdamW, clipping at 1.0, and token-based warmup/cosine scheduling. They start fresh rather than extending the earlier pilots. Architectures remain a one-layer width-256 LSTM and an eight-layer width-512 Transformer with eight heads, SwiGLU width 1,536, pre-LayerNorm, rotary positions and tied embedding/output weights.

Both runs have identical sampled token counts at every optimizer update and matching source, manifest, token index, tokenizer, and validation-window identities. Every update is present and all logged losses/gradient norms are finite. Complete-update overshoot is included in the table. Crops are sampled within individual performances; 100 million targets should not be described as a deterministic number of dataset epochs. Supplied prefixes and padding are excluded from loss. Optimizer time excludes crop preparation, validation, checkpoint writes and initialization.

Separate 20-warmup/100-measured-microbatch benchmarks, including crop sampling and optimizer work, measured 71,939 valid targets/s for LSTM and 55,731/s for Transformer, with peak allocations of 1.88 and 4.18 GiB. Benchmark extrapolations exclude validation/checkpoint overhead. Full-run peak allocations are recorded in the machine-readable evidence.

These are single-seed training experiments with different parameter counts. A lower validation loss here does not establish a musical-quality ranking. Exact recovery was previously established for the deterministic FP32 diagnostics; these BF16 runs have separate saved-checkpoint loss replay, without a claim of exact interrupted-versus-continuous BF16 training replay.

## Controlled generation

The fixed protocol was declared before these samples were generated: three authored eight-second prompts (arpeggio, melody with held accompaniment, syncopated minor chords), seeds 42/43, all three models, temperature 0.98, min-p 0.035, 384-new-token ceiling, and a 12-second new-onset horizon. Authored prompts are engineering fixtures; no test performance was selected. Every declared attempt is retained, including failures.

| Prompt | Seed | LSTM new notes / stop | Transformer new notes / stop | Aria new notes / stop |
| --- | ---: | --- | --- | --- |
| arpeggio | 42 | 123 / time horizon | 69 / time horizon | 54 / time horizon |
| arpeggio | 43 | 127 / token budget | 127 / token budget | 54 / time horizon |
| melody | 42 | 107 / time horizon | 57 / time horizon | 30 / time horizon |
| melody | 43 | 127 / token budget | 110 / time horizon | 31 / time horizon |
| syncopation | 42 | 105 / time horizon | 120 / time horizon | 80 / time horizon |
| syncopation | 43 | 127 / token budget | 117 / time horizon | 75 / time horizon |

MIDI serialization changed note endings in 12 successful samples. The largest recorded adjustment is 1400 ms. The audit checks unchanged pitch, velocity, onset and note counts against the raw sampled tokens, and records changed endings separately. This exporter behavior can affect repeated-note articulation; zero merged notes does not mean every sampled duration is unchanged.

The suite contains **18 successful attempts and 0 failures**. Successful exports preserve all quantized prompt notes and generated note groups with zero merges; matching prompt WAV hashes are checked within each group. Every successful sample's MIDI/WAV hashes were rechecked from disk. Piano audio is nonzero and unclipped at fixed gain 0.25, using the same checksum-pinned renderer/SoundFont and settings as the reconstruction pairs.

An earlier suite startup failed before any model attempt because the checksum helper received string paths instead of `Path` objects. The corrected run kept the declared protocol unchanged. Its empty startup directory, failure record and original source snapshot remain available; the 18-attempt result above describes the subsequent completed model suite.

All models use shared grammar/range/density constraints. This differs from Aria's native sampler and does not measure each model's unconstrained syntax validity. Note releases and renderer tails may extend beyond the horizon. Cards report the selected checkpoint's actual training-token count; a best checkpoint can precede the final update. No samples receive an invented human score, and model labels remain visible on the comparison page.

## Listening and application

[Reconstruction pairs](http://127.0.0.1:8766/reconstruction-listening/) compare the opening 20 seconds of 12 training performances using the same piano. All 24 audio files pass the automated checks; human timing/sustain assessment is pending. Representation limits and browser-check scope are described in [reconstruction listening notes](reconstruction-listening.md). The user's authorization to run training is separate from an auditory assessment.

[Full-model comparisons](http://127.0.0.1:8766/full-listening/) contain every fixed prompt/seed group. [Piano continuation studio](http://127.0.0.1:8767/) accepts uploaded piano MIDI or the demo and exposes the full-corpus LSTM/Transformer plus pretrained Aria. Historical pilot jobs keep their original labels. The cooperative GPU lock prevents project training, generation, benchmark and checkpoint-validation commands from overlapping; the app displays when the GPU is busy. It does not control unrelated GPU applications.

The final app completed a full-corpus Transformer demo (job `a6c21d24ea7c460981caef27d75f74ce`, 69 new notes) and an uploaded authored melody with the full-corpus LSTM (job `029b0b772288483dac8da5aa0bf5a864`, 107 new notes). Their raw inputs, sampled tokens, three MIDI files and three WAV files match the corresponding fixed-suite artifacts byte for byte. Model labels, checkpoint hashes and source identities are verified. Every offered artifact returned matching bytes over HTTP, and WAV range requests returned the requested bytes with status 206. A historical Transformer job still reports its pilot label.

Browser checks found finite durations and no media errors for all 24 comparison players. Playback time advanced for all three models in the first group; switching from Transformer to Aria paused the Transformer before its end. The LSTM upload result and its audio metadata loaded in the app, but its app player was not checked for advancing playback. These are interface checks; they do not establish audible quality or a human listening score.

At that milestone, all **43 tests** passed, with a separate eight-test registry recheck. Ruff lint/format and dependency checks passed. Its wheel includes byte-identical interface HTML/server files and declares `filelock`, with SHA256 `657cac341fd809e536f17d3a557e66fdf540c5f1153c50d5247773ef070af007`. Python source then matched the frozen generation identity. Details are in [app/browser evidence](../outputs/full-browser-verification.json) and [package evidence](../outputs/full-package-verification.json); subsequent live-app changes have their own snapshot.

## Artifacts and reproduction

- [Loss curves](../outputs/full-training-curves.png).
- [Complete audited evidence](full-training-results.json).
- [Core training/sample audit](../outputs/full-milestone-verification.json) and [final app/browser verification](../outputs/full-browser-verification.json).
- [Checkpoint loss replay](../outputs/full-checkpoint-verification.json).
- [All declared sample attempts](../outputs/full-generation/suite.json).
- [Fixed sample protocol](../configs/generation_suite.json).
- [LSTM summary](../runs/lstm-full-20261006-001/summary.json) and [selected checkpoint](../runs/lstm-full-20261006-001/best.pt).
- [TRANSFORMER summary](../runs/transformer-small-full-20261006-001/summary.json) and [selected checkpoint](../runs/transformer-small-full-20261006-001/best.pt).

Exact training source: `225d8b0cb206711fd40fc1512e233a47c26133ac3e9567e9ff270212c17431ae`. The archive `outputs/full-training-source-225d8b0cb206711f.zip` has SHA256 `3e31458be7b5f84fbc8f6bbe1d760ac0ecdafd82cee7f8580086c88e6d89346d`. The extracted directory is `outputs/full-training-code-225d8b0cb206711f`. No Git commit is recorded yet. Data, weights, outputs and frozen checkouts are ignored artifacts; preserve them separately when moving the project.

Exact app/generation source for this recorded milestone: `be011d138b141641f9612bd886cc10d29264bb094baaec2caaebd2a05d7bbee7`. Archive `outputs/full-generation-source-be011d138b141641.zip` has SHA256 `258fb2313ca8a7d5f5aff2e872c8b36204c456835a98ddefc06a605c5bcd8958`. Every successful fixed-suite sample records this generation identity, separately from its checkpoint's original training identity.

From the main workspace, use the environment executable and the archived scripts for strict load/replay:

```powershell
$pythonExe = '.\.venv\Scripts\python.exe'
& $pythonExe outputs/full-training-code-225d8b0cb206711f/scripts/verify_checkpoints.py --runs runs/lstm-full-20261006-001 runs/transformer-small-full-20261006-001
& $pythonExe scripts/build_listening_page.py --suite outputs/full-generation/suite.json --output outputs/full-listening/index.html
& $pythonExe scripts/serve_app.py
& $pythonExe -m http.server 8766 --bind 127.0.0.1 --directory outputs
```

Resuming uses the archived `train.py --resume runs/SELECTED_RUN/latest.pt` with the main workspace as the current directory. These runs have completed their budgets; resume validates the completed state and does not extend training. For a fresh experiment, use a new run directory and preserve its own source snapshot. The source for subsequent app/generation work is recorded separately.

Next: record human listening judgments, decide whether more training or optional adaptation is worthwhile, follow [the demonstration guide](demo-guide.md), and finish the write-up/presentation before 13 October. Full-corpus fitting is complete for this milestone; final project evaluation and submission are still outstanding.
## Subsequent Transformer experiment · 8 October

The results above describe the original matched 100M-target baselines. A separate 50M-target Transformer continuation now reaches 150,007,166 cumulative targets and validation loss 3.1935. The studio selects that continued checkpoint, while this original suite and its source/checkpoints remain preserved. See [the Transformer experiment](transformer-improvement.md) and its separate A/B listening page. Further training has not been applied to the LSTM or pretrained Aria.
