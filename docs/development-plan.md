# Project development plan

The user's implementation-plan DOCX is the reference blueprint. The user confirmed a revised submission deadline of **13 October 2026**; this supersedes the document's 10 October deadline. A deadline time and review slot have not been supplied.

The initial scope is expressive solo piano: a MIDI prompt, reproducible continuation, editable MIDI and piano audio. The data foundation, training/recovery checks, GPU benchmarks, pretrained baseline, rendering and upload app are implemented. Both full-corpus scratch runs completed 100,003,399 valid targets and passed saved-checkpoint loss replay. Eighteen fixed samples and 12 reconstruction listening pairs are available; the app exposes the full-corpus models. Human listening assessment and final submission material remain outstanding.

## Proposed schedule

| Dates | Outcome | Evidence |
| --- | --- | --- |
| 5 October | Windows/CUDA setup, dataset acquisition, validator, reconstruction | Hardware snapshot, manifests, reconstruction reports |
| 6 October | Completed screens, full-data runs, baseline/rendering, upload app and fixed comparison suite | Verified 1,262 caches, both 100M-target runs, 18 samples, 12 listening pairs, 43 tests |
| 7 October | Human reconstruction and continuation listening | Recorded judgments with sample IDs/settings |
| 8–9 October | Decide whether further fitting, sampler changes or optional adaptation is useful | Evidence-based choice; preserve new runs separately |
| 10 October | Rehearse selected models and freeze demo protocol | Tested uploads, live generation and fallback comparisons |
| 11 October | Freeze candidates and collect controlled comparisons | Numerical diagnostics, listening responses and failures |
| 12 October | Rehearse demo and finish submission material | README rehearsal, slides and write-up |
| 13 October | Final repository/access and submission audit | Submitted artifacts and final code identity |

Move demonstration readiness earlier if the assigned review precedes this schedule. Automated reconstruction, tiny-data learning, checkpoint recovery and memory/speed checks preceded the user-authorized full runs. Human auditory review remains separate and pending. Further training or adaptation should use the combined numerical and listening evidence.

## Implementation decisions so far

- Keep the existing Windows workspace for code. All runtime directories are configurable or relative; the work can be moved to a local folder outside OneDrive later.
- Use an isolated Python 3.11 environment and the official CUDA-enabled PyTorch wheel.
- Download MAESTRO MIDI and metadata; omit the large paired-audio archive.
- Pin the reviewed Aria tokenizer source and retain its full vocabulary.
- Preserve original MIDI and official splits. Record warnings, exclusions, normalization, and hashes.
- Quarantine recordings with cross-split canonical metadata ambiguity in a derived manifest, while preserving the official manifest and all original split labels.
- Keep diagnostic harmonic preview audio for timing checks; use checksum-pinned FluidSynth and GeneralUser GS piano audio with fixed gain and settings for generated samples.
- Sample windows only within a performance, preserve complete note groups, rebase timing, and exclude padding and the supplied prefix from next-token loss.
- Train both scratch screens at a 2,048-token context and five-million-valid-token budget; record any complete-update overshoot. Keep fixed validation crops identical and exclude test data from training and model selection.
- Record configuration, source/data identities, sampler and global random states, optimizer state, and token-based learning-rate progress in atomic checkpoints.
- Compare scratch and pretrained weights under one explicit constrained sampling protocol; distinguish it from upstream native sampling and distinguish untouched checkpoint files from BF16 inference casting.
- Reject MIDI exports that merge generated note groups, preserve prompt notes, and record whether EOS was sampled or appended solely for serialization.

The 6 October work brought full fitting and candidate integration forward. LSTM validation loss reached 4.3161 and Transformer 3.2758 on identical fixed full-corpus crops; all four saved checkpoints reproduced those losses. The 18-sample suite covers three authored prompts and two seeds under one sampler/renderer protocol. Every artifact passed its integrity audit; repeated-note ending changes and token-limit stops are recorded. All 1,262 caches retain their earlier audit evidence. See `docs/full-training-results.md` for current results and `docs/reconstruction-listening.md` for auditory-review scope. Optional adaptation depends on listening evidence and time. A remote repository, human rehearsal, final write-up/presentation and submission remain outstanding.

On 8 October, the user prioritized melody and musical coherence. A separate 50M-target Transformer continuation completed, reaching 150,007,166 cumulative targets. Fixed validation loss improved to 3.1935; conditional next-pitch accuracy rose from 21.13% to 25.31%. All 16 new matched generations succeeded, and eight A/B listening pairs are ready. The app selects the continued checkpoint; original matched-budget baselines remain preserved. Human responses are still pending. See `docs/transformer-improvement.md` for numerical evidence and the listening protocol.


Later on 8 October, the user supplied eight informal Transformer judgments and selected musical augmentation with continuation-focused loss. That 10M-target screen completed and passed checkpoint, generation and local HTTP checks. Fixed validation loss is 3.1786; next-pitch accuracy is 26.00%. The app now selects **Transformer · musical fine-tuning**. Eight new pairs await listening; the first feedback and original experiments remain preserved. See `docs/transformer-musical-training.md`. Final presentation/write-up, rehearsal and submission remain outstanding.
