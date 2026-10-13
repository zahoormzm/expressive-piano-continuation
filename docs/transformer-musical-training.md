# Musical augmentation and continuation-focused Transformer training

The user selected this experiment on 8 October 2026 after reporting weak melody/coherence, unstable timing and weak harmony. This adds a bounded 10M scored-target run to our current 150M Transformer. Training, saved-checkpoint replay and all declared comparison generations are complete.

## Listening evidence and hypothesis

The supplied responses match the previous comparison suite and checkpoint mappings. One listener judged eight pairs: five preferences for the continued model, two for the original, and one unclear. Seven judgments preceded identity reveal (four continued, two original, one unclear). The fifth continued preference followed reveal. Comments mention timing going off and weak harmony. This supports testing another training method; it is not a population-level preference result.

The preserved attachment and verified mappings are recorded in `outputs/listening-feedback/transformer-feedback-summary.json`. These preferences and the listening prompts are not training targets. No preference optimization or training on the user's Für Elise example is performed.

## Training method

Half of sampled training crops receive passage augmentation. Choose one transposition between −5 and +6 semitones that keeps every pitch within MIDI 21–108. Move all voices by that same amount, preserving intervals and chords. Keep velocities unchanged. Choose one time scale between 0.9 and 1.1 for the whole passage, measured from its first note, and scale durations consistently. Quantize to the existing 10 ms grid and rebuild five-second segment markers. This is a variation of pitch/time augmentation supported by the [pinned Aria toolkit](https://github.com/EleutherAI/aria-utils/blob/7c815755c6889a784b6a1b3343e9e8daa3e7d128/ariautils/tokenizer/absolute.py); our crop implementation adds explicit boundary and replay handling.

If timing quantization would collapse distinct repeated-pitch onsets, keep the original timing for that crop. Count durations clipped at the representation's five-second maximum and any tail notes removed to fit the context. Never invent an end-of-piece label for a truncated crop or divide a note's pitch/onset/duration group. The augmentation creates new training views; source MIDI and token caches remain immutable.

On three quarters of crops, supply a random two-to-eight-second musical prompt and exclude that prompt from the loss. Measure the horizon from the first sounding note; shorten it at a complete onset-group boundary if necessary to leave at least half the crop's notes as continuation targets. This keeps chords intact and avoids empty target batches. On the other quarter, retain ordinary next-token loss. Augmentation and continuation masking are independent choices, so many crops use both.

The model already used real previous notes as context during autoregressive training. The new mask changes which predictions contribute to the gradient, giving more weight to following an established musical idea. It does not add a theme-memory module, predict whole phrases at once, or replace teacher forcing with generated history. It may help the prompt transition; it does not guarantee coherent long-form structure.

## Run and evaluation protocol

Keep the 36,356,608-parameter architecture, 2,048-token context, corpus and tokenizer. Initialize from `runs/transformer-continued-150m-20261008-001/best.pt`, retaining model weights, Adam state, piece/crop RNG and global RNG. Start a separate augmentation RNG and record its state and statistics in every checkpoint. Exact resume restores both RNG streams and requires identical source/configuration/data. Use BF16, microbatch four, accumulation four, and learning rate 3e-5 decreasing to 1.5e-5 across 10M additional scored targets. A final complete update can exceed that budget slightly.

Raw validation stays unaugmented on the same 536 crops and 1,093,516 targets. Save the best raw-validation checkpoint separately from the final checkpoint. Also evaluate a fixed continuation mask requesting six seconds of supplied context, with the same complete-onset and half-note cap. Compare pitch, onset and duration losses plus conditional next-pitch accuracy. These remain teacher-forced predictive measurements, not listener scores. No test data is used for fitting or selection.

The fresh listening protocol compares the current 150M checkpoint with the final trained candidate, even if the raw-validation best remains the initial weights. Four existing prompts, seeds 44 and 45, eight-second prompts and 12-second new-onset horizons yield 16 declared generations/eight pairs. Keep temperature 0.9, min-p 0.035, remaining-context token budget, renderer and gain matched. Retain every declared attempt and failure. Seeds differ from the previous listener's clips; the prompts overlap, so this is still an informal assessment.

This single screen changes two training methods and adds compute. It cannot isolate augmentation from prompt masking or distinguish their effect from another equal-budget ordinary-training run. The original LSTM/Transformer 100M comparison remains separate.

The predeclared app rule keeps the current 150M model unless raw loss, fixed continuation loss and conditional pitch accuracy all improve, and saved-checkpoint and generation checks pass. A numerical improvement still needs fresh listening to establish audible coherence. See `outputs/transformer-musical-experiment-protocol.json` and `configs/transformer_musical_comparison.json`.

## Reproduction

Run commands from the project root with the existing environment. The source archive and frozen tree were verified byte-for-byte; identities are in `outputs/transformer-musical-source.json`.

```powershell
# Reproduce in a fresh run directory; never overwrite the recorded experiment.
.\.venv\Scripts\python.exe outputs/transformer-musical-code-0cc89abfc60f9acc/scripts/train.py --config configs/transformer_musical.yaml --initialize-from runs/transformer-continued-150m-20261008-001/best.pt --run-dir runs/YOUR_NEW_MUSICAL_RUN

# Recover the recorded experiment if interrupted before its budget completes.
.\.venv\Scripts\python.exe outputs/transformer-musical-code-0cc89abfc60f9acc/scripts/train.py --resume runs/transformer-musical-160m-20261008-001/latest.pt
```

All 63 tests pass, including whole-passage interval/timing preservation, complete-event truncation, quantization collision handling, complete-chord prompt masks, unchanged validation and exact CPU replay of augmentation, masks, dropout, weights and crop sampling. Numerical replay on this deterministic CPU test does not promise identical results on every GPU/software version.

## Results

| Metric | Current 150M checkpoint | Musical fine-tuning final |
| --- | ---: | ---: |
| Raw validation loss | 3.193548 | 3.178622 |
| Fixed continuation validation loss | 3.192593 | 3.177317 |
| Conditional pitch loss | 2.781383 | 2.752962 |
| Conditional pitch top-1 accuracy | 25.31% | 26.00% |
| Piano note/velocity loss | 4.127949 | 4.095417 |
| Onset loss | 2.288979 | 2.282291 |
| Duration loss | 3.215913 | 3.210881 |

The run completed 10,014,758 additional scored targets in 351 updates, reaching 160,021,924 cumulative scored targets. Because prompts are now masked, this total is a lineage accounting measure rather than an equal-objective exposure comparison with the earlier runs. Optimizer computation took 209.4 seconds (3.49 minutes), excluding loading, sampling, validation and writes. Peak allocated GPU memory was 4.16 GiB.

Raw validation loss changed by -0.47%. The best raw checkpoint was at local update 300 with 8,555,999 scored targets; its loss was 3.178238. The listening comparison uses the final trained checkpoint, preserving the complete declared experiment.

Of 5,616 sampled training windows, 2,862 received augmentation and 2,588 moved to another key; 4,257 used continuation masks. The augmentation counted 0 timing-collision fallbacks, 5,099 duration clamps and 246 tail notes removed to fit the context. Continuation prompts masked 826,890 musical tokens, which remained visible as input.

Strict loading reproduced both saved checkpoint losses. The independent diagnostic replayed both parent and final losses within tolerance. Continuation metrics score 985,823 unchanged validation targets; conditional pitch metrics use 361,722 real validation notes. The original checkpoints, comparison suites, earlier listening page and feedback artifacts remain unchanged.

All 16 declared generations succeeded. All 48 generated WAVs have nonzero RMS and no clipped samples. Prompts and note counts were preserved and no note groups merged. Repeated-pitch ending changes remain visible in individual reports.

[Listen to the eight new A/B pairs](http://127.0.0.1:8766/transformer-musical-listening/) before revealing checkpoint names, then download responses. No human ratings of this new candidate have been collected. The numerical results do not establish better harmony, timing stability or long-form coherence in generated music.

The candidate passes all predeclared numerical/integrity criteria for app selection. App activation and local HTTP checks are recorded separately. Browser visual and playback inspection remain unverified due to the earlier browser security restriction.

Evidence: `docs/transformer-musical-training.json`, `outputs/transformer-musical-validation.json`, `outputs/transformer-musical-checkpoint-verification.json`, `outputs/transformer-musical-comparison/suite.json`, `outputs/transformer-musical-source.json` and `outputs/transformer-musical-training-curves.png`.


The app now selects **Transformer · musical fine-tuning**. A live app generation reproduced the declared candidate clip, sampled IDs and all six MIDI/WAV artifacts byte-for-byte. Served bytes and audio ranges passed for the activated candidate; historical 150M job labels were preserved. The new page's 40 audio references served matching bytes and its JavaScript passed syntax checks. These file/HTTP checks do not verify visual layout or playback advancement. Evidence: `outputs/transformer-musical-app-verification.json` and `outputs/transformer-musical-listening-verification.json`.
