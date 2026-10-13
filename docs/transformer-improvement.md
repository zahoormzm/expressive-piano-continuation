# Transformer continuation experiment · 8 October 2026

Status: the 50M-target continuation and numerical/artifact verification are complete. The user's priority is melody and musical coherence. Numerical and listening results will be recorded separately.

The original Transformer was still improving at the end of its 100M-target budget. Its fixed validation loss fell from 3.320636 at 80.31M targets to 3.275785 at 100.00M. Continued training is therefore the first experiment. Training compute affects next-token prediction in [language-model scaling studies](https://arxiv.org/abs/2001.08361); applying that idea to this musical model is an inference supported by its own learning curve, not a prediction of audible quality.

## Declared experiment

Continue `runs/transformer-small-full-20261006-001/best.pt` in the new directory `runs/transformer-continued-150m-20261008-001`. Preserve the original checkpoint and its fixed 18-sample comparison suite. The parent checkpoint SHA256 is `60fe63531cf34107dcd2b3ea91b6aed5260a45117883fc23c58ebce910799053`.

The 36,356,608-parameter architecture, 2,048-token context, tokenizer, 957 training performances, 134 validation performances and 536 validation crops stay identical. Validation contains 1,093,516 valid prediction targets. Test data is excluded from fitting and selection.

`configs/transformer_continue.yaml` allows 50,000,000 **additional** valid targets. Adam moments, crop-sampler position and random states continue from the parent. The local counters and cosine learning-rate schedule restart: learning rate 0.00006 declining to 0.00003, no warmup. BF16, microbatch 4, accumulation 4, weight decay 0.1 and gradient clipping 1 remain fixed. The initial validation replay was exactly 3.2757848754184895.

This is a new training experiment with explicit parent lineage. Exact resume continues to require the same source, configuration and data identities. Every new checkpoint records the parent file hash, source/data/configuration identities and prior training totals. Local and cumulative targets are reported separately. A complete optimizer update can slightly exceed the additional budget.

## Selection and listening protocol

Save the checkpoint with the lowest loss on the existing fixed validation crops. A checkpoint must improve the parent's loss, pass strict saved-checkpoint replay and produce checked MIDI/audio before it can replace the app's Transformer selection. This is numerical selection; musical preference still needs the user's listening judgments. The original LSTM/Transformer 100M-budget comparison remains a separate matched-budget experiment.

`configs/transformer_comparison.json` declares all 16 before/after attempts: four prompts, two seeds and both selected checkpoints. Three prompts are authored; Für Elise is the prepared informal user input, not a held-out generalization test. Each pair uses an eight-second prompt, 12-second new-onset horizon, temperature 0.9, min-p 0.035 and the same remaining-context token budget. All attempts and any failures are retained. The fixed piano sound and gain are shared.

The local A/B page varies presentation order reproducibly by prompt/seed, hides checkpoint names initially, plays the shared prompt and both continuations, and permits downloading listening responses. This is a small informal assessment. No listener responses are invented. Responses indicate whether identities had been revealed when the judgment was recorded.

Validation diagnostics also inspect piano-note, onset and duration token loss. A conditional pitch diagnostic marginalizes velocity and predicts the next pitch from real previous notes. This measures local teacher-forced prediction; it does not measure long-range melody or an independently generated performance's coherence.

## Reproduction

From the main project root:

```powershell
# A fresh run directory is required for initialization.
.\.venv\Scripts\python.exe outputs/transformer-continue-code-89807c47f601b2fe/scripts/train.py --config configs/transformer_continue.yaml --initialize-from runs/transformer-small-full-20261006-001/best.pt --run-dir runs/YOUR_NEW_RUN

# Resume this recorded run from its own frozen training source.
.\.venv\Scripts\python.exe outputs/transformer-continue-code-89807c47f601b2fe/scripts/train.py --resume runs/transformer-continued-150m-20261008-001/latest.pt

# Use the separately frozen comparison source and fresh output directories.
.\.venv\Scripts\python.exe outputs/transformer-comparison-code-6f1a2820366e1519/scripts/generate_suite.py --config configs/transformer_comparison.json --output outputs/YOUR_NEW_COMPARISON
.\.venv\Scripts\python.exe scripts/build_transformer_comparison.py --suite outputs/YOUR_NEW_COMPARISON/suite.json --output outputs/YOUR_NEW_LISTENING/index.html
```

The training source is `89807c47f601b2fe04cf591f02cdcba771c077bf298df3828aaae69d3d54e1b9`; the comparison/diagnostic source is `6f1a2820366e151984d00850a72178c9100d15205f84b81538de3b6d2845abc6`. Archive hashes and locations are in `outputs/transformer-continue-source.json` and `outputs/transformer-comparison-source.json`. All 53 tests passed, including compatible continuation, incompatible-parent rejection, parent immutability, cumulative accounting and exact CPU recovery with dropout.

## Results

| Metric | Original checkpoint | Continued checkpoint |
| --- | ---: | ---: |
| Fixed validation loss (nats; lower is better) | 3.275785 | 3.193548 |
| Cumulative valid targets at the selected checkpoint | 100,003,399 | 150,007,166 |
| Conditional next-pitch loss | 2.915464 | 2.781383 |
| Conditional next-pitch top-1 accuracy | 21.13% | 25.31% |
| Piano note / velocity token loss | 4.274001 | 4.127949 |
| Onset token loss | 2.344726 | 2.288979 |
| Duration token loss | 3.261899 | 3.215913 |

Validation loss decreased by 2.51%; perplexity decreased by 7.89%. Perplexity is exp(mean loss), a measure of next-token uncertainty. These are predictive metrics, not listener scores. Conditional pitch accuracy is measured on 361,722 real validation notes.

The run completed 50,003,767 additional targets in 1,620 updates, for 150,007,166 cumulative targets. Optimizer computation took 1673.4 seconds (27.89 minutes), excluding initialization, validation and checkpoint writes. Peak allocated GPU memory was 4.16 GiB.

Best and latest checkpoints passed strict loading from the frozen training source and fixed-validation replay. The independent category diagnostic also reproduced both checkpoint losses within tolerance. All 16 declared generations succeeded; all 48 WAVs were nonzero and unclipped, prompts/counts were preserved and no note groups were merged. Changes to repeated-pitch endings remain recorded in individual reports. The original checkpoint hash remains unchanged.

[Open the A/B listening page](http://127.0.0.1:8766/transformer-listening/). Listen before revealing names, choose which continuation follows the melody more coherently, and download the responses. No human judgments have been collected. The eight pairs are a small informal musical assessment. Browser visual inspection was blocked by browser security policy; numerical, file and HTTP checks are recorded separately.

Evidence: `outputs/transformer-continued-checkpoint-verification.json`, `outputs/transformer-comparison-validation.json`, `outputs/transformer-comparison/suite.json`, `outputs/transformer-source-verification.json`, `outputs/transformer-improvement-curves.png`, and `docs/transformer-improvement.json`.

The running [studio](http://127.0.0.1:8767/) now offers **Transformer · continued training**. A live demo request with the A/B protocol settings reproduced the candidate's sampled IDs and all six MIDI/WAV files byte-for-byte. Served audio bytes and partial audio requests passed; historical labels and baseline artifacts were preserved. The 40 listening-page players' served files, HTML structure and JavaScript syntax passed checks. This does not establish visual browser layout or playback advancement: automated browser inspection was rejected by browser security policy. App and listening checks are in `outputs/transformer-app-verification.json` and `outputs/transformer-listening-verification.json`.

## Subsequent listening feedback

The preceding status records completion of the 150M experiment. Later on 8 October, the user supplied eight informal judgments: five preferences for the continued model, two for the original, one unclear. One of the five continued preferences was recorded after revealing checkpoint names. The seven hidden-name judgments were four continued, two original, one unclear. The attachment's suite hash and checkpoint mappings were verified; the preserved raw responses and summary are indexed by `outputs/listening-feedback/transformer-feedback-summary.json`.

Comments identify unstable timing and weak harmony. One listener's eight pairs support a follow-up hypothesis, not a general preference claim. The user selected [musical augmentation and continuation-focused training](transformer-musical-training.md) for a separate experiment. The earlier numerical results, checkpoints and generated suite remain preserved.

