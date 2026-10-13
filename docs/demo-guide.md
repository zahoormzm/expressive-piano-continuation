# Local demonstration guide

The full-run and application checks have passed; this guide prepares the human rehearsal. The submission deadline is 13 October 2026. A review slot and deadline time have not been supplied.

## Start the local tools

From the project root, run the app and static artifact server in separate terminals:

```powershell
.\.venv\Scripts\python.exe scripts/serve_app.py
.\.venv\Scripts\python.exe -m http.server 8766 --bind 127.0.0.1 --directory outputs
```

Open [the studio](http://127.0.0.1:8767/), [full-model comparisons](http://127.0.0.1:8766/full-listening/), and [reconstruction pairs](http://127.0.0.1:8766/reconstruction-listening/). Confirm the studio reports readiness and the full-corpus model labels appear. Training and project GPU jobs share one lock; finish them before a live generation demo. Listen to the selected audio and set the computer's volume before presenting.

The studio's Transformer selection is **Transformer · musical fine-tuning**. The app now selects **Transformer · musical fine-tuning**. The musical candidate's fixed validation loss is 3.1786, versus 3.1935 for the previous 150M checkpoint. The original 18-sample page retains the matched 100M-budget baselines. Use [the musical fine-tuning A/B page](http://127.0.0.1:8766/transformer-musical-listening/) to assess melody, harmony and timing before revealing checkpoint names. Its eight pairs and downloadable responses are informal; predictive metrics alone do not establish an audible winner. The [earlier Transformer comparison](http://127.0.0.1:8766/transformer-listening/) remains preserved. See [the training method and measured results](transformer-musical-training.md).

## A four-minute walkthrough

1. **Starting idea (30 seconds).** Explain that the application continues solo-piano MIDI and produces editable MIDI plus piano audio. It models note events, timing and velocity. The audio comes from a fixed piano renderer.
2. **Live generation (60 seconds).** Keep the demo at an eight-second prompt and 12-second continuation, select pretrained Aria, and generate with seed 42. Show the real loading/generating/rendering stages. Play from the beginning and point out the eight-second prompt boundary. Display the separate continuation and MIDI/WAV download controls.
3. **A different idea (45 seconds).** Choose Upload MIDI and select the authored `outputs/prompts/melody.mid`. Select the full-corpus Transformer or LSTM and generate. Explain that seed/temperature/min-p control variation and that a dense or long prompt can exceed context. Check the resulting prompt-preservation message.
4. **Measured comparison (60 seconds).** Open the static comparison page. It contains three fixed authored prompts, two seeds and all three models; any failed attempts remain recorded. Play matching samples from one prompt/seed group. Show [the full-run loss chart](../outputs/full-training-curves.png) and [the measured report](full-training-results.md). Explain that validation uses identical held-out crops, and that the Transformer and LSTM have different parameter counts.
5. **Limits and next work (45 seconds).** Show a reconstruction pair. Explain 10 ms timing, quantized velocity, the five-second duration limit, and pedal handling. State the actual status of human listening assessment. Aria uses unmodified upstream generation weights; no project LoRA result has been produced. Final musical-quality judgments require listening evidence.

Avoid describing a numerical loss result as an audible winner. Keep optimizer computation time distinct from total waiting time, and sampling time distinct from model loading and audio rendering. The generated onset horizon can end before the audio release tail. Shared decoding constraints enforce valid note structure; they do not demonstrate unconstrained model syntax accuracy.

The app now labels the requested duration as a continuation limit and uses the model context left after the prompt. An ending token or context exhaustion can still shorten the result. The page reports actual continuation audio duration and the stop reason; audio includes note releases and renderer tails. The earlier fixed comparison suite retains its declared 384-token ceiling. See [the measured duration-budget fix](duration-budget-fix.md).

## Rehearsal checks

- Confirm the local model files, renderer binaries and SoundFont exist; no model download should be needed during the demo.
- Generate once with the exact prompt/model/settings you intend to show, then listen to the result. Keep the fixed comparison page as a reproducible fallback if live generation encounters an error.
- Try the actual uploaded piano MIDI before the review. The app accepts synchronous piano files up to 2 MB, ten minutes and 10,000 notes, in A0–C8, without drums or unfinished notes.
- Open the exported MIDI in an editor if one is available and verify the prompt/continuation boundary is understandable to the audience.
- Record any listening problems and the sample IDs/settings. Do not replace an unfavorable fixed-protocol result with an unrecorded chosen sample.

## Submission material still to prepare

The code, README, dependency lock, experiment configs and reports belong in the repository. Data, checkpoints, generated artifacts and frozen source snapshots are local ignored files. Preserve the needed artifacts separately when sharing or moving the project, and retain dataset/checkpoint/SoundFont attribution. No remote repository or project-code license has been selected yet.

The final write-up and slides should cover the problem, retained-data policy, event representation, model architectures, matched training protocol, measured losses/throughput, controlled sample protocol, actual human listening findings, limitations and a demo link or recording. This guide is preparation for a rehearsal; it is not evidence that a human rehearsal or submission has happened.
