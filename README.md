# Expressive piano continuation

We are building a local assistant that extends a piano MIDI idea and exports editable MIDI and piano audio. The planned experiments compare an LSTM, a scratch Transformer, and an accessible pretrained Aria model. The submission deadline is **13 October 2026**, as confirmed by the user.

The project implements data validation and reconstruction, resumable LSTM and Transformer training, prompt continuation with scratch checkpoints and pretrained Aria, checked MIDI export, and piano SoundFont rendering. Both original full-corpus scratch runs completed **100,003,399 valid targets**. A separate Transformer continuation reached **150,007,166 cumulative targets**; a further musical-training screen adds **10,014,758 scored targets**, reaching **160,021,924** in its recorded lineage. Musical training uses passage pitch/time augmentation and continuation-focused loss. Fixed validation loss improved from **3.1935 to 3.1786**, and conditional next-pitch accuracy from **25.31% to 26.00%**. The app now selects **Transformer · musical fine-tuning**. Original baselines remain preserved. **18 original comparison samples**, **two sets of eight Transformer A/B pairs** and **12 reconstruction pairs** are available. The first Transformer set received eight informal judgments from one listener; the new musical candidate still needs listening. Final submission material remains outstanding. See [original full-run results](docs/full-training-results.md), [ordinary continuation](docs/transformer-improvement.md), and [musical fine-tuning](docs/transformer-musical-training.md).

Measured foundation results: all **1,276 MIDI files parsed**; the conservative manifest retains **1,262 performances** after 13 metadata-ambiguity exclusions and one note-integrity exclusion. A 32-performance pilot produced 633,382 tokens. Twelve full-performance reconstructions and all 14 targeted tests passed. Full evidence and limitations are recorded in `docs/experiment-log.md`.

## Windows setup

The private repository is [zahoormzm/expressive-piano-continuation](https://github.com/zahoormzm/expressive-piano-continuation). Student submission files are in [submission](submission/README.md): PDF write-up, editable slides, contribution allocation and artifact manifest. A private release carries the checkpoints and listening evidence. Follow [artifact restoration](docs/artifact-handoff.md) after cloning:

```powershell
git clone https://github.com/zahoormzm/expressive-piano-continuation.git
cd expressive-piano-continuation
```

Python 3.11 and Git are required. Use the environment's executable directly; PowerShell activation is unnecessary.

```powershell
py -3.11 -m venv .venv
$pythonExe = '.\.venv\Scripts\python.exe'
& $pythonExe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
& $pythonExe -m pip install -r requirements-lock.txt
& $pythonExe -m pip install -e . --no-deps
& $pythonExe scripts/verify_environment.py
```

This project environment has passed FP32 and BF16 forward/backward checks on the RTX 5070, using PyTorch 2.14.1+cu130 and CUDA 13.0. The CUDA build is supplied with the PyTorch wheel. The complete local hardware and dependency snapshot is saved in `outputs/environment.json`.

If a sandbox blocks loading a PyTorch DLL, run the verification from a normal local terminal. The successful check did not require changing Windows application-control settings.

## Data and reconstruction

```powershell
& $pythonExe scripts/download_data.py
& $pythonExe scripts/validate_data.py --dataset data/maestro-v3.0.0
& $pythonExe scripts/prepare_splits.py
& $pythonExe scripts/tokenize_data.py --limit 32
```

The downloader uses only the official MIDI archive, verifies its published SHA256, and preserves the original files. The validator audits every performance and retains the official train, validation, and test assignments. It writes:

- `data/manifests/maestro-v3.jsonl`: immutable metadata, hashes, MIDI statistics, warnings, and exclusions.
- `outputs/validation/maestro-v3.json`: counts, parse failures, split-integrity checks, and the manifest identity.

The original metadata contains five canonical title groups spanning official splits. Some titles name multiple movements or song transcriptions, so the metadata alone does not establish actual musical overlap. The original audit deliberately returns a nonzero exit code for these unresolved groups. Inspect its report, then run `prepare_splits.py` to create `data/manifests/maestro-v3-clean.jsonl` with a documented conservative exclusion policy. It excludes every recording in ambiguous groups, preserves all original split labels, and writes `outputs/validation/maestro-v3-clean.json`. Tokenization uses this clean manifest by default and refuses retained groups crossing splits.

The `--limit 32` command is a clearly labelled tokenization pilot sampled across the ordered corpus. For all usable performances, keep its summary separate from the pilot:

```powershell
& $pythonExe scripts/tokenize_data.py --reconstruction-count 0 --summary outputs/tokenization-full-summary.json
& $pythonExe scripts/check_token_cache.py
```

The completed full cache contains **21,164,701 tokens** across 957 training, 134 validation, and 171 test performances. Every cache passed hash/event-syntax checks, and 3,273 train/validation crops passed the trainer's format checks. No notes were removed during normalization. Test files were checked for integrity and syntax only; no test crops or model evaluation were run. The original 32-performance summary remains byte-identical to its stored pilot artifact. Each cache contains one complete performance; the trainer samples complete-event windows from these caches. See [application and full-cache results](docs/app-results.md).

Tokenization writes a versioned cache under `data/tokens/`, the full vocabulary/configuration identity, and `outputs/tokenization-summary.json`. Up to 12 training performances across years are reconstructed into `outputs/reconstruction/`. It checks quantized notes and saved MIDI by reading the result back from disk.

To reconstruct a supplied file:

```powershell
& $pythonExe scripts/reconstruct.py --midi prompts/example.mid --output outputs/reconstruction/custom
& $pythonExe scripts/render_preview.py --midi outputs/reconstruction/custom/reconstructed.mid --wav outputs/reconstruction/custom/preview.wav --seconds 20
```

Supply a real MIDI file for `prompts/example.mid`. Preview audio uses harmonic diagnostic synthesis and is useful for timing checks. The generation samples below use a separate FluidSynth piano renderer for listening comparisons.

## Representation and integrity limits

The tokenizer is pinned to [EleutherAI/aria-utils revision 7c815755](https://github.com/EleutherAI/aria-utils/tree/7c815755c6889a784b6a1b3343e9e8daa3e7d128). We keep its complete published vocabulary of **17,727 tokens**. Its absolute representation uses 5-second segments, 10 ms timing steps, and velocity quantization. Note durations above 5 seconds are clipped and counted in reconstruction reports.

The scratch preprocessing path preserves leading silence, disables randomized diminish tokens, fixes prefix ordering, and normalizes sustain and repeated-pitch overlaps using the toolkit. It records changed or merged notes. CC64 is consumed into note durations and is removed before rendering to prevent applying sustain twice. CC66 sostenuto and CC67 una corda are reported in the original audit but are not preserved by this representation. The wrapper supplies MIDI's default initial tempo if the first explicit tempo change occurs after tick zero.

Quantization can create additional same-pitch collisions during MIDI export. Export changes are counted separately from tokenizer reconstruction. Generation forbids duplicate pitches at a single onset and refuses exports that merge note groups. The exporter can shorten repeated-pitch note endings: the full sample audit records unchanged pitch, velocity, onset and counts, and separately measures ending changes (up to 1,400 ms in this suite). Original/reconstructed audio still needs musical inspection; the user authorized these full runs using the completed automated checks. The pretrained adapter passed vocabulary/special-ID, strict weight-load, and native prompt-token agreement checks on the authored demonstration prompt. That agreement does not establish identical preprocessing for all inputs; leading silence, sustain normalization, simultaneous-note ordering and boundary clipping remain explicit choices.

Composition checks use normalized canonical metadata and exact MIDI hashes. They do not yet perform musical-fingerprint matching across differently named works. Exclusions remain visible, and raw files are never repaired in place. Reusing a manifest path with changed content fails instead of silently changing an experiment's data.

## Checks

```powershell
& $pythonExe -m pytest -q --basetemp outputs/test-temp
& $pythonExe -m ruff check src scripts tests
```

The test temporary directory stays within `outputs/`. The **63 passing tests** cover data integrity, reconstruction, training windows and loss masks, causality, accumulation, constrained sampling, complete note groups, horizon stopping, prompt preservation, repeated-pitch rejection, LSTM inference-cache agreement, upload validation, job isolation, request boundaries, restart handling, audio ranges, cross-process GPU locking, pedal excerpts, model-label preservation across registry changes, automatic context budgeting and historical duration reporting. Continuation tests additionally cover parent compatibility, optimizer/sampler/RNG recovery, parent immutability and cumulative training accounting. Musical-training tests check global pitch/time transformations, complete-chord prompt masking, unchanged validation and exact replay of augmentation, masks and weights.

## Scratch training pilots

```powershell
& $pythonExe scripts/check_training.py
& $pythonExe scripts/benchmark.py --config configs/lstm.yaml
& $pythonExe scripts/benchmark.py --config configs/transformer_small.yaml
& $pythonExe scripts/train.py --config configs/lstm.yaml
& $pythonExe scripts/train.py --config configs/transformer_small.yaml
```

The screening configurations use a 2,048-token context, BF16, microbatch one, accumulation 16, and five million valid target tokens per model. A final complete optimizer update may slightly exceed the budget; the exact total is recorded. Models use the same vocabulary, source manifest, fixed validation crops, and loss definition. Inputs and targets are shifted once. Padding and the supplied instrument/BOS prefix are excluded from loss. Every crop starts and ends on complete note groups, retains its piece identity, and rebases five-second segments. It does not carry notes that began before the crop.

The baseline uses one 256-unit LSTM layer and separate embedding/output weights. The Transformer uses eight layers, width 512, eight attention heads, SwiGLU width 1,536, pre-LayerNorm, rotary positions, and tied embedding/output weights. Parameter counts are measured after construction. Each LSTM call begins with fresh hidden state; the Transformer combines causal and padding masks.

The default configurations use the **32-performance token pilot**: 25 training performances and two validation performances. They do not load the five test performances. This limited subset supports an engineering screen; numerical losses are not a final full-corpus comparison or a measure of audible quality. The full-corpus configurations below use the separately verified cache of all retained performances.

`check_training.py` uses smaller diagnostic architectures at FP32 to verify tiny-data memorization and interrupted-versus-continuous replay with dropout enabled. Both models passed on the RTX 5070, with zero observed weight/loss difference after recovery. The main BF16 runs may have numerical reproducibility limits; exact replay was established for the recorded deterministic FP32 diagnostic, on this device and software version.

The benchmark performs 20 warmup microbatches and 100 measured microbatches, including actual crop sampling, forward/backward, clipping, and optimizer updates. It records valid-token throughput, peak allocated/reserved VRAM, architecture, data hashes, and configuration. Its runtime extrapolation excludes validation and checkpoint writes.

Both 6 October pilots completed **5,030,045 valid target tokens over 163 updates**. Fixed validation loss fell from 9.785 to **6.762** for the LSTM and from 9.880 to **6.076** for the Transformer. The throughput benchmarks measured 22,584 and 45,411 valid tokens/second, with peak GPU allocations of 0.63 and 1.57 GiB respectively. All four best/latest checkpoint files loaded and reproduced their recorded losses. See [measured training results](docs/scratch-training-results.md) for the comparison, exact run paths, chart, and limitations.

Each training run creates `resolved-config.json`, `identity.json`, hardware/dependency metadata, `metrics.jsonl`, `summary.json`, and atomic `latest.pt`/`best.pt` checkpoints under its unique `runs/` directory. Checkpoints contain weights, optimizer state, token-based cosine/warmup progress, crop-sampler state, and Python/NumPy/PyTorch/CUDA random states. Biases and normalization parameters do not receive weight decay. Checkpoints and validation are scheduled at completed update boundaries.

```powershell
# Replace SELECTED_RUN with an actual directory printed by the trainer.
& $pythonExe scripts/train.py --resume runs/SELECTED_RUN/latest.pt
& $pythonExe scripts/plot_training.py --runs runs/LSTM_RUN runs/TRANSFORMER_RUN
```

Resume requires the same saved configuration, source code, data/tokenizer hashes, and validation-window identity. If a process is interrupted inside an optimizer update, resume the last complete atomic checkpoint; partial gradients are not saved. `--stop-after-updates N` permits a controlled stop at absolute update N. A completed budget is not automatically extended by resume.

To continue compatible weights in a new experiment, supply `--config NEW_CONFIG --initialize-from PARENT_CHECKPOINT --run-dir NEW_DIRECTORY`. Architecture, data, tokenizer, validation windows and optimizer settings must agree. Weights, Adam moments, crop position and RNG states continue; the new local learning-rate schedule and counters start at zero. Parent hashes and cumulative training totals remain recorded. See [the completed 50M-target continuation](docs/transformer-improvement.md) for its frozen source and exact commands.

The repository does not yet have a Git commit. Runs record a hash of the Python source tree and explicitly record the missing commit identity. Checkpoints load with PyTorch's restricted `weights_only=True` path.

The source for the recorded pilots is preserved in `outputs/training-source-2be60a28e4f5aaa7.zip`; its contents have been checked against the recorded source identity. The current source includes subsequent generation work, so use the isolated original source when strictly resuming these two pilots. Its already-extracted checkout is `outputs/training-code-2be60a28e4f5aaa7`. From the main project directory:

```powershell
& $pythonExe outputs/training-code-2be60a28e4f5aaa7/scripts/train.py --resume runs/lstm-pilot-20261006-001/latest.pt
& $pythonExe outputs/training-code-2be60a28e4f5aaa7/scripts/train.py --resume runs/transformer-small-pilot-20261006-001/latest.pt
```

If moving the project, extract the ZIP to that same clean directory with `Expand-Archive`; keep the current working directory at the main project root so dataset/run paths resolve. Do not mix newer source files into the frozen checkout. Both checkpoints passed strict loading and reproduced validation loss from this isolated source (`outputs/resume-with-frozen-source.json`). These pilots have already reached their configured budgets; resume checks the completed run rather than extending it.

Code, project notes, and the dependency lock belong in Git. Data, model weights, outputs, the virtual environment, and downloaded source checkouts are ignored. The repository has not been published or shared.

## Local MIDI upload and generation application

```powershell
& $pythonExe scripts/serve_app.py
```

Open [Piano continuation studio](http://127.0.0.1:8767/). Choose **Try the demo** or **Upload MIDI**, select a model, set prompt/continuation lengths, and click **Generate continuation**. The app displays loading, generation, and rendering stages, then provides combined and separate audio players, full/continuation MIDI, WAV downloads, and generation details. The latest job survives a page reload. Models with missing files are marked unavailable.

The app accepts synchronous solo-piano `.mid`/`.midi` files up to 2 MB, ten minutes, and 10,000 notes, within A0–C8 and without drums or unfinished notes. Piano-family voices render on acoustic grand piano. Prompt and continuation limits are 1–30 seconds; the default is 8 + 12 seconds. Variation settings expose seed, temperature, and min-p. App generation now uses the context left after the actual prompt, rather than a fixed 384-new-token ceiling: LSTM/Transformer have 2,048 total tokens and Aria has 8,192. The selected duration is a maximum new-onset horizon; the model can predict an ending or exhaust context earlier. A shorter prompt leaves more room. The result reports actual continuation audio length and why generation stopped, with note releases and renderer tails distinguished from the horizon. Reported sampling time excludes loading and rendering. See [the duration fix and measured replays](docs/duration-budget-fix.md).

The CLI retains its explicit 384-token default for reproducible fixed comparisons. Pass `--max-new-tokens auto` to use remaining context, or an integer to keep a declared budget. The original 18-sample suite and its source snapshot remain preserved.

Uploads and job records stay under `outputs/app-jobs/<job-id>/`; generated files are under its `generation/` directory. Original uploads are preserved as `input.mid`, with their display filenames stored as metadata. One isolated GPU worker runs at a time, and a ten-minute worker limit reports a failure if exceeded. Artifacts are offered only after successful generation and integrity checks. A stopped server marks interrupted jobs as failed on the next lookup. Restart with the same command; Ctrl+C stops the server. `--port` selects another local port.

This server binds only to `127.0.0.1` and requires no external account or Python web framework. `configs/models.json` selects the validated full-corpus LSTM/Transformer best checkpoints and the pretrained baseline. New jobs preserve their selected label/checkpoint; historical pilot jobs retain their original labels. Missing files are marked unavailable. Project training, generation, benchmarks and checkpoint validation share a cooperative GPU lock. The fixed comparison batch reserves that lock throughout all attempts; the app disables generation while it is busy. This does not control unrelated GPU applications.

## Generate MIDI and piano audio

The existing demonstration uses the same eight-second, 36-note authored arpeggio for every model. Each model generates new note onsets for up to 12 seconds, with seed 42, temperature 0.98, min-p 0.035, and a 384-token ceiling. A shared grammar preserves complete notes, monotonic onsets, piano pitch range, positive velocity/duration, and bounded simultaneous-note density. This is the project's comparison sampler; it differs from the full upstream Aria sampler. Note releases can extend beyond the onset horizon. Model-predicted EOS and export-only EOS are recorded separately.

```powershell
# Verify/download the piano renderer and SoundFont; --pretrained also downloads
# the 2.63 GB generation checkpoint and pins the official Aria source checkout.
& $pythonExe scripts/setup_assets.py --pretrained
& $pythonExe scripts/verify_pretrained.py --prompt outputs/prompts/arpeggio.mid

# These output directories must be new. Supply your own solo-piano MIDI instead
# of the demonstration prompt to try another idea.
& $pythonExe scripts/generate.py --checkpoint runs/lstm-full-20261006-001/best.pt --prompt outputs/prompts/arpeggio.mid --output outputs/generation/lstm-custom-001 --render
& $pythonExe scripts/generate.py --checkpoint runs/transformer-small-full-20261006-001/best.pt --prompt outputs/prompts/arpeggio.mid --output outputs/generation/transformer-custom-001 --render
& $pythonExe scripts/generate.py --backend aria --prompt outputs/prompts/arpeggio.mid --output outputs/generation/aria-custom-001 --render
```

For a fresh workspace, `scripts/make_demo_prompt.py` creates the authored prompt at `outputs/prompts/arpeggio.mid`; it refuses to replace an existing file. The pretrained file is the public `model-gen.safetensors` in `loubb/aria-medium-base`, pinned by repository revision and SHA256 in `configs/assets.json`. Aria's 658,538,496 parameters load strictly through its pinned native PyTorch implementation. No adaptation is applied; FP32 checkpoint tensors are cast to BF16 for inference. Native cached-versus-prefill logits pass the recorded BF16 tolerance check. The original checkpoint file is preserved.

Each generation folder contains `sampled-tokens.npy`, `prompt.mid`, `combined.mid`, `continuation.mid`, and `report.json`. `--render` adds all three matching WAVs and renderer reports. The continuation MIDI starts at time zero relative to the prompt boundary. Reports include source/checkpoint/tokenizer identities, settings, stopping reason, prompt preservation, independently parsed MIDI note counts, GPU allocation, timings, and audio integrity checks. Inputs that exceed model context are rejected rather than silently truncated.

Playback uses checksum-pinned FluidSynth 2.6.1 and GeneralUser GS 2.0.3 acoustic grand piano, 44.1 kHz stereo PCM16, fixed gain 0.25, polyphony 256, and no chorus/reverb. There is no per-sample loudness normalization. Renderer release tails add audio time after the last note. To render an existing MIDI separately:

```powershell
& $pythonExe scripts/render_piano.py --midi outputs/generation/aria-arpeggio-002/combined.mid --wav outputs/aria-piano-preview.wav
```

The earlier pilot samples remain under `outputs/generation/`; see [their generation results](docs/generation-results.md). The full comparison suite is under `outputs/full-generation/`, with three authored prompts, seeds 42/43 and all three models. All 18 attempts succeeded with preserved prompts, zero merged notes, and non-silent/unclipped piano audio. Four scratch samples stopped at the 384-token ceiling before the onset horizon. Changed note endings are recorded separately. These are engineering samples with no human musical-quality ranking. See [full-run results](docs/full-training-results.md).

## Local listening page

```powershell
& $pythonExe scripts/build_listening_page.py --suite outputs/full-generation/suite.json --output outputs/full-listening/index.html
& $pythonExe -m http.server 8766 --bind 127.0.0.1 --directory outputs
```

Open [full-model comparisons](http://127.0.0.1:8766/full-listening/) while the server is running. Each prompt/seed group plays the shared prompt and all three continuations, shows the eight-second boundary on piano rolls, pauses other players when one starts, and links MIDI/reports. Cards show actual checkpoint tokens, stopping reasons and export-change counts. The page also works from `outputs/full-listening/index.html`. The [earlier pilot page](http://127.0.0.1:8766/listening/) remains available.

[Reconstruction listening](http://127.0.0.1:8766/reconstruction-listening/) contains 12 source/reconstruction pairs using the same piano and fixed gain. All 24 WAVs passed automated checks; human timing/sustain assessment is pending. See [listening notes](docs/reconstruction-listening.md). `scripts/make_evaluation_prompts.py` creates the two additional authored prompts in a fresh workspace, and `scripts/prepare_evaluation_prompts.py` verifies and renders the declared inputs. Both refuse to overwrite their output artifacts.

[Transformer A/B listening](http://127.0.0.1:8766/transformer-listening/) compares the original and continued checkpoints in eight matched pairs. Listen before revealing their names, judge melody and coherence, and download the responses. This uses its own declared settings and preserves the original three-model comparison. See [the continuation experiment](docs/transformer-improvement.md).

## Full-corpus experiment configurations

`configs/lstm_full.yaml` and `configs/transformer_full.yaml` use the separate full cache, the measured architectures, a 2,048-token context, BF16, microbatch four, accumulation four, and **100 million valid targets per model**. They start fresh rather than extending the pilots. Validation uses four fixed crops from each of 134 validation performances; validation/checkpoints occur every 200 completed updates.

Both runs completed **100,003,399 valid targets in 3,239 updates**, with identical sampled token trajectories. Fixed validation loss fell from 9.7861 to **4.3161** for LSTM and from 9.8901 to **3.2758** for Transformer. All best/latest checkpoints reproduced their losses with zero difference. These validation crops differ from the earlier small pilots, so their numerical losses should not be compared directly. Training uses 957 performances; test data remains excluded from fitting, selection and model evaluation.

Strict replay uses the preserved training source, from the main project directory:

```powershell
& $pythonExe outputs/full-training-code-225d8b0cb206711f/scripts/verify_checkpoints.py --runs runs/lstm-full-20261006-001 runs/transformer-small-full-20261006-001
& $pythonExe scripts/plot_training.py --runs runs/lstm-full-20261006-001 runs/transformer-small-full-20261006-001 --output outputs/full-training-curves.png
```

The exact training ZIP/directory and the separate final app/generation ZIP/directory are identified in `outputs/full-training-source.json` and `outputs/full-generation-source.json`. Keep their source versions separate. Full runs have completed their fixed budgets; strict resume validates the saved state rather than extending training. A fresh experiment needs a new run directory and its own source snapshot. The fixed suite runner is `scripts/generate_suite.py`; use a fresh `--output` directory and retain every declared attempt. Full evidence is in [the measured report](docs/full-training-results.md) and its JSON companion.

## Next milestones

1. Record human reconstruction and continuation listening judgments.
2. Use that evidence to decide on further training, sampler changes or optional adaptation.
3. Rehearse with [the demo guide](docs/demo-guide.md).
4. Finish the write-up, presentation and repository/artifact handoff before 13 October.

See `docs/development-plan.md` for the revised schedule and `docs/experiment-log.md` for measured progress.

## Sources and attribution

[MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) is supplied by Google and the International Piano-e-Competition under CC BY-NC-SA 4.0. Retain attribution and observe the dataset's terms. Cite Hawthorne and colleagues, *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*, ICLR 2019.

[Aria utilities](https://github.com/EleutherAI/aria-utils) use Apache-2.0. The pretrained baseline uses [Aria](https://github.com/EleutherAI/aria), by Bradshaw and colleagues, *Scaling Self-Supervised Representation Learning for Symbolic Piano Performance*, ISMIR 2025, and the author's [generation checkpoint](https://huggingface.co/loubb/aria-medium-base). [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.1) uses LGPL-2.1-or-later. [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS) is by S. Christian Collins and uses its own license, preserved at `data/assets/GeneralUser-GS-LICENSE.txt`. Dataset, checkpoint, asset, and project-code licensing are separate; no project-code license has been selected yet.

Installation was checked against the [official PyTorch documentation](https://pytorch.org/get-started/locally/), and the installed wheel was validated by actual GPU execution.

