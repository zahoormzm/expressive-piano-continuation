# Local application and full-corpus preparation

This report preserves the earlier 6 October app/cache milestone. Subsequent work completed both full-corpus runs, integrated their validated checkpoints, produced 18 fixed samples and passed 43 tests. See [current full-run results](full-training-results.md); the measurements below remain historical evidence.

On **6 October 2026**, the piano continuation pipeline was connected to a working local browser application, and token preparation expanded from the 32-performance pilot to every retained performance. All **39 tests** pass. The longer full-corpus models have not been trained, and human listening assessment remains outstanding.

## Use the application

From the project root:

```powershell
& .\.venv\Scripts\python.exe scripts/serve_app.py
```

Open [Piano continuation studio](http://127.0.0.1:8767/). Select the demo or upload piano MIDI, choose Aria/Transformer/LSTM, and generate. The page shows actual worker stages, offers combined and separate piano audio, and downloads full MIDI, continuation MIDI, WAV, and generation details. It restores the latest job after a reload and distinguishes an earlier result while a new job runs.

Files remain under `outputs/app-jobs/<job-id>/`. Original input bytes use `input.mid`, and generated artifacts use `generation/`. The displayed filename is metadata, never a filesystem path. The application binds to loopback, starts one isolated model process at a time, and serves only allowlisted artifacts from completed jobs. Native audio byte ranges support seeking. A ten-minute process limit and persistent failure messages prevent a stalled job from looking complete.

Accepted inputs are synchronous solo-piano MIDI up to 2 MB, ten minutes, and 10,000 notes; pitch A0–C8, piano-family programs, no drums, and complete positive-duration notes. Prompt and continuation lengths are 1–30 seconds. Models still have bounded token context and a 384-new-token ceiling. Longer/dense inputs can therefore reach a context or generation-token limit. Model input is quantized, sustain becomes note lengths, and other pedal controls are not retained.

## Actual browser checks

The demo path generated an Aria sample with 54 continuation notes and 36 preserved prompt notes. Its audio played with an advancing clock and no reported media error. The full MIDI downloaded through the browser and matched the generated file byte for byte.

The upload path selected the authored `arpeggio.mid` through the browser file chooser, switched to the scratch Transformer, and generated 76 continuation notes. Combined and continuation-only audio played, with the other player pausing when playback switched. Reloading restored the completed result. Missing-file validation displayed a clear error and re-enabled the form; changing back to the demo cleared the error. A final Aria job completed through the revised interface, with no browser console errors. The interface was visually inspected.

These are real model/renderer jobs using the engineering fixture, not a human musical-quality comparison. The native Aria file and scratch model weights are unchanged. The app uses the same shared constrained decoding protocol described in [generation results](generation-results.md).

Automated tests cover invalid numeric settings and uploads, display-name path separation, rejection of non-piano/malformed MIDI before launching a worker, one job at a time, interrupted-job recovery, host/origin restrictions, artifact containment, and HTTP audio ranges. Existing model, data, and generation tests still pass. JavaScript syntax, Ruff lint/format, and dependencies were checked. Detailed evidence is in [app-results.json](app-results.json).

The final application/cache-audit Python source identity is `289f32e05e48cff6d5415f9718ec154df30f94325dac206d91dd4f1b5d8ce1d0`, preserved in [app-source-289f32e05e48cff6.zip](../outputs/app-source-289f32e05e48cff6.zip), including the current HTML and configuration files. Its ZIP SHA256 is `a7b2afa6c0495d81ae827cc7ec39768c8e1dda8e531cbb9004b5924fcf29c37b`. The initial demo/upload jobs used separately preserved source `9a8e7ae705983901633d9e3683a6bc8813ce9a277499626363937cdeeee38715`; the final demo uses the current identity. The current HTML has its own recorded hash because the Python source hash does not include HTML. A local wheel build also verified that the interface HTML is included in the package.

## Complete token cache

| Split | Performances | Tokens | Complete note groups | Crop checks |
| --- | ---: | ---: | ---: | ---: |
| Train | 957 | 17,056,047 | 5,646,553 | 2,871 |
| Validation | 134 | 1,919,535 | 635,135 | 402 |
| Test | 171 | 2,189,119 | 724,905 | 0 |
| Total | 1,262 | 21,164,701 | 7,006,593 | 3,273 |

Preparation completed with no failures and no notes removed during normalization. Each cache passed SHA256 checks, integer/vocabulary checks, complete-event parsing, real-EOS validation, positive velocity/duration, and monotonic timing. Beginning/middle/end crops from every train/validation performance passed the trainer's 2,048-token collation checks. Test tokens were checked for file integrity and event syntax only: no test crops, fitting, or evaluation were run.

The full index is `data/tokens/c8ad763cae79f391/index-fac385b0854fd132-all.jsonl`, SHA256 `b9c6c0b9c4bb287355ba80f2f12104d1c59e2bd6894061c2c1376bfd9f85040f`. Token files occupy 84,820,340 bytes. Metadata groups and exact source hashes retain the previously checked split separation; musical-fingerprint matching remains a separate limitation.

Preparation uses `outputs/tokenization-full-summary.json`. The original `outputs/tokenization-summary.json` remains byte-identical to the cached pilot summary and still resolves the old pilot configurations. Shared per-performance files are immutable, and the full index has its own identity. Evidence: [full-cache-verification.json](../outputs/full-cache-verification.json).

To reproduce preparation and the streaming audit:

```powershell
& .\.venv\Scripts\python.exe scripts/tokenize_data.py --reconstruction-count 0 --summary outputs/tokenization-full-summary.json
& .\.venv\Scripts\python.exe scripts/check_token_cache.py
```

The `--summary` option prevents accidentally replacing the pilot's configuration summary. The original reconstruction examples remain available; this run did not create extra reconstruction examples or claim new listening results.

## Next experiment

Both full-cache YAML configurations resolve successfully. They use the pilot architectures and shared 100-million-valid-target budgets, fresh initialization, the full training split, and four fixed crops per validation performance. Validation and checkpoints occur every 200 completed updates. The full dataset has much more validation work than the pilot, so pilot compute timings are not an accurate wall-time promise for these runs.

Next: record human reconstruction listening, select and run substantive scratch experiments, compare multiple fixed prompts/seeds, and add selected checkpoints to the app. The working demo and full data foundation are ready ahead of the **13 October 2026** submission; final model quality and submission material remain unfinished.
