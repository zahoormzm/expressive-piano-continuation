# Live continuation duration — 6 October 2026

A 30-second prompt and requested 30-second Transformer continuation produced only 11.6 seconds of continuation audio. The job `6970d1d06c6b49f1b40a088cfbb63a98` used 694 prompt tokens and stopped after 383 new tokens, with 127 notes ending 8.35 seconds after the prompt. Its app command inherited the CLI's fixed 384-token default, although the model had 1,354 context tokens available after that prompt.

The app now explicitly passes `--max-new-tokens auto`. Generation allocates the selected model's remaining context after the actual encoded prompt. LSTM/Transformer retain a 2,048-token total context and pretrained Aria retains 8,192. A prompt must leave at least one complete three-token note. Explicit integer budgets still reject overflow, and the CLI default remains 384. The frozen training runs and original 18-sample comparison protocol have not changed.

The interface calls the requested time a **continuation limit**, displays actual continuation audio length, and explains whether generation exhausted context, predicted an ending, or reached the new-onset horizon. Historical fixed-budget results receive duration information from their existing reports without rewriting stored jobs or changing their model labels. Note-release lengths and piano renderer tails can extend the audio past the onset horizon; no silence was padded to simulate additional generation.

| Input / identical settings before and after | Prompt tokens | Old audio | Updated audio | Updated final note end | Updated stop |
| --- | ---: | ---: | ---: | ---: | --- |
| Uploaded sonata; 30 + 30 seconds, seed 44, temperature 0.96, min-p 0.035 | 694 | 11.60 s | 28.20 s | 25.00 s | Context token ceiling |
| Prepared Für Elise; same durations and variation settings | 529 | 12.96 s | 33.11 s | 30.07 s | New-onset horizon |

The updated sonata job is `3abde6e934534d639ab0b344cfb63315`: 449 new notes, 1,352 sampled tokens within its 1,354-token budget, and 13.50 seconds of sampling. The updated Für Elise job is `070622f1f67648728d5dad0c022abb89`: 346 new notes, 1,045 sampled tokens within its 1,519-token budget, and 10.81 seconds of sampling. The latter result appeared in the app after the fix and is the result shown in the saved screenshot.

Both pairs have byte-identical input MIDI and identical settings/checkpoint hashes. Each new sampled sequence begins with the complete older sampled sequence. All MIDI hashes/independent parses, prompt and note-count preservation, and audio hashes/nonzero/clipping checks passed. Audio range requests returned matching bytes. The actual UI upload replay completed, and the refreshed page shows the corrected metric and stopping explanation. No human musical-quality score is assigned.

All **46 tests** passed in 7.24 seconds, including automatic/fixed budget boundary cases and historical result enrichment without mutation. Ruff lint and formatting passed. The exact source identity is `d3f9487d30fc2597d7b5ab0d8b7c1f3d13b7384dcaab2b644ee07ad66aff3fc3`, frozen in `outputs/duration-fix-source-d3f9487d30fc2597.zip` and its extracted checkout. The ZIP SHA256 is `384a0dfad3dc934d6340e487eff4087ba4b773a369100955588db2ec76cee7f2`.

[Measured verification](../outputs/duration-fix-verification.json) includes source identity, before/after jobs, hashes and checks. [The saved browser screenshot](../outputs/duration-fix-app.jpg) shows the 30-second Für Elise continuation and its audio tail. The temporary loopback screenshot-saving utility was stopped after saving the artifact.

Requested time remains an upper bound. The dense sonata still reaches context before 30 seconds, even with the corrected budget. Shortening its prompt frees more context; Aria has a larger context. Further duration-control or sliding-window generation would be a separate change with its own evaluation.
