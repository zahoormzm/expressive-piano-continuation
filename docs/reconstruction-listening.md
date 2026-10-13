# Reconstruction listening — 6 October 2026

Twelve source/reconstruction pairs are ready at [the local listening page](http://127.0.0.1:8766/reconstruction-listening/). Each pair covers the opening 20 seconds of a retained training performance. No validation or test performance was selected for this inspection.

Both sides use the same verified FluidSynth 2.6.1 and GeneralUser GS 2.0.3 acoustic grand piano, gain 0.25, 44.1 kHz stereo PCM16, no chorus/reverb, and no separate loudness normalization. All 24 WAVs are non-silent and unclipped. These are MIDI renders, rather than the dataset's acoustic recordings.

The source excerpt preserves original note timing/velocity and CC64/66/67 pedal messages, converted to a flat tempo with millisecond precision. The reconstruction excerpt uses the saved quantized reconstruction: sustain is already represented in note lengths. Notes are clipped and all three pedal controls are released at the 20-second boundary. Renderer release tails extend the WAV beyond 20 seconds. Retaining a pedal message does not establish that this SoundFont reproduces the acoustic instrument's response to it.

| Pair | Composer and work | Notes on each side |
| --- | --- | ---: |
| 1 | Bach — French Suite No. 5 | 162 |
| 2 | Chopin — Complete Etudes, Op. 25 | 279 |
| 3 | Mussorgsky — Pictures at an Exhibition | 93 |
| 4 | Schubert — Sonata D. 962 | 178 |
| 5 | Chopin — Etude Op. 25 No. 5 | 314 |
| 6 | Liszt — Sonetto 104 del Petrarca | 101 |
| 7 | Mendelssohn — Variations Serieuses | 82 |
| 8 | Debussy — La puerta del vino | 71 |
| 9 | Scriabin — Fantasy, Op. 28 | 127 |
| 10 | Debussy — Les fées sont d'exquises danseuses | 291 |
| 11 | Schubert — Impromptu Op. 142 No. 3 | 131 |
| 12 | Debussy — L'isle joyeuse | 217 |

The browser check confirmed that the first source and reconstruction players load, playback time advances, and switching players pauses the other. The artifact reports validate all 24 WAVs; this browser check did not audition all pairs or establish audible similarity.

Listen for changed attacks, notes ending too early, repeated-note articulation, pedal texture, and dynamics. The representation uses 10 ms timing steps and quantized velocity, clips note durations above five seconds, and does not retain CC66/67. Full-performance clipping counts in each pair report are not counts limited to the 20-second excerpt.

Human assessment is pending. An optional request for the user's assessment of the first few pairs has been sent. The user separately authorized the full-corpus training runs; starting those runs is not evidence that reconstruction has passed an auditory review.

The complete machine-readable evidence is [the pair report](../outputs/reconstruction-listening/report.json). Each `pair-01` through `pair-12` directory preserves MIDI, WAV, hashes, renderer settings, and source identifiers. Reproduction uses `scripts/prepare_reconstruction_listening.py` with a fresh output directory. Exact code is preserved in `outputs/full-training-source-225d8b0cb206711f.zip`.
