# MIDI generation and piano playback results

On **6 October 2026**, both trained scratch models and the pretrained Aria generation checkpoint produced continuations of the same authored piano prompt. All selected MIDI files preserve the quantized prompt and generated note counts, and all piano WAVs are non-silent and unclipped. The local listening page is ready. Human musical-quality assessment remains outstanding.

## Selected samples

| Model | Parameters | Generated notes | Sampling time | Peak GPU allocation | Combined WAV |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM pilot | 9,620,287 | 45 | 0.77 s | 0.078 GiB | 23.75 s |
| Scratch Transformer pilot | 36,356,608 | 76 | 2.94 s | 0.185 GiB | 23.55 s |
| Pretrained Aria | 658,538,496 | 54 | 3.21 s | 1.318 GiB | 22.95 s |

These are single measurements on the RTX 5070 using BF16. Sampling time includes prompt inference and autoregressive decoding, and excludes model loading and audio rendering. Token and note counts differ between samples, so these timings are not a throughput benchmark. GPU values are peak allocated memory, not total device usage. The scratch checkpoints each trained on the small 25-performance training cache for 5,030,045 valid target tokens; their outputs are preliminary.

| Model | MIDI | Piano audio | Full evidence |
| --- | --- | --- | --- |
| LSTM | [Prompt + continuation](../outputs/generation/lstm-arpeggio-003/combined.mid), [continuation only](../outputs/generation/lstm-arpeggio-003/continuation.mid) | [Combined WAV](../outputs/generation/lstm-arpeggio-003/combined.wav) | [Sample report](../outputs/generation/lstm-arpeggio-003/report.json) |
| Scratch Transformer | [Prompt + continuation](../outputs/generation/transformer-arpeggio-003/combined.mid), [continuation only](../outputs/generation/transformer-arpeggio-003/continuation.mid) | [Combined WAV](../outputs/generation/transformer-arpeggio-003/combined.wav) | [Sample report](../outputs/generation/transformer-arpeggio-003/report.json) |
| Aria | [Prompt + continuation](../outputs/generation/aria-arpeggio-002/combined.mid), [continuation only](../outputs/generation/aria-arpeggio-002/continuation.mid) | [Combined WAV](../outputs/generation/aria-arpeggio-002/combined.wav) | [Sample report](../outputs/generation/aria-arpeggio-002/report.json) |

Open [the local listening page](http://127.0.0.1:8766/listening/) while the server is running. It provides playback, piano rolls, and MIDI downloads. From the project root, restart it with:

```powershell
& .\.venv\Scripts\python.exe -m http.server 8766 --bind 127.0.0.1 --directory outputs
```

The page source is [outputs/listening/index.html](../outputs/listening/index.html); it can also be opened directly. See the README for generation and asset setup commands. Machine-readable results are in [generation-results.json](generation-results.json).

## Shared protocol

The input is a project-authored eight-second arpeggio, with 36 notes and 111 prompt tokens. It is an engineering fixture, not a held-out MAESTRO example. All three exported prompt MIDI hashes and prompt WAV hashes agree. The source prompt SHA256 is `0c83a17a0fa638545afc484026f743c312ee33004e2dd1c08ae3e9ac64e1d938`.

Each model uses seed 42, temperature 0.98, min-p 0.035, a 384-new-token ceiling, and new-note onsets in **[8, 20) seconds**. Shared decoding enforces complete pitch/onset/duration groups, monotonic onsets, pitch 21–108, positive velocity/duration, at most 32 simultaneous note groups, and no repeated pitch at the same onset. A private seeded generator samples only legal tokens. There is no replacement sequence or fallback melody.

All selected samples stop when their clock reaches the time horizon, after 138, 231, and 165 new tokens respectively. None samples EOS; an EOS token is appended only for MIDI serialization and is marked in the report. The decoder permits the clock transition that ends the horizon rather than forcing more notes at the last legal instant. Notes must begin within the horizon, but their releases may extend later: final continuation note endings are 12.79, 12.10, and 11.96 seconds after the prompt boundary. FluidSynth adds further release/quiet tails. The resulting WAV durations therefore exceed the nominal prompt plus onset horizon.

The prompt wrapper resolves sustain once, preserves leading silence, clips quantized notes at the declared boundary, and uses the MIDI parser's simultaneous-note order. Generation rejects context overflow. The LSTM carries its inference hidden state; the scratch Transformer recomputes its bounded context; Aria uses its native PyTorch KV cache without compilation. The shared constrained sampler is different from Aria's upstream sampling implementation, so this is a pretrained-weight baseline within the project protocol.

## Pretrained identity and compatibility

The author's public [Aria model repository](https://huggingface.co/loubb/aria-medium-base) contains `model-gen.safetensors`, the generation-finetuned variant. We downloaded that file at revision `5a1ef36c5007ee10fbe57d61d96725a7244bd2b2`. Its 2,634,170,640 bytes match SHA256 `5e6b07ee2680f0044deea30da613d89b12e3e30d012d1812b388d6b8d8306c90`. The native source is [EleutherAI/aria commit c2f67bc9](https://github.com/EleutherAI/aria/tree/c2f67bc90b49335543c0249fe00fb84a9337faf8), pinned and checked for local modifications.

The measured medium model has 16 layers, width 1,536, 24 attention heads, an 8,192-token context, and vocabulary 17,727. Strict state loading passed. No further training or adapters are applied. The downloaded FP32 checkpoint remains intact; tensors are cast to BF16 for inference, which can change numerical predictions.

The published vocabulary size, EOS ID, and PAD ID agree with the pinned tokenizer. This authored prompt's encoded IDs exactly match the native upstream prompt function. Cached incremental logits and full prefill are finite and pass the BF16 comparison tolerance (`rtol=0.03`, `atol=0.25`), with maximum absolute difference 0.09375 and RMSE 0.008874. This validates the recorded check, rather than establishing exact cache replay or equivalent preprocessing for arbitrary MIDI inputs. Evidence: [pretrained-verification.json](../outputs/pretrained-verification.json).

## Piano rendering and integrity

All three use the same [FluidSynth 2.6.1 Windows release](https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.1) and [GeneralUser GS 2.0.3 source revision](https://github.com/mrbumpy409/GeneralUser-GS/tree/684543d5e5efaef08d02be50dcda8d552478fa60), acoustic grand program 0, fixed gain 0.25, polyphony 256, no chorus/reverb, and 44.1 kHz stereo PCM16. Source archives, executable/DLL bytes, and the SoundFont are verified before rendering. The asset registry is [configs/assets.json](../configs/assets.json). S. Christian Collins's SoundFont license is preserved in `data/assets/GeneralUser-GS-LICENSE.txt`.

There is no independent loudness normalization. The generated continuation peaks are approximately 0.197, 0.194, and 0.039; velocity differences therefore remain audible. Every WAV has zero clipped samples. Each MIDI passes saved-file readback and an independent note audit. All samples preserve the 36 prompt notes and every generated group, with **zero note merges** on export. Reports expose any ending adjustments separately.

All **32 automated tests** pass, including prompt clipping/preservation, no duplicate onset/pitch, complete-note stopping, horizon transitions, seeded sampling, and cached LSTM agreement. Ruff lint/format and dependency checks pass. Browser checks confirm finite metadata durations, advancing playback for all four players, no reported media errors, and automatic pause when switching players. Visual inspection confirms the model labels, counts, piano rolls, and download links. These checks establish working playback; they do not replace listening assessment.

## Reproducibility and limits

The exact generation source identity is `684de2619cda4d725a7570f8e6004723879a6703e3ff9b2d102be2985e70b869`, preserved in [generation-source-684de2619cda4d72.zip](../outputs/generation-source-684de2619cda4d72.zip). ZIP SHA256: `65fa21af77d0150f1de736731b98acbab4608966548a1b96b0fae4847cf69e40`. A later page-builder lint-only edit is outside that sample snapshot; it does not alter the decoder or sample files.

Training source remains separately frozen in `outputs/training-source-2be60a28e4f5aaa7.zip`. Both old pilot checkpoints passed strict load and reproduced validation losses from the isolated frozen checkout with zero difference. [Resume evidence](../outputs/resume-with-frozen-source.json) and the README specify how to use it after source changes. Runtime assets, samples, and source snapshots are ignored by Git and must be preserved separately when moving the project.

No human listening score, multiple-seed comparison, or full-corpus training result is claimed. Different model sizes, training histories, and decode implementations limit comparisons. Earlier development samples remain on disk, while this page selects only the folders listed above. Next work is reconstruction listening, full-corpus preparation, a local MIDI upload/generation application, and controlled multi-prompt evaluation ahead of the **13 October** submission.
