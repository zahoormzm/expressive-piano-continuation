# Scratch training screen — 6 October 2026

Both scratch models completed **5,030,045 valid target tokens in 163 optimizer updates** on the RTX 5070. The requested budget was five million tokens; completing the final update added 30,045 tokens. Both models learned, their checkpoints loaded successfully, and all 24 targeted tests passed.

This is an engineering screen on **25 training performances and eight fixed crops from two validation performances**. The five cached test performances were not loaded. The results do not establish audible quality or a final full-corpus model ranking. The project deadline remains **13 October 2026**.

## Measured results

Loss is mean next-token cross-entropy in nats; lower is better on these fixed validation crops.

| Model | Parameters | Initial validation loss | Final/best validation loss | Optimizer computation |
| --- | ---: | ---: | ---: | ---: |
| One-layer LSTM, width 256 | 9,620,287 | 9.7851 | 6.7621 | 219.93 seconds |
| Eight-layer Transformer, width 512 | 36,356,608 | 9.8802 | 6.0759 | 107.98 seconds |

The Transformer achieved lower loss at the matched token budget in this single-seed screen. It has substantially more parameters. Broader data, further training, generated samples, and listening comparisons are needed before choosing a final candidate.

Separate throughput benchmarks used 20 warmup and 100 measured microbatches, including real crop sampling, forward/backward passes, clipping, and optimizer updates. They exclude validation and checkpoint writes.

| Model | Valid target tokens/second | Peak GPU allocation | Peak GPU reservation |
| --- | ---: | ---: | ---: |
| LSTM | 22,584 | 0.63 GiB | 0.73 GiB |
| Transformer | 45,411 | 1.57 GiB | 1.65 GiB |

The table measures PyTorch allocation/reservation rather than total device memory used by all applications. Both configurations fit the 12 GB device without gradient checkpointing. Training computation times above exclude crop preparation, validation, and checkpoint writes.

## Shared protocol and verification

Both configurations used BF16 with PyTorch 2.14.1+cu130, seed 42, a 2,048-token context, microbatch one, and accumulation 16. They shared the manifest, tokenizer, token index, sampled token counts at every update, and fixed validation-window identity. Validation covered 16,365 valid target tokens.

The trainer shifts labels once, excludes supplied prefix/BOS and padding from loss, weights accumulation by actual valid targets, clips global gradients, and uses token-based warmup/cosine scheduling. Crops retain complete note groups within a performance and rebase time segments. Notes that began before the crop are not carried into its context.

Smaller deterministic FP32 diagnostics memorized one real training crop over 250 updates: LSTM loss fell from 9.8214 to 0.0720, and Transformer loss from 9.8081 to 0.0013. Recovery checks with dropout enabled reproduced subsequent losses, weights, and sampler state with zero observed numerical difference. These are debugging checks; exact numerical replay was established for the diagnostics on this hardware/software, rather than the main BF16 runs.

The full pilot `best.pt` and `latest.pt` files were independently loaded with `weights_only=True` and the trainer's strict identity/configuration checks. Each reproduced its recorded validation loss with zero observed difference. All logged losses and gradient norms are finite, update numbers are contiguous, and token totals agree with the summaries. Both best checkpoints are the final update.

The 24 tests cover foundation integrity, crop timing and boundaries, label shifting, loss masks, causal predictions, and weighted accumulation. Ruff lint/format and dependency checks pass.

## Artifacts and reproduction

- [LSTM summary](../runs/lstm-pilot-20261006-001/summary.json) and [best checkpoint](../runs/lstm-pilot-20261006-001/best.pt).
- [Transformer summary](../runs/transformer-small-pilot-20261006-001/summary.json) and [best checkpoint](../runs/transformer-small-pilot-20261006-001/best.pt).
- [Loss curves](../outputs/training-curves.png).
- [Machine-readable evidence](training-results.json), including configurations, hashes, benchmark results, and checkpoint verification.
- [Exact training source snapshot](../outputs/training-source-2be60a28e4f5aaa7.zip). Its contents reproduce the recorded source identity; the repository has no Git commit yet.

```powershell
$pythonExe = '.\.venv\Scripts\python.exe'
& $pythonExe scripts/train.py --config configs/lstm.yaml
& $pythonExe scripts/train.py --config configs/transformer_small.yaml
& $pythonExe scripts/plot_training.py --runs runs/lstm-pilot-20261006-001 runs/transformer-small-pilot-20261006-001
```

The first two commands create new runs. Resume uses `scripts/train.py --resume runs/SELECTED_RUN/latest.pt` with the exact saved source/configuration and data identities. A completed budget is not extended automatically. Checkpoints contain optimizer, schedule, sampler, and global random states; partial optimizer updates are not saved. Data, checkpoints, previews, and source snapshots are local ignored artifacts and must be preserved separately when moving the project.

## Next work

Inspect source/reconstruction audio and prepare tokens for the retained corpus. Add prompt-conditioned generation, verify an accessible pretrained Aria checkpoint and its tokenizer, and establish a consistent sampled-piano renderer. Then produce reproducible continuations for listening and choose substantive training runs using broader evidence.
