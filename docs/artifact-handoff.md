# Restore the private project artifacts

The code, configurations, reports and submission files live in Git. Trained weights, complete training state and listening evidence live in the private [submission release](https://github.com/zahoormzm/expressive-piano-continuation/releases/tag/submission-2026-10-13). An account with repository read access is required.

Download the ZIP assets listed in `submission/artifacts-manifest.json` into a `downloads` directory at the project root. The release also supplies `SHA256SUMS.txt`. From the project root, after creating the Python environment described in README:

```powershell
.\.venv\Scripts\python.exe tools/restore_artifacts.py --downloads downloads
.\.venv\Scripts\python.exe scripts/setup_assets.py
.\.venv\Scripts\python.exe scripts/serve_app.py
```

The restore tool verifies each archive's byte count and SHA256, rejects paths outside the repository and refuses to overwrite a changed local file. Archives preserve their original `runs/`, `outputs/` and `data/` paths. They include all six recorded scratch-training runs, including pilot/full LSTM and Transformer checkpoints, the continued Transformer and the musical Transformer. They also include preparation artifacts, source snapshots, diagnostics and saved listening/generation outputs. Environments and temporary test/publishing caches are rebuildable and are excluded.

`scripts/setup_assets.py` fetches the exact upstream Aria checkpoint, pinned Aria implementation, piano SoundFont and FluidSynth assets from `configs/assets.json`, with checksums. These third-party assets are separate from our scratch-model checkpoints. Run `scripts/download_data.py` only if the original MAESTRO MIDI files are needed. Its official download and data license remain the authoritative source. The release retains data preparation/derived tokens under the MAESTRO license and does not duplicate the original raw MIDI download.

For strict replay or resume of a recorded run, use its frozen training scripts from the restored source snapshot, as documented in the experiment reports. The repository `.gitattributes` also preserves code bytes instead of changing line endings during checkout. Each checkpoint identifies its own source, data and tokenizer; recent app code should not silently stand in for an older training source.

Reviewers can inspect the PDF, slides and reports without downloading multi-gigabyte checkpoints. For a live demo, restore the archives and upstream assets before the review. The owner still needs to grant faculty/TA repository access using their GitHub usernames.
