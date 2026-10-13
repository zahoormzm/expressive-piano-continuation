"""Produce immutable per-performance token caches and reconstruction diagnostics."""

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ariautils.midi import MidiDict

from music_project.io import sha256_file, write_bytes, write_json, write_jsonl
from music_project.tokens.aria import AriaTokenizer


def prepare(
    dataset: Path,
    manifest: Path,
    output: Path,
    limit: int | None,
    reconstruction_count: int,
    summary_path: Path = Path("outputs/tokenization-summary.json"),
) -> dict:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row["usable"]]
    composition_splits, content_splits = defaultdict(set), defaultdict(set)
    for row in selected:
        composition_splits[tuple(row["composition_key"])].add(row["split"])
        content_splits[row["sha256"]].add(row["split"])
    if any(len(group) > 1 for group in composition_splits.values()) or any(
        len(group) > 1 for group in content_splits.values()
    ):
        raise ValueError("Retained manifest crosses splits; run scripts/prepare_splits.py first")
    if not selected:
        raise ValueError("Manifest has no usable performances")
    dataset = dataset.resolve()
    if limit is not None:
        count = min(limit, len(selected))
        selected = [selected[index * len(selected) // count] for index in range(count)]
    tokenizer = AriaTokenizer()
    cache = output / tokenizer.identity["sha256"][:16]
    write_json(cache / "tokenizer.json", tokenizer.identity, immutable=True)
    records, failures, reconstruction_reports = [], [], []
    # Use training performances across years for reconstruction development.
    training = sorted(
        (row for row in selected if row["split"] == "train"),
        key=lambda row: (row["year"], row["canonical_composer"], row["note_count"]),
    )
    examples = []
    for year in sorted({row["year"] for row in training}):
        examples.append(next(row for row in training if row["year"] == year))
    for row in training:
        if len(examples) >= reconstruction_count:
            break
        if row not in examples:
            examples.append(row)
    example_paths = {row["relative_path"] for row in examples[:reconstruction_count]}
    for index, row in enumerate(selected, 1):
        path = (dataset / row["relative_path"]).resolve()
        try:
            if not path.is_relative_to(dataset):
                raise ValueError("Manifest path leaves dataset directory")
            if sha256_file(path) != row["sha256"]:
                raise ValueError("Source MIDI changed since validation")
            midi, normalization = tokenizer.normalize(MidiDict.from_midi(path))
            ids = tokenizer.encode(midi)
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(ids, dtype=np.uint32), allow_pickle=False)
            relative = f"{row['split']}/{row['sha256']}.npy"
            write_bytes(cache / relative, buffer.getvalue(), immutable=True)
            records.append(
                {
                    "relative_path": row["relative_path"],
                    "split": row["split"],
                    "source_sha256": row["sha256"],
                    "token_file": relative,
                    "token_sha256": sha256_file(cache / relative),
                    "token_count": len(ids),
                    "normalization": normalization,
                }
            )
            if row["relative_path"] in example_paths:
                directory = Path("outputs/reconstruction") / row["sha256"][:16]
                report = tokenizer.reconstruct(path, directory)
                report.update(
                    {
                        "relative_path": row["relative_path"],
                        "year": row["year"],
                        "composer": row["canonical_composer"],
                        "title": row["canonical_title"],
                        "directory": str(directory),
                    }
                )
                reconstruction_reports.append(report)
        except Exception as exc:  # noqa: BLE001 -- preserve each toolkit failure in the report
            failures.append({"path": row["relative_path"], "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0 or index == len(selected):
            print(
                f"Tokenized {index}/{len(selected)} performances; failures={len(failures)}",
                flush=True,
            )
    manifest_digest = sha256_file(manifest)
    index_name = f"index-{manifest_digest[:16]}-{'all' if limit is None else limit}.jsonl"
    write_jsonl(cache / index_name, records, immutable=True)
    summary = {
        "manifest_sha256": manifest_digest,
        "tokenizer_sha256": tokenizer.identity["sha256"],
        "vocab_size": tokenizer.tokenizer.vocab_size,
        "cache_directory": str(cache),
        "index_file": index_name,
        "limited_pilot": limit is not None,
        "selected_performances": len(selected),
        "tokenized_performances": len(records),
        "token_totals_by_split": {
            split: sum(row["token_count"] for row in records if row["split"] == split)
            for split in ("train", "validation", "test")
        },
        "performance_totals_by_split": dict(Counter(row["split"] for row in records)),
        "failures": failures,
        "reconstruction_examples": reconstruction_reports,
        "notes_removed_by_normalization": sum(
            row["normalization"]["removed_or_merged_notes"] for row in records
        ),
        "augmentation": "none",
        "windowing": "one cache per performance; trainer samples complete-event windows",
    }
    write_json(cache / f"summary-{index_name.removesuffix('.jsonl')}.json", summary)
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("vocab_size", "tokenized_performances", "token_totals_by_split")
            },
            indent=2,
        )
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} tokenization/reconstruction failures; inspect the report"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/maestro-v3.0.0"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/maestro-v3-clean.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("data/tokens"))
    parser.add_argument("--limit", type=int, help="Optional explicitly labelled small pilot")
    parser.add_argument("--reconstruction-count", type=int, default=12)
    parser.add_argument("--summary", type=Path, default=Path("outputs/tokenization-summary.json"))
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or args.reconstruction_count < 0:
        parser.error("limit must be positive and reconstruction-count must be nonnegative")
    prepare(
        args.dataset,
        args.manifest,
        args.output,
        args.limit,
        args.reconstruction_count,
        args.summary,
    )
