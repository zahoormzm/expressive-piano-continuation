"""Stream-audit a complete token cache and test training crops without fitting a model."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from music_project.data.windows import PerformanceTokens, collate
from music_project.io import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/tokenization-full-summary.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/maestro-v3-clean.jsonl")
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/full-cache-verification.json"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary["limited_pilot"] or summary["failures"]:
        raise ValueError("A complete cache without preparation failures is required")
    if sha256_file(args.manifest) != summary["manifest_sha256"]:
        raise ValueError("Manifest identity changed")
    cache = Path(summary["cache_directory"]).resolve()
    index_path = (cache / summary["index_file"]).resolve()
    if not index_path.is_relative_to(cache):
        raise ValueError("Index path leaves the token cache")
    identity = json.loads((cache / "tokenizer.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(
            {key: value for key, value in identity.items() if key != "sha256"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if digest != identity["sha256"] or digest != summary["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity changed")
    sources = {
        row["relative_path"]: row
        for row in (
            json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
        )
        if row["usable"]
    }
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != len(sources) or len({r["relative_path"] for r in records}) != len(records):
        raise ValueError("Full index has missing or duplicated performances")
    groups, hashes = defaultdict(set), defaultdict(set)
    for source in sources.values():
        groups[tuple(source["composition_key"])].add(source["split"])
        hashes[source["sha256"]].add(source["split"])
    if any(len(group) > 1 for group in groups.values()) or any(
        len(group) > 1 for group in hashes.values()
    ):
        raise ValueError("Retained source groups or exact hashes cross splits")
    started = time.perf_counter()
    counts, totals, events, crop_counts = Counter(), Counter(), Counter(), Counter()
    byte_count = 0
    for index, record in enumerate(records, 1):
        source = sources.get(record["relative_path"])
        if (
            source is None
            or source["split"] != record["split"]
            or source["sha256"] != record["source_sha256"]
        ):
            raise ValueError("Index record differs from its source manifest")
        path = (cache / record["token_file"]).resolve()
        if not path.is_relative_to(cache) or sha256_file(path) != record["token_sha256"]:
            raise ValueError("Token cache path or hash changed")
        ids = np.load(path, allow_pickle=False)
        if len(ids) != record["token_count"]:
            raise ValueError("Token count differs from its index")
        performance = PerformanceTokens(ids, identity["vocabulary"], record["source_sha256"])
        split = record["split"]
        counts[split] += 1
        totals[split] += len(ids)
        events[split] += len(performance.events)
        byte_count += path.stat().st_size
        if split in ("train", "validation"):
            for start in sorted({0, len(performance.events) // 2, len(performance.events) - 1}):
                window = performance.crop(start, 2048)
                batch = collate([window], 2048, identity["special_token_ids"]["<P>"])
                if batch["valid_tokens"] <= 0 or batch["source_ids"] != [record["source_sha256"]]:
                    raise ValueError("Training crop has invalid targets or piece identity")
                crop_counts[split] += 1
        if index % 100 == 0 or index == len(records):
            print(f"Audited {index}/{len(records)} token caches", flush=True)
    if (
        dict(counts) != summary["performance_totals_by_split"]
        or dict(totals) != summary["token_totals_by_split"]
    ):
        raise ValueError("Cache totals differ from the preparation summary")
    report = {
        "complete": True,
        "manifest_sha256": summary["manifest_sha256"],
        "tokenizer_sha256": digest,
        "index_sha256": sha256_file(index_path),
        "index": index_path.as_posix(),
        "performances_by_split": dict(counts),
        "tokens_by_split": dict(totals),
        "note_groups_by_split": dict(events),
        "training_crop_checks_by_split": dict(crop_counts),
        "context_length": 2048,
        "token_file_bytes": byte_count,
        "seconds": time.perf_counter() - started,
        "all_token_hashes_and_grammar_passed": True,
        "test_scope": "Token-file integrity and event syntax only; no test crops, model fitting, or evaluation",
        "normalization_notes_removed": summary["notes_removed_by_normalization"],
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
