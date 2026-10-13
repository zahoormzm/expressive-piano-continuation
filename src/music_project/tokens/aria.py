"""Wrap the published vocabulary without inventing checkpoint token IDs."""

import copy
import hashlib
import importlib.metadata
import json
import random
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from music_project.data.midi import read_performance
from music_project.io import write_json

ARIA_REVISION = "7c815755c6889a784b6a1b3343e9e8daa3e7d128"


class TempoMap:
    """Independent tick-to-second conversion, including implicit initial tempo."""

    def __init__(self, midi: MidiDict):
        tempos = {0: 500_000}
        tempos.update({message["tick"]: message["data"] for message in midi.tempo_msgs})
        self.ticks = sorted(tempos)
        self.tempos = [tempos[tick] for tick in self.ticks]
        self.seconds = [0.0]
        self.ticks_per_beat = midi.ticks_per_beat
        for index in range(1, len(self.ticks)):
            self.seconds.append(
                self.seconds[-1]
                + (self.ticks[index] - self.ticks[index - 1])
                * self.tempos[index - 1]
                / (1_000_000 * self.ticks_per_beat)
            )

    def at(self, tick: int) -> float:
        index = bisect_right(self.ticks, tick) - 1
        return self.seconds[index] + (tick - self.ticks[index]) * self.tempos[index] / (
            1_000_000 * self.ticks_per_beat
        )


def note_signature(midi: MidiDict) -> list[tuple]:
    timeline = TempoMap(midi)
    return sorted(
        (
            message["data"]["pitch"],
            message["data"]["velocity"],
            round(timeline.at(message["data"]["start"]) * 1000),
            round(timeline.at(message["data"]["end"]) * 1000),
        )
        for message in midi.note_msgs
    )


class AriaTokenizer:
    def __init__(self):
        distribution = importlib.metadata.distribution("ariautils")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        actual_revision = direct_url.get("vcs_info", {}).get("commit_id")
        if actual_revision != ARIA_REVISION:
            raise RuntimeError(
                f"Expected reviewed ariautils revision {ARIA_REVISION}; "
                f"installed revision is {actual_revision}"
            )
        self.tokenizer = AbsTokenizer()
        identity = {
            "toolkit": "EleutherAI/aria-utils",
            "revision": actual_revision,
            "class": "AbsTokenizer",
            "config": self.tokenizer.config,
            "vocabulary": list(self.tokenizer.vocab),
            "vocab_size": self.tokenizer.vocab_size,
            "special_token_ids": {
                token: self.tokenizer.tok_to_id[token] for token in self.tokenizer.special_tokens
            },
            "pipeline": {
                "preserve_leading_silence": True,
                "add_diminish_token": False,
                "append_eos": True,
                "sustain": "resolve once into note durations",
                "overlaps": "ariautils resolve_pedal/resolve_overlaps; counted",
                "augmentation": "none",
                "prefix_seed": 0,
                "implicit_initial_tempo": "500000 us/beat before first event",
            },
        }
        content = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.identity = identity | {"sha256": hashlib.sha256(content).hexdigest()}

    def normalize(self, original: MidiDict) -> tuple[MidiDict, dict]:
        midi = copy.deepcopy(original)
        channels = {message["channel"] for message in midi.note_msgs}
        if len(channels) != 1 or 9 in channels:
            raise ValueError("First pipeline requires solo piano on one non-drum channel")
        if any(
            not 0 <= message["data"] <= 7
            for message in midi.instrument_msgs
            if message["channel"] in channels
        ):
            raise ValueError("Non-piano program on an active channel")
        if any(message["data"]["end"] <= message["data"]["start"] for message in midi.note_msgs):
            raise ValueError("Nonpositive note duration")
        before = note_signature(midi)
        raw_pedals = len(midi.pedal_msgs)
        implicit_tempo = not any(message["tick"] == 0 for message in midi.tempo_msgs)
        if implicit_tempo:
            # MIDI specifies 500000 us/beat before its first tempo event.
            # The pinned toolkit otherwise uses a later tempo for that region.
            midi.tempo_msgs.insert(0, {"type": "tempo", "data": 500_000, "tick": 0})
        midi.resolve_pedal()
        after = note_signature(midi)
        # Duration already includes sustain. Never apply CC64 a second time.
        midi.pedal_msgs = []
        midi.metadata = {}
        return midi, {
            "source_note_count": len(before),
            "normalized_note_count": len(after),
            "implicit_initial_tempo_inserted": implicit_tempo,
            "sustain_messages_consumed": raw_pedals,
            "note_records_changed_by_sustain_or_overlap": sum(
                (Counter(before) - Counter(after)).values()
            ),
            "removed_or_merged_notes": len(before) - len(after),
        }

    def encode(self, midi: MidiDict) -> list[int]:
        # Upstream shuffles prefixes with Python's global RNG. Keep encoding
        # deterministic without changing the caller's training/sampling RNG.
        state = random.getstate()
        try:
            random.seed(0)
            tokens = self.tokenizer.tokenize(
                copy.deepcopy(midi),
                remove_preceding_silence=False,
                add_dim_tok=False,
                add_eos_tok=True,
            )
        finally:
            random.setstate(state)
        ids = self.tokenizer.encode(tokens)
        if self.tokenizer.tok_to_id[self.tokenizer.unk_tok] in ids:
            raise ValueError("Encoding contains unknown tokens")
        return ids

    def decode(self, ids: list[int]) -> MidiDict:
        if not ids or any(index not in self.tokenizer.id_to_tok for index in ids):
            raise ValueError("Empty sequence or token ID outside the published vocabulary")
        return self.tokenizer.detokenize(self.tokenizer.decode(ids))

    def expected_notes(self, normalized: MidiDict) -> list[tuple]:
        timeline = TempoMap(normalized)
        expected = []
        tokenizer = self.tokenizer
        for message in normalized.note_msgs:
            note = message["data"]
            onset_ms = round(timeline.at(note["start"]) * 1000)
            duration_ms = round((timeline.at(note["end"]) - timeline.at(note["start"])) * 1000)
            segment, local_onset = divmod(onset_ms, tokenizer.abs_time_step_ms)
            onset = segment * tokenizer.abs_time_step_ms + tokenizer._quantize_onset(local_onset)
            expected.append(
                (
                    note["pitch"],
                    tokenizer._quantize_velocity(note["velocity"]),
                    onset,
                    onset + tokenizer._quantize_dur(duration_ms),
                )
            )
        return sorted(expected)

    def reconstruct(self, path: Path, directory: Path) -> dict:
        source_audit = read_performance(path)
        if source_audit.issues.get("unmatched_note_on") or source_audit.issues.get(
            "nonpositive_note_duration"
        ):
            raise ValueError("Source has unresolved or nonpositive notes; inspect the audit")
        original = MidiDict.from_midi(path)
        if len(source_audit.notes) != len(original.note_msgs):
            raise ValueError("Independent parser and tokenizer disagree about source note count")
        normalized, normalization = self.normalize(original)
        ids = self.encode(normalized)
        reconstructed = self.decode(ids)
        expected = self.expected_notes(normalized)
        actual = note_signature(reconstructed)
        if actual != expected:
            raise ValueError("Decoded notes differ from the expected quantized source notes")
        directory.mkdir(parents=True, exist_ok=True)
        copy.deepcopy(normalized).to_midi().save(directory / "normalized_source.mid")
        serialized = copy.deepcopy(reconstructed)
        before_export = note_signature(serialized)
        serialized.to_midi().save(directory / "reconstructed.mid")
        after_export = note_signature(serialized)
        reloaded = MidiDict.from_midi(directory / "reconstructed.mid")
        if note_signature(reloaded) != after_export:
            raise ValueError("Serialized MIDI does not round-trip through the MIDI parser")
        timeline = TempoMap(normalized)
        max_export_end_adjustment = None
        if len(before_export) == len(after_export) and all(
            before[:3] == after[:3] for before, after in zip(before_export, after_export)
        ):
            max_export_end_adjustment = max(
                (abs(before[3] - after[3]) for before, after in zip(before_export, after_export)),
                default=0,
            )
        report = {
            "source_file": path.name,
            "tokenizer_sha256": self.identity["sha256"],
            "token_count": len(ids),
            "normalization": normalization,
            "expected_quantized_note_count": len(expected),
            "decoded_note_count": len(actual),
            "quantized_reconstruction_passed": True,
            "midi_serialization_passed": True,
            "note_records_changed_on_export": sum(
                (Counter(before_export) - Counter(after_export)).values()
            ),
            "notes_merged_on_export": len(before_export) - len(after_export),
            "max_export_end_adjustment_ms": max_export_end_adjustment,
            "duration_clipped_at_5000ms_count": sum(
                (timeline.at(message["data"]["end"]) - timeline.at(message["data"]["start"])) * 1000
                > 5000
                for message in normalized.note_msgs
            ),
            "pedal_limit": "CC64 sustain is represented in duration; CC66 sostenuto and "
            "CC67 una corda are not retained by this representation.",
        }
        write_json(directory / "report.json", report)
        return report
