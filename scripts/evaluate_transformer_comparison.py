"""Replay shared validation and inspect note-pitch, onset and duration prediction."""

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.nn import functional as F

from music_project.data.musical_training import continuation_window
from music_project.data.windows import collate
from music_project.io import sha256_file, write_json
from music_project.resources import gpu_slot
from music_project.training.engine import Trainer, precision_context


@torch.no_grad()
def evaluate(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.name not in ("best.pt", "latest.pt") or payload["format_version"] != 1:
        raise ValueError("Evaluate a recorded best.pt or latest.pt checkpoint")
    trainer = Trainer(payload["config"])
    # Evaluation uses this script's source; require every recorded data/crop identity.
    fields = ("manifest_sha256", "tokenizer_sha256", "index_sha256", "validation_windows_sha256")
    if any(trainer.identity[k] != payload["identity"][k] for k in fields):
        raise ValueError("Evaluation data or fixed windows differ from training")
    if trainer.train_data.identity != payload["tokenizer"]:
        raise ValueError("Checkpoint tokenizer differs from evaluation")
    trainer.model.load_state_dict(payload["model"], strict=True)
    trainer.model.eval()
    vocabulary = payload["tokenizer"]["vocabulary"]
    categories = ("piano_note", "onset", "duration", "segment_or_end", "other")
    totals = {key: {"loss_sum": 0.0, "targets": 0} for key in categories}
    token_category = []
    for token in vocabulary:
        kind = token[0] if isinstance(token, (list, tuple)) else token
        token_category.append({"piano": 0, "onset": 1, "dur": 2, "<T>": 3, "<E>": 3}.get(kind, 4))
    token_category = torch.tensor(token_category, device=trainer.device)
    piano_ids = sorted(
        (i for i, t in enumerate(vocabulary) if isinstance(t, (list, tuple)) and t[0] == "piano"),
        key=lambda i: vocabulary[i][1:],
    )
    piano_tokens = [vocabulary[i] for i in piano_ids]
    velocities = sorted({t[2] for t in piano_tokens})
    if [(t[1], t[2]) for t in piano_tokens] != [(p, v) for p in range(128) for v in velocities]:
        raise ValueError("Unexpected piano vocabulary layout")
    piano_ids = torch.tensor(piano_ids, device=trainer.device)
    token_pitch = torch.tensor(
        [t[1] if isinstance(t, (list, tuple)) and t[0] == "piano" else -1 for t in vocabulary],
        device=trainer.device,
    )
    pitch_loss, pitch_correct, note_count = 0.0, 0, 0
    settings = trainer.config["training"]
    focused_windows = [
        continuation_window(w, vocabulary, 6000)[0] for w in trainer.validation_windows
    ]
    continuation_loss, continuation_tokens = 0.0, 0
    for start in range(0, len(trainer.validation_windows), settings["microbatch"]):
        batch = collate(
            trainer.validation_windows[start : start + settings["microbatch"]],
            settings["context_length"],
            payload["tokenizer"]["special_token_ids"]["<P>"],
        )
        inputs, targets, mask = (
            batch[k].to(trainer.device) for k in ("input_ids", "targets", "attention_mask")
        )
        with precision_context(trainer.device, trainer.config["precision"]):
            logits = trainer.model(inputs, mask)
        valid = targets != -100
        target_ids = targets[valid]
        scores = logits[valid].float()
        losses = F.cross_entropy(scores, target_ids, reduction="none")
        focused = collate(
            focused_windows[start : start + settings["microbatch"]],
            settings["context_length"],
            payload["tokenizer"]["special_token_ids"]["<P>"],
        )
        focus_mask = (focused["targets"].to(trainer.device) != -100)[valid]
        continuation_loss += losses[focus_mask].double().sum().item()
        continuation_tokens += int(focus_mask.sum())
        kinds = token_category[target_ids]
        for index, category in enumerate(categories):
            selected = kinds == index
            totals[category]["loss_sum"] += losses[selected].double().sum().item()
            totals[category]["targets"] += int(selected.sum())
        notes = kinds == 0
        # Marginalize velocities and condition on the next token being a piano note.
        # This is teacher-forced prediction, not a measure of complete melody coherence.
        note_scores = scores[notes][:, piano_ids].reshape(-1, 128, len(velocities))
        pitch_scores = torch.logsumexp(note_scores, dim=-1)
        correct_pitch = token_pitch[target_ids[notes]]
        pitch_loss += F.cross_entropy(pitch_scores, correct_pitch, reduction="sum").item()
        pitch_correct += int((pitch_scores.argmax(-1) == correct_pitch).sum())
        note_count += int(notes.sum())
    valid_tokens = sum(t["targets"] for t in totals.values())
    loss = sum(t["loss_sum"] for t in totals.values()) / valid_tokens
    if checkpoint.name == "best.pt":
        expected = payload["state"]["best_validation_loss"]
    else:
        summary = json.loads((checkpoint.parent / "summary.json").read_text())
        if payload["state"]["update"] != summary["update"]:
            raise ValueError("Latest checkpoint does not match the final recorded update")
        expected = summary["final_validation_loss"]
    if not math.isclose(loss, expected, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError(f"Validation replay mismatch: {loss} versus {expected}")
    for row in totals.values():
        row["mean_loss"] = row["loss_sum"] / row["targets"] if row["targets"] else None
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_identity": payload["identity"],
        "evaluation_source_sha256": trainer.identity["source_sha256"],
        "validation_loss": loss,
        "expected_validation_loss": expected,
        "replay_delta": loss - expected,
        "valid_tokens": valid_tokens,
        "windows": len(trainer.validation_windows),
        "continuation_validation": {
            "loss": continuation_loss / continuation_tokens,
            "valid_tokens": continuation_tokens,
            "requested_prompt_seconds": 6,
            "method": "complete onset groups; prompt capped at half the crop notes; no augmentation",
        },
        "categories": totals,
        "conditional_note_pitch": {
            "mean_loss": pitch_loss / note_count,
            "top1_accuracy": pitch_correct / note_count,
            "notes": note_count,
            "definition": "Given real previous tokens and that the next token is a piano note, marginalize velocity and predict pitch across all 128 MIDI pitches. No constrained sampler applied.",
        },
        "test_data_used": False,
        "human_assessed": False,
    }
    del trainer, payload
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=2, type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/transformer-comparison-validation.json")
    )
    args = parser.parse_args()
    with gpu_slot("Transformer comparison validation"):
        results = []
        for checkpoint in args.checkpoints:
            result = evaluate(checkpoint)
            results.append(result)
            print(
                f"{checkpoint}: loss={result['validation_loss']:.6f}, pitch accuracy={result['conditional_note_pitch']['top1_accuracy']:.3%}",
                flush=True,
            )
        fields = (
            "manifest_sha256",
            "tokenizer_sha256",
            "index_sha256",
            "validation_windows_sha256",
        )
        if any(
            results[0]["checkpoint_identity"][k] != results[1]["checkpoint_identity"][k]
            for k in fields
        ):
            raise ValueError("Checkpoints must share data and evaluation identities")
        write_json(
            args.output, {"results": results, "test_data_used": False, "human_assessed": False}
        )


if __name__ == "__main__":
    main()
