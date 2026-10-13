"""Audit and render the declared comparison prompts without using the GPU."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.audio.piano import render_piano
from music_project.generation.midi import build_prompt, save_checked
from music_project.io import sha256_file, write_json
from music_project.tokens.aria import AriaTokenizer


def main():
    config_path = Path("configs/generation_suite.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path("outputs/evaluation-prompts")
    root.mkdir(parents=True, exist_ok=False)
    tokenizer = AriaTokenizer()
    results = []
    for declared in config["prompts"]:
        directory = root / declared["id"]
        directory.mkdir()
        tokens, midi, prompt = build_prompt(
            Path(declared["path"]), config["prompt_seconds"], tokenizer
        )
        _, checked = save_checked(midi, directory / "prompt.mid")
        if checked["notes"] != prompt["prompt_notes"] or checked["notes_merged_on_export"]:
            raise ValueError("Comparison prompt export changed note groups")
        audio = render_piano(directory / "prompt.mid", directory / "prompt.wav")
        result = {
            "declared": declared,
            "prompt": prompt,
            "midi": checked,
            "audio": audio,
            "context_tokens_with_generation": len(tokens) + config["max_new_tokens"],
            "fits_scratch_context": len(tokens) + config["max_new_tokens"] <= 2048,
        }
        if not result["fits_scratch_context"]:
            raise ValueError("Comparison prompt exceeds scratch context")
        write_json(directory / "report.json", result)
        results.append(result)
        print(
            f"Verified {declared['id']}: {prompt['prompt_notes']} notes, {len(tokens)} prompt tokens",
            flush=True,
        )
    write_json(
        root / "report.json", {"config_sha256": sha256_file(config_path), "prompts": results}
    )


if __name__ == "__main__":
    main()
