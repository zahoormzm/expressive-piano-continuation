"""Render same-piano source/reconstruction pairs from the existing training examples."""

import argparse
import html
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.audio.excerpts import piano_excerpt
from music_project.audio.piano import render_piano
from music_project.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction-listening"))
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads(Path("outputs/tokenization-summary.json").read_text(encoding="utf-8"))
    results, cards = [], []
    for index, example in enumerate(summary["reconstruction_examples"], 1):
        directory = args.output / f"pair-{index:02d}"
        directory.mkdir()
        source = Path("data/maestro-v3.0.0") / example["relative_path"]
        reconstruction = Path(example["directory"]) / "reconstructed.mid"
        pair = {
            "year": example["year"],
            "composer": example["composer"],
            "title": example["title"],
            "human_assessed": False,
        }
        for kind, path in [("source", source), ("reconstruction", reconstruction)]:
            pair[kind] = piano_excerpt(
                path, directory / f"{kind}.mid", args.seconds, retain_pedals=kind == "source"
            )
            pair[kind]["audio"] = render_piano(directory / f"{kind}.mid", directory / f"{kind}.wav")
        pair["representation"] = {
            key: example[key]
            for key in [
                "duration_clipped_at_5000ms_count",
                "max_export_end_adjustment_ms",
                "notes_merged_on_export",
                "pedal_limit",
            ]
        }
        write_json(directory / "report.json", pair)
        results.append(pair)
        title = html.escape(f"{example['composer']} — {example['title']} ({example['year']})")
        cards.append(
            f'<section><h2>{title}</h2><p>Original timing, velocity and pedal messages</p><audio aria-label="Play source {index}" controls preload="metadata" src="pair-{index:02d}/source.wav"></audio><p>Quantized reconstruction, with sustain in note lengths</p><audio aria-label="Play reconstruction {index}" controls preload="metadata" src="pair-{index:02d}/reconstruction.wav"></audio><p><a href="pair-{index:02d}/report.json">Pair details</a></p></section>'
        )
        print(
            f"Rendered reconstruction listening pair {index}/{len(summary['reconstruction_examples'])}",
            flush=True,
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reconstruction listening</title><style>body{font:16px/1.5 system-ui;margin:0;background:#f3f4ef;color:#183039}main{max-width:950px;margin:auto;padding:32px 20px}h1{font-size:36px}h2{font-size:19px}section{background:white;padding:24px;border:1px solid #dce3dd;border-radius:15px;margin:20px 0}p{color:#65767a}audio{width:100%}a{color:#286956}</style><main><h1>Reconstruction listening</h1><p>Compare the first 20 seconds of each training example, rendered with the same piano and fixed gain. Listen for note timing, sustain, and long-note changes. Pedal controls are released at the excerpt boundary. These pairs have no human score yet.</p>'
        + "".join(cards)
        + '</main><script>const players=[...document.querySelectorAll("audio")];for(const p of players)p.addEventListener("play",()=>{for(const other of players)if(other!==p)other.pause()});</script></html>'
    )
    (args.output / "index.html").write_text(document, encoding="utf-8")
    write_json(
        args.output / "report.json",
        {
            "seconds": args.seconds,
            "pairs": results,
            "human_assessed": False,
            "same_renderer_and_gain": True,
        },
    )


if __name__ == "__main__":
    main()
