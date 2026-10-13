"""Build a local comparison page from measured MIDI/audio sample artifacts."""

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import _bootstrap  # noqa: F401

from music_project.data.midi import read_performance

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Piano continuation lab</title><style>
:root{color-scheme:light;--ink:#182b32;--muted:#657379;--line:#dce2df;--paper:#f3f4ef}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
header{background:#182b32;color:#fafcf9;padding:18px 6vw;display:flex;justify-content:space-between;align-items:center}
.brand{font-weight:650;letter-spacing:.02em}.local{font-size:12px;color:#c2d7cf;letter-spacing:.08em;text-transform:uppercase}
main{max-width:1120px;margin:auto;padding:42px 24px 56px}.eyebrow{color:#2e7665;font-size:12px;font-weight:650;letter-spacing:.1em;text-transform:uppercase}
h1{font-size:clamp(30px,4.3vw,48px);line-height:1.12;margin:12px 0 16px;letter-spacing:-.035em}h2{font-size:22px;margin:0}p{margin:8px 0;color:var(--muted)}
.intro{max-width:720px}.tags{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 28px}.tag{border:1px solid var(--line);border-radius:30px;padding:5px 12px;font-size:12px;background:#fafbf7}
.prompt{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:center;background:#e4ede6;border-radius:16px;padding:24px;margin-bottom:30px}
.prompt h2{font-size:18px}.prompt p{font-size:14px}.prompt audio{width:100%}a{color:#276b5a;text-underline-offset:3px}
.models{display:grid;gap:20px;margin-bottom:42px}.card{border:1px solid var(--line);border-radius:18px;background:#fff;padding:24px}.group-title{margin:34px 0 14px}.failure{background:#fff3ed;padding:18px;border-radius:12px}
.card-top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.model-info p{font-size:13px;margin:4px 0 0}.stats{text-align:right;font-size:13px;color:var(--muted)}.stats strong{color:var(--ink);font-size:18px}
.roll{width:100%;display:block;margin:18px 0 12px;overflow:visible}.roll text{font:11px system-ui,sans-serif;fill:#7b878a}.roll .cursor{stroke:#192e35;stroke-width:1.5;opacity:0}
.playback{display:flex;align-items:center;gap:20px;flex-wrap:wrap}.playback audio{flex:1;min-width:240px;height:42px}.downloads{display:flex;gap:16px;font-size:13px;white-space:nowrap}
.caption{font-size:12px;color:var(--muted);margin-top:12px}.note{border-top:1px solid var(--line);margin-top:30px;padding-top:20px;font-size:13px;max-width:920px}.note strong{color:var(--ink)}
audio{accent-color:#36735f}button,a,audio{outline-offset:4px}@media(max-width:640px){main{padding:28px 16px}.prompt{grid-template-columns:1fr;gap:12px}.card{padding:18px}.card-top{gap:12px}.local{display:none}.stats{max-width:130px}.downloads{width:100%}.roll{margin-top:24px}}
</style></head><body><header><span class="brand">♩ Piano continuation lab</span><span class="local">Local project · 6 October 2026</span></header>
<main><div class="eyebrow">Controlled generation samples</div><h1>Piano ideas.<br>Three ways forward.</h1>
<p class="intro">Compare each model using the same prompt, seed and sampling settings within each group. Every player includes the prompt; the dashed line marks where generation begins. Listen for continuity, harmony, rhythm and dynamics.</p>
<div class="tags"><span class="tag">Same piano + fixed gain</span><span class="tag">Prompt and note counts preserved</span><span class="tag">Shared constrained sampler</span></div>
{{GROUPS}}
<p class="note"><strong>These are engineering samples, with no human listening score yet.</strong> Scratch cards show the actual training tokens at the selected checkpoint. Aria uses the author's generation-finetuned weights without further adaptation. All three use shared constrained decoding, which differs from Aria's upstream sampler. The requested horizon bounds new note onsets; a token limit or sampled end can stop earlier. Note releases and renderer tails can extend later. Playback uses FluidSynth 2.6.1, GeneralUser GS 2.0.3 acoustic grand piano, 44.1 kHz stereo PCM, gain 0.25, and no reverb or chorus. Labels are visible; this page is not a blinded listening study.</p>
</main><script>
const players=Array.from(document.querySelectorAll('audio'));
for(const player of players){player.addEventListener('play',()=>{for(const other of players)if(other!==player)other.pause()});const card=player.closest('.card');if(!card)continue;const cursor=card.querySelector('.cursor');player.addEventListener('timeupdate',()=>{const duration=Number(card.dataset.timeline);const x=40+player.currentTime/duration*920;cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);cursor.style.opacity='1'})}
</script></body></html>"""


def group(samples, output, description):
    reports = [
        json.loads((sample / "report.json").read_text(encoding="utf-8")) for sample in samples
    ]
    if len({report["audio"]["prompt"]["wav_sha256"] for report in reports}) != 1:
        raise ValueError("Comparison samples must have identical prompt audio")
    if (
        len(
            {
                (
                    r["seed"],
                    r["temperature"],
                    r["min_p"],
                    tuple(r["sampling"]["constraints"]["new_note_onsets_ms"]),
                )
                for r in reports
            }
        )
        != 1
    ):
        raise ValueError("Comparison samples must share sampling settings")
    for report in reports:
        export = report["export"]
        if (
            not export["note_count_preserved"]
            or not export["prompt_preserved"]
            or any(
                export[kind]["notes_merged_on_export"]
                for kind in ["prompt", "combined", "continuation"]
            )
        ):
            raise ValueError("Comparison samples must preserve prompt and note groups")
        if any(audio["clipped_samples"] or audio["rms"] <= 0 for audio in report["audio"].values()):
            raise ValueError("Comparison audio must be nonzero and unclipped")
    timeline = math.ceil(max(report["audio"]["combined"]["seconds"] for report in reports) / 4) * 4

    def link(path):
        return quote(os.path.relpath(path, output.parent).replace("\\", "/"), safe="/")

    cards = []
    titles = {"lstm": "LSTM", "transformer": "Scratch Transformer", "aria": "Pretrained Aria"}
    colors = {"lstm": "#a47143", "transformer": "#6162a6", "aria": "#32806b"}
    for sample, report in zip(samples, reports):
        backend = report["model"]["backend"]
        label = titles[backend]
        notes = read_performance(sample / "combined.mid").notes
        low, high = min(note.pitch for note in notes) - 3, max(note.pitch for note in notes) + 3
        boundary = report["prompt"]["boundary_ms"] / 1000
        edge = 40 + boundary / timeline * 920
        svg = [
            f'<svg class="roll" viewBox="0 0 1000 152" role="img" aria-label="Piano roll for {label}; continuation begins at {boundary:g} seconds">',
            '<rect x="40" y="24" width="920" height="98" rx="6" fill="#f7f8f5"/>',
            f'<rect x="40" y="24" width="{edge - 40:.2f}" height="98" rx="6" fill="#e9eee9"/>',
            f'<text x="48" y="15">PROMPT</text><text x="{edge + 8:.2f}" y="15">CONTINUATION</text>',
        ]
        for second in range(0, timeline + 1, 4):
            x = 40 + second / timeline * 920
            svg.append(
                f'<line x1="{x}" y1="24" x2="{x}" y2="122" stroke="#dce3dc"/><text x="{x}" y="144" text-anchor="middle">{second}s</text>'
            )
        for note in notes:
            x = 40 + note.start_seconds / timeline * 920
            width = max(1.2, (note.end_seconds - note.start_seconds) / timeline * 920)
            y = 27 + (high - note.pitch) / (high - low) * 90
            color = "#92a69b" if note.start_seconds < boundary else colors[backend]
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="2.8" rx="1" fill="{color}"/>'
            )
        svg.extend(
            [
                f'<line x1="{edge}" y1="22" x2="{edge}" y2="126" stroke="#61736b" stroke-dasharray="3 3"/>',
                '<line class="cursor" x1="40" y1="24" x2="40" y2="122"/></svg>',
            ]
        )
        parameters = report["model"]["parameter_count"] / 1e6
        role = (
            "Unmodified generation weights"
            if backend == "aria"
            else f"Scratch checkpoint · {report['model']['training_valid_tokens'] / 1e6:.2f}M training tokens"
        )
        stop = html.escape(report["sampling"]["stop_reason"].replace("_", " "))
        cards.append(f'''<section class="card" data-timeline="{timeline}" aria-label="{label} sample">
<div class="card-top"><div class="model-info"><h2>{label}</h2><p>{role} · {parameters:.2f}M parameters</p></div><div class="stats"><strong>{report["export"]["continuation"]["notes"]} notes</strong><br>sampled in {report["sampling_seconds"]:.2f}s</div></div>
{"".join(svg)}<div class="playback"><audio controls preload="metadata" aria-label="Play {label} prompt and continuation" src="{link(sample / "combined.wav")}"></audio><div class="downloads"><a href="{link(sample / "combined.mid")}" download>Full MIDI</a><a href="{link(sample / "continuation.mid")}" download>Continuation MIDI</a></div></div>
<p class="caption">Stopped at {stop} · Prompt preserved · Zero notes merged · {report["export"]["combined"]["note_records_changed_on_export"]} note records adjusted during export · Audio unclipped · <a href="{link(sample / "report.json")}">Sample details</a></p></section>''')
    first = reports[0]
    boundary = first["prompt"]["boundary_ms"] / 1000
    end = first["sampling"]["constraints"]["new_note_onsets_ms"][1] / 1000
    return f'''<h2 class="group-title">{html.escape(description)} · seed {first["seed"]}</h2>
<p>Temperature {first["temperature"]} · min-p {first["min_p"]} · {end - boundary:g}-second new-onset horizon</p>
<section class="prompt" aria-label="Shared prompt"><div><h2>The starting idea</h2><p>{boundary:g} seconds · {first["prompt"]["prompt_notes"]} quantized notes</p><a href="{html.escape(link(samples[0] / "prompt.mid"))}" download>Download prompt MIDI</a></div><audio controls preload="metadata" aria-label="Play {html.escape(description)} seed {first["seed"]} prompt" src="{html.escape(link(samples[0] / "prompt.wav"))}"></audio></section>
<div class="models">{"".join(cards)}</div>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/generation/lstm-arpeggio-003"),
            Path("outputs/generation/transformer-arpeggio-003"),
            Path("outputs/generation/aria-arpeggio-002"),
        ],
    )
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/listening/index.html"))
    args = parser.parse_args()
    groups = []
    if args.suite:
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        if not suite["complete"]:
            raise ValueError("Finish every declared suite attempt before building the comparison")
        for prompt in suite["protocol"]["prompts"]:
            for seed in suite["protocol"]["seeds"]:
                attempts = [
                    a
                    for a in suite["attempts"]
                    if a["prompt"] == prompt["id"] and a["seed"] == seed
                ]
                samples = [Path(a["directory"]) for a in attempts if a["success"]]
                if samples:
                    groups.append(group(samples, args.output, prompt["description"]))
                for attempt in attempts:
                    if not attempt["success"]:
                        groups.append(
                            f'<p class="failure">{html.escape(prompt["description"])} · seed {seed} · {html.escape(attempt["label"])}: generation failed. This attempt is retained in the suite report.</p>'
                        )
    else:
        groups.append(group(args.samples, args.output, "Authored arpeggio"))
    content = PAGE.replace("{{GROUPS}}", "\n".join(groups))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
