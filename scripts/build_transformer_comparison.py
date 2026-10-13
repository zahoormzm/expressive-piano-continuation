"""Build a local A/B listening form from all declared Transformer comparison attempts."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import _bootstrap  # noqa: F401

from music_project.io import sha256_file


def build(suite_path: Path, output: Path):
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    protocol = suite["protocol"]
    if not suite["complete"] or suite["failures"]:
        raise ValueError("Complete every declared comparison attempt successfully first")
    if {m["id"] for m in protocol["models"]} != {"baseline", "continued"}:
        raise ValueError("Expected baseline and continued Transformer checkpoints")
    labels = {m["id"]: m["label"] for m in protocol["models"]}
    mapping, groups = {}, []
    training_tokens = {}

    def link(path):
        return quote(os.path.relpath(path, output.parent).replace("\\", "/"), safe="/")

    for prompt in protocol["prompts"]:
        for seed in protocol["seeds"]:
            key = f"{prompt['id']}-seed{seed}"
            attempts = [
                a for a in suite["attempts"] if a["prompt"] == prompt["id"] and a["seed"] == seed
            ]
            if len(attempts) != 2 or {a["model"] for a in attempts} != set(labels):
                raise ValueError("Each group requires exactly one sample of each checkpoint")
            attempts.sort(key=lambda a: a["model"])
            # Reproducible presentation order, independent of model outputs or preferences.
            if hashlib.sha256(key.encode()).digest()[0] % 2:
                attempts.reverse()
            cards, reports = [], []
            mapping[key] = {}
            for letter, attempt in zip("AB", attempts):
                sample = Path(attempt["directory"])
                report = json.loads((sample / "report.json").read_text(encoding="utf-8"))
                reports.append(report)
                training_tokens[attempt["model"]] = report["model"]["training_valid_tokens"]
                if (
                    report["model"]["checkpoint_sha256"]
                    != suite["checkpoint_hashes"][attempt["model"]]
                ):
                    raise ValueError("Sample checkpoint identity disagrees with the declared suite")
                if (
                    not report["export"]["prompt_preserved"]
                    or not report["export"]["note_count_preserved"]
                ):
                    raise ValueError("Prompt or note counts were not preserved")
                for kind in ("prompt", "combined", "continuation"):
                    audio = report["audio"][kind]
                    if audio["rms"] <= 0 or audio["clipped_samples"]:
                        raise ValueError("Comparison audio must be nonzero and unclipped")
                    if sha256_file(sample / f"{kind}.wav") != audio["wav_sha256"]:
                        raise ValueError("Comparison WAV bytes disagree with their report")
                    if report["export"][kind]["notes_merged_on_export"]:
                        raise ValueError("Comparison export merged note groups")
                mapping[key][letter] = {
                    "model": attempt["model"],
                    "label": labels[attempt["model"]],
                    "checkpoint_sha256": report["model"]["checkpoint_sha256"],
                    "report": link(sample / "report.json"),
                }
                stop = html.escape(report["sampling"]["stop_reason"].replace("_", " "))
                cards.append(f'''<article><h3>Version {letter}</h3>
<label>New music only<audio controls preload="metadata" aria-label="{html.escape(key)} version {letter} continuation" src="{link(sample / "continuation.wav")}"></audio></label>
<details><summary>Hear the prompt followed by this continuation</summary><audio controls preload="metadata" aria-label="{html.escape(key)} version {letter} combined" src="{link(sample / "combined.wav")}"></audio></details>
<p class="small">Stopped at {stop}. Note releases and piano tails can extend beyond the onset limit.</p>
<a href="{link(sample / "continuation.mid")}" download>Download MIDI</a>
<p class="identity" data-key="{key}" data-letter="{letter}" hidden></p></article>''')
            if (
                reports[0]["audio"]["prompt"]["wav_sha256"]
                != reports[1]["audio"]["prompt"]["wav_sha256"]
            ):
                raise ValueError("The paired prompt audio must be byte-identical")
            for field in ("seed", "temperature", "min_p", "generation_limits", "tokenizer_sha256"):
                if reports[0][field] != reports[1][field]:
                    raise ValueError(f"Paired settings disagree: {field}")
            groups.append(f'''<section data-group="{key}"><h2>{html.escape(prompt["description"])} · seed {seed}</h2>
<label>The shared starting idea · {protocol["prompt_seconds"]} seconds<audio controls preload="metadata" aria-label="{html.escape(key)} prompt" src="{link(Path(attempts[0]["directory"]) / "prompt.wav")}"></audio></label>
<div class="pair">{"".join(cards)}</div>
<label for="vote-{key}">Which continuation follows the idea more coherently?</label>
<select id="vote-{key}"><option value="">Choose after listening</option><option>A</option><option>B</option><option value="tie">About equal</option><option value="unclear">Neither / unclear</option></select>
<label for="notes-{key}">Optional listening notes</label><textarea id="notes-{key}" placeholder="Melody, harmony, phrasing, repetition, or the transition…"></textarea></section>''')
    title = html.escape(protocol.get("comparison_title", "Did more training help the melody?"))
    method = html.escape(protocol.get("comparison_description", "Additional next-token training."))
    baseline_tokens, candidate_tokens = training_tokens["baseline"], training_tokens["continued"]
    metadata = {
        "suite_sha256": sha256_file(suite_path),
        "suite": link(suite_path),
        "protocol": protocol,
        "mapping": mapping,
        "human_assessed": False,
    }
    data = json.dumps(metadata, ensure_ascii=True).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transformer melody comparison</title><style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f4ef;color:#182b32;font:16px/1.5 system-ui,sans-serif}}header{{background:#182b32;color:white;padding:20px max(24px,calc((100vw - 1060px)/2))}}main{{max-width:1060px;margin:auto;padding:32px 24px}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.15}}h2{{font-size:22px}}h3{{margin:0 0 14px}}p{{max-width:850px}}section{{margin:32px 0;padding:24px;background:white;border:1px solid #dce2df;border-radius:16px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}article{{padding:20px;background:#f5f8f4;border-radius:12px}}audio{{display:block;width:100%;margin:8px 0 12px}}label{{display:block;margin:14px 0 6px}}select,textarea,button{{font:inherit;border:1px solid #bdcbc1;border-radius:8px;padding:10px}}textarea{{width:100%;min-height:74px}}button{{background:#286650;color:white;cursor:pointer;margin:8px 10px 8px 0}}a{{color:#286650}}.small{{font-size:13px;color:#657379}}details{{font-size:14px}}.identity{{font-size:14px;font-weight:600}}#saved{{color:#286650}}@media(max-width:650px){{.pair{{grid-template-columns:1fr}}main{{padding:24px 16px}}section{{padding:18px}}}}
</style></head><body><header>Piano continuation lab · 8 October 2026</header><main>
<h1>{title}</h1>
<p>Listen to the shared starting idea, then versions A and B. Judge whether the melody and harmony follow the prompt, whether the phrasing makes sense, and whether repetition feels purposeful. Checkpoint names stay hidden until you reveal them; their order varies between pairs.</p>
<p class="small">Same 36.36M-parameter Transformer, context, prompt, seed, sampler and piano sound. The starting checkpoint has {baseline_tokens / 1e6:.2f}M training targets; the candidate has {candidate_tokens / 1e6:.2f}M in total. {method} These are {len(groups)} informal listening pairs, with no collected scores yet. Für Elise is a user test excerpt, not a held-out generalization test. Requested continuation: {protocol["new_seconds"]} seconds of new note onsets; temperature {protocol["temperature"]}, min-p {protocol["min_p"]}, remaining-context token budget.</p>
<button id="download">Download listening responses</button><button id="reveal">Reveal checkpoint names</button><span id="saved" role="status"></span>
{"".join(groups)}<p class="small">Responses are saved in this browser on this computer. Download them to record an assessment. This page does not send responses to a server. Revealing identities before voting can influence your preferences. Export can shorten repeated-pitch endings; full sample reports record each change. <a href="{link(suite_path)}">Protocol and technical records</a></p></main>
<script id="metadata" type="application/json">{data}</script><script>
const meta=JSON.parse(document.getElementById('metadata').textContent),storage='transformer-listening-'+meta.suite_sha256;
let answers={{}},revealed=false;try{{const stored=JSON.parse(localStorage.getItem(storage)||'{{}}');answers=stored.answers||{{}};revealed=Boolean(stored.revealed);}}catch{{}}
function save(){{try{{localStorage.setItem(storage,JSON.stringify({{answers,revealed}}));document.getElementById('saved').textContent='Saved in this browser';}}catch{{document.getElementById('saved').textContent='Download responses to keep them';}}}}
for(const group of document.querySelectorAll('[data-group]')){{const key=group.dataset.group,vote=document.getElementById('vote-'+key),notes=document.getElementById('notes-'+key);vote.value=answers[key]?.preference||'';notes.value=answers[key]?.notes||'';for(const el of [vote,notes])el.addEventListener('input',()=>{{answers[key]={{preference:vote.value,notes:notes.value,identities_revealed:revealed,updated:new Date().toISOString()}};save();}});}}
function reveal(){{revealed=true;for(const el of document.querySelectorAll('.identity')){{const item=meta.mapping[el.dataset.key][el.dataset.letter];el.hidden=false;el.textContent=item.label;}}document.getElementById('reveal').textContent='Checkpoint names revealed';save();}}
document.getElementById('reveal').onclick=reveal;if(revealed)reveal();
document.getElementById('download').onclick=()=>{{const result={{suite_sha256:meta.suite_sha256,exported:new Date().toISOString(),answers,mapping:meta.mapping,human_assessed:Object.values(answers).some(a=>a.preference)}};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{{type:'application/json'}}));const a=document.createElement('a');a.href=url;a.download='transformer-listening-responses.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
for(const audio of document.querySelectorAll('audio'))audio.addEventListener('play',()=>{{for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause();}});
</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/transformer-listening/index.html")
    )
    args = parser.parse_args()
    build(args.suite, args.output)


if __name__ == "__main__":
    main()
