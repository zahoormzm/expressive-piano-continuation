"""Acquire pinned local rendering assets and, optionally, Aria generation weights."""

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.data.download import safe_extract
from music_project.io import sha256_file, write_json


def acquire(asset):
    target = Path(asset["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(target.name + ".part")
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "music-project/0.1"})
        print(f"Downloading {target.name}", flush=True)
        downloaded, last_report = 0, time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as output:
            while chunk := source.read(4 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 15:
                    print(f"  {downloaded / 1e6:.0f} MB downloaded", flush=True)
                    last_report = time.monotonic()
        if sha256_file(temporary) != asset["sha256"]:
            raise ValueError(f"Checksum mismatch for {temporary}; inspect before retry")
        temporary.replace(target)
    if sha256_file(target) != asset["sha256"]:
        raise ValueError(f"Existing asset checksum mismatch: {target}")
    print(f"Verified {target.name}", flush=True)
    return {"path": target.as_posix(), "sha256": asset["sha256"], "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained", action="store_true", help="Also download 2.63 GB of weights"
    )
    args = parser.parse_args()
    assets = json.loads(Path("configs/assets.json").read_text())
    report = {}
    for name in ["fluidsynth", "soundfont"] + (["pretrained"] if args.pretrained else []):
        report[name] = acquire(assets[name])
    renderer = Path(assets["fluidsynth"]["directory"])
    safe_extract(Path(assets["fluidsynth"]["path"]), renderer)
    executables = list(renderer.rglob("fluidsynth.exe"))
    if len(executables) != 1:
        raise ValueError("Unexpected FluidSynth archive layout")
    report["fluidsynth"]["executable"] = executables[0].as_posix()
    soundfont_license = Path("data/assets/GeneralUser-GS-LICENSE.txt")
    if not soundfont_license.exists():
        with urllib.request.urlopen(assets["soundfont"]["license_url"], timeout=30) as source:
            soundfont_license.write_bytes(source.read())
    if args.pretrained:
        asset = assets["pretrained"]
        source = Path(asset["source_directory"])
        if not source.exists():
            subprocess.run(["git", "clone", asset["source_repository"], str(source)], check=True)
            subprocess.run(
                ["git", "-C", str(source), "checkout", asset["source_revision"]], check=True
            )
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        )
        if revision.strip() != asset["source_revision"]:
            raise ValueError("Aria source revision disagrees with asset registry")
        report["pretrained"]["source_revision"] = revision.strip()
    write_json(Path("outputs/asset-setup.json"), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
