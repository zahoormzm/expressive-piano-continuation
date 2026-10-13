"""Pinned FluidSynth fast rendering, with fixed settings and PCM signal checks."""

import json
import subprocess
import wave
import zipfile
from pathlib import Path

import numpy as np

from music_project.data.midi import read_performance
from music_project.io import sha256_file, write_json


def render_piano(
    midi: Path, wav: Path, sample_rate=44100, gain=0.25, registry=Path("configs/assets.json")
):
    if sample_rate not in (22050, 44100, 48000) or not 0 < gain <= 1:
        raise ValueError("Supported sample rate and gain in (0, 1] required")
    assets = json.loads(registry.read_text())
    renderer, soundfont = assets["fluidsynth"], assets["soundfont"]
    archive, directory = Path(renderer["path"]), Path(renderer["directory"])
    sf2 = Path(soundfont["path"])
    if sha256_file(archive) != renderer["sha256"] or sha256_file(sf2) != soundfont["sha256"]:
        raise ValueError("Rendering asset checksum mismatch; run setup_assets.py")
    # Verify extracted program and DLL bytes against the pinned archive.
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not entry.is_dir() and entry.filename.lower().endswith((".exe", ".dll")):
                path = (directory / entry.filename).resolve()
                if not path.is_relative_to(directory.resolve()) or path.read_bytes() != source.read(
                    entry
                ):
                    raise ValueError("Extracted renderer program/library changed")
    executables = list(directory.rglob("fluidsynth.exe"))
    if len(executables) != 1:
        raise ValueError("FluidSynth executable missing or ambiguous")
    performance = read_performance(midi)
    if performance.duration_seconds > 300:
        raise ValueError("Render a bounded MIDI excerpt of at most five minutes")
    if any(note.program != 0 or note.channel == 9 for note in performance.notes):
        raise ValueError("Comparison renderer requires acoustic grand piano program zero")
    wav.parent.mkdir(parents=True, exist_ok=True)
    if wav.exists():
        raise FileExistsError(f"Refusing to overwrite existing audio: {wav}")
    command = [
        str(executables[0].resolve()),
        "-ni",
        "-F",
        str(wav.resolve()),
        "-T",
        "wav",
        "-O",
        "s16",
        "-r",
        str(sample_rate),
        "-g",
        str(gain),
        "-R",
        "0",
        "-C",
        "0",
        "-o",
        "synth.polyphony=256",
        str(sf2.resolve()),
        str(midi.resolve()),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    with wave.open(str(wav), "rb") as source:
        if (
            source.getsampwidth() != 2
            or source.getnchannels() != 2
            or source.getframerate() != sample_rate
        ):
            raise ValueError("Renderer did not produce the expected stereo PCM format")
        frames = source.getnframes()
        pcm = np.frombuffer(source.readframes(frames), dtype="<i2")
    audio = pcm.astype(np.float64) / 32768
    peak = float(np.max(np.abs(audio)))
    if (
        peak == 0
        or frames / sample_rate < max(note.end_seconds for note in performance.notes) - 0.05
    ):
        raise ValueError("Piano audio is silent or ends before its notes")
    clipped = int(np.count_nonzero(np.abs(pcm.astype(np.int32)) >= 32760))
    if clipped:
        raise ValueError(
            "Piano audio clips at the fixed gain; reduce gain consistently for every comparison"
        )
    report = {
        "renderer": "FluidSynth",
        "version": renderer["version"],
        "piano_soundfont": True,
        "renderer_archive_sha256": renderer["sha256"],
        "soundfont": soundfont["version"],
        "soundfont_sha256": soundfont["sha256"],
        "input_sha256": sha256_file(midi),
        "wav_sha256": sha256_file(wav),
        "sample_rate": sample_rate,
        "channels": 2,
        "sample_format": "PCM16",
        "seconds": frames / sample_rate,
        "gain": gain,
        "reverb": False,
        "chorus": False,
        "polyphony": 256,
        "peak": peak,
        "rms": float(np.sqrt(np.mean(audio**2))),
        "clipped_samples": clipped,
        "log": (result.stdout + result.stderr).strip(),
    }
    write_json(wav.with_suffix(".json"), report)
    return report
