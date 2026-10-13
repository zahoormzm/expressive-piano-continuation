"""Simple harmonic diagnostic audio; final listening tests need a piano SoundFont."""

import wave
from pathlib import Path

import numpy as np

from music_project.data.midi import read_performance


def render_preview(midi: Path, wav: Path, seconds: float = 20.0, sample_rate: int = 22050) -> dict:
    if seconds <= 0 or sample_rate <= 0:
        raise ValueError("Positive duration and sample rate required")
    performance = read_performance(midi)
    start = min(note.start_seconds for note in performance.notes)
    duration = min(seconds, performance.duration_seconds - start)
    release = 0.3
    audio = np.zeros(round((duration + release) * sample_rate), dtype=np.float64)
    for note in performance.notes:
        onset = note.start_seconds - start
        if onset >= duration:
            continue
        note_duration = min(note.end_seconds - note.start_seconds, duration - onset)
        begin = max(0, round(onset * sample_rate))
        count = min(round((note_duration + release) * sample_rate), len(audio) - begin)
        if count <= 0:
            continue
        time = np.arange(count) / sample_rate
        envelope = np.minimum(time / 0.008, 1.0) * np.exp(-time / 1.8)
        envelope *= np.where(
            time <= note_duration, 1.0, np.maximum(0.0, 1 - (time - note_duration) / release)
        )
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        tone = np.zeros(count, dtype=np.float64)
        for harmonic, amplitude in ((1, 1.0), (2, 0.35), (3, 0.15), (4, 0.08)):
            if frequency * harmonic < sample_rate / 2:
                tone += amplitude * np.sin(2 * np.pi * frequency * harmonic * time)
        audio[begin : begin + count] += tone * envelope * (note.velocity / 127)
    peak = float(np.max(np.abs(audio)))
    if peak == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is silent or nonfinite")
    audio *= 0.9 / peak
    pcm = (audio * 32767).astype("<i2")
    wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return {
        "renderer": "diagnostic harmonic synthesis",
        "piano_soundfont": False,
        "sample_rate": sample_rate,
        "seconds": len(pcm) / sample_rate,
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "source_start_seconds": start,
        "release_tail_seconds": release,
    }
