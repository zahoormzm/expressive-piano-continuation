import wave

import mido
import numpy as np

from music_project.audio.preview import render_preview


def test_preview_is_non_silent_has_headroom_and_release_tail(midi_builder, tmp_path):
    midi = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (500, mido.Message("note_off", note=60)),
        ]
    )
    output = tmp_path / "preview.wav"
    report = render_preview(midi, output)
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == 22050
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert np.max(np.abs(pcm.astype(np.int32))) < 32767
    assert report["seconds"] == 0.8
    assert report["rms"] > 0
    assert np.any(pcm[11025:])
