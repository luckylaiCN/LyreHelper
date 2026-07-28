from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import pytest

from lyrehelper.analysis import analyze_audio, transcribe_notes

SAMPLE_ROOT_VALUE = os.environ.get("LYREHELPER_TEST_SAMPLE_DIR")
SAMPLE_ROOT = Path(SAMPLE_ROOT_VALUE) if SAMPLE_ROOT_VALUE else None
pytestmark = pytest.mark.skipif(
    SAMPLE_ROOT is None or not SAMPLE_ROOT.is_dir(),
    reason="optional instrument sample library unavailable; set LYREHELPER_TEST_SAMPLE_DIR",
)


def test_real_instrument_note_has_no_harmonic_spray() -> None:
    assert SAMPLE_ROOT is not None
    audio, sample_rate = librosa.load(SAMPLE_ROOT / "C3.wav", sr=22050, mono=True)
    notes = transcribe_notes(audio, sample_rate)
    assert any(note.midi_note % 12 == 0 for note in notes)
    assert len(notes) <= 2
    assert all(note.velocity == 96 for note in notes)


def test_machine_timed_instrument_audio_locks_constant_bpm() -> None:
    assert SAMPLE_ROOT is not None
    sample_rate = 22050
    samples = [
        librosa.load(SAMPLE_ROOT / f"{name}.wav", sr=sample_rate, mono=True)[0]
        for name in ("C3", "D3", "E3", "G3")
    ]
    duration = 24.0
    audio = np.zeros(round(duration * sample_rate), dtype=np.float32)
    for index, start in enumerate(np.arange(0, duration, 0.5)):
        sample = samples[index % len(samples)]
        offset = round(start * sample_rate)
        length = min(len(sample), len(audio) - offset)
        audio[offset : offset + length] += sample[:length]
    audio /= np.max(np.abs(audio))
    result = analyze_audio(audio, sample_rate)
    assert 119.8 <= result.average_bpm <= 120.2
    assert result.max_bpm - result.min_bpm < 0.01
    assert result.human_score < 1.0
    assert result.grid_accuracy > 99.0
    assert result.timing_deviation_ms < 3.0
    assert 40 <= len(result.notes) <= 60
