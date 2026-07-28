from __future__ import annotations

import csv
import io
import os
import wave

import mido
import numpy as np

from lyrehelper.exporters import build_tempo_midi, export_audio, export_session, prune_archives
from lyrehelper.models import (
    AnalysisResult,
    BeatMarker,
    ChordSegment,
    KeySegment,
    NoteEvent,
    TempoPoint,
)


def result_fixture() -> AnalysisResult:
    return AnalysisResult(
        duration=4.0,
        tempo=[TempoPoint(0, 120), TempoPoint(2, 123)],
        beats=[BeatMarker(0, 1, True, 1), BeatMarker(0.5, 0.7, False, 2)],
        chords=[ChordSegment(0, 2, "C", "C major", "I", 0.9)],
        keys=[KeySegment(0, 4, "C major", 0.8)],
        average_bpm=121.5,
        min_bpm=120,
        max_bpm=123,
        bpm_std=1.5,
        human_score=25,
        mechanical_index=75,
        mode="rhythmic",
        notes=[NoteEvent(0.25, 1.25, 60, 261.63, "C4", 96, 0.9)],
    )


def test_midi_contains_valid_header_and_tempo_events() -> None:
    midi = build_tempo_midi(result_fixture())
    assert midi[:4] == b"MThd"
    assert b"MTrk" in midi
    assert midi.count(b"\xff\x51\x03") == 2
    assert b"DOWNBEAT" in midi
    assert bytes((0x90, 60, 96)) in midi
    assert bytes((0x80, 60, 0)) in midi
    parsed = mido.MidiFile(file=io.BytesIO(midi))
    messages = [message for track in parsed.tracks for message in track]
    assert sum(message.type == "note_on" and message.velocity > 0 for message in messages) == 1
    assert sum(message.type == "note_off" for message in messages) == 1


def test_midi_seeds_tick_zero_when_analysis_starts_later() -> None:
    result = result_fixture()
    result.tempo = [TempoPoint(8.0, 96), TempoPoint(10.0, 98)]
    midi = build_tempo_midi(result)
    assert midi.count(b"\xff\x51\x03") == 3


def test_export_session_writes_two_standard_files(tmp_path) -> None:
    midi_path, csv_path = export_session(result_fixture(), tmp_path, "session")
    assert midi_path.name == "session_transcription.mid"
    assert csv_path.name == "session_chords.csv"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "session_chords.csv",
        "session_transcription.mid",
    ]
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["start_time", "end_time", "chord_name", "key"],
        ["0.000", "2.000", "C", "C major"],
    ]


def test_export_audio_writes_pcm_wave(tmp_path) -> None:
    audio_path = export_audio(np.array([-1.0, 0.0, 1.0], dtype=np.float32), 22050, tmp_path, "session")

    with wave.open(str(audio_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 22050
        assert handle.getnframes() == 3


def test_prune_archives_keeps_latest_ten_sessions_as_file_groups(tmp_path) -> None:
    for index in range(12):
        session = f"20260722_1200{index:02d}_001"
        paths = (
            tmp_path / f"{session}_audio.wav",
            tmp_path / f"{session}_transcription.mid",
            tmp_path / f"{session}_chords.csv",
        )
        for path in paths:
            path.write_bytes(b"archive")
            os.utime(path, (index + 1, index + 1))
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="ascii")

    removed = prune_archives(tmp_path, keep=10)

    assert removed == ["20260722_120001_001", "20260722_120000_001"]
    assert unrelated.exists()
    assert len(list(tmp_path.glob("*_audio.wav"))) == 10
    assert len(list(tmp_path.glob("*_transcription.mid"))) == 10
    assert len(list(tmp_path.glob("*_chords.csv"))) == 10
