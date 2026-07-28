from __future__ import annotations

import csv
import math
import wave
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mido
import numpy as np

from .analysis import chord_function, tempo_standard_deviation, update_performance_score
from .models import (
    AnalysisSnapshot,
    BeatMarker,
    ChordSegment,
    KeySegment,
    MonitorState,
    NoteEvent,
    TempoPoint,
)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    session_id: str
    midi_path: Path
    chord_path: Path | None
    audio_path: Path | None
    modified: datetime


def list_history(output_directory: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    if not output_directory.exists():
        return entries
    midi_paths = list(output_directory.glob("*_transcription.mid"))
    midi_paths.extend(output_directory.glob("*_tempo.mid"))
    for midi_path in midi_paths:
        suffix = "_transcription.mid" if midi_path.name.endswith("_transcription.mid") else "_tempo.mid"
        session_id = midi_path.name.removesuffix(suffix)
        chord_path = output_directory / f"{session_id}_chords.csv"
        audio_path = output_directory / f"{session_id}_audio.wav"
        entries.append(
            HistoryEntry(
                session_id,
                midi_path,
                chord_path if chord_path.exists() else None,
                audio_path if audio_path.exists() else None,
                datetime.fromtimestamp(midi_path.stat().st_mtime),
            )
        )
    return sorted(entries, key=lambda item: item.modified, reverse=True)


def _note_name(midi_note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def _audio_duration(path: Path | None) -> float:
    if path is None:
        return 0.0
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / max(1, handle.getframerate())
    except (OSError, wave.Error):
        return 0.0


def _load_chords(path: Path | None) -> tuple[list[ChordSegment], list[KeySegment]]:
    if path is None:
        return [], []
    chords: list[ChordSegment] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                chords.append(
                    ChordSegment(
                        float(row["start_time"]),
                        float(row["end_time"]),
                        row["chord_name"],
                        row["key"],
                        chord_function(row["chord_name"], row["key"]),
                    )
                )
    except (OSError, KeyError, TypeError, ValueError):
        return [], []
    keys: list[KeySegment] = []
    for chord in chords:
        if keys and keys[-1].key == chord.key and chord.start <= keys[-1].end + 0.01:
            previous = keys[-1]
            keys[-1] = KeySegment(previous.start, chord.end, previous.key, previous.confidence)
        else:
            keys.append(KeySegment(chord.start, chord.end, chord.key, 0.7))
    return chords, keys


def load_history_snapshot(entry: HistoryEntry) -> AnalysisSnapshot:
    midi = mido.MidiFile(entry.midi_path)
    tempo = 500_000
    elapsed = 0.0
    tempo_points: list[TempoPoint] = []
    beats: list[BeatMarker] = []
    notes: list[NoteEvent] = []
    active: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            tempo_points.append(TempoPoint(elapsed, float(mido.tempo2bpm(tempo)), 1.0))
        elif message.type == "marker":
            label = message.text.upper()
            if label.startswith("DOWNBEAT") or label.startswith("BEAT"):
                parts = label.split()
                beat_number = int(parts[-1]) if parts[-1].isdigit() else 0
                beats.append(BeatMarker(elapsed, 1.0, label.startswith("DOWNBEAT"), beat_number))
        elif message.type == "note_on" and message.velocity > 0:
            active[message.note].append((elapsed, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            if active[message.note]:
                start, velocity = active[message.note].pop(0)
                notes.append(
                    NoteEvent(
                        start,
                        elapsed,
                        message.note,
                        440.0 * 2.0 ** ((message.note - 69) / 12.0),
                        _note_name(message.note),
                        velocity,
                        1.0,
                    )
                )
    duration = max(float(midi.length), _audio_duration(entry.audio_path), elapsed)
    for midi_note, pending in active.items():
        for start, velocity in pending:
            notes.append(
                NoteEvent(
                    start,
                    duration,
                    midi_note,
                    440.0 * 2.0 ** ((midi_note - 69) / 12.0),
                    _note_name(midi_note),
                    velocity,
                    1.0,
                )
            )
    chords, keys = _load_chords(entry.chord_path)
    bpms = np.asarray([point.bpm for point in tempo_points], dtype=float)
    average = float(bpms.mean()) if len(bpms) else 0.0
    deviation = tempo_standard_deviation(tempo_points)
    snapshot = AnalysisSnapshot(
        state=MonitorState.STANDBY,
        device_name="Archived session",
        quality="archived",
        mode="history",
        session_id=entry.session_id,
        elapsed=duration,
        playhead=0.0,
        notes=sorted(notes, key=lambda item: (item.start, item.midi_note, item.end)),
        tempo=tempo_points,
        beats=beats,
        chords=chords,
        keys=keys,
        average_bpm=average,
        min_bpm=float(bpms.min()) if len(bpms) else 0.0,
        max_bpm=float(bpms.max()) if len(bpms) else 0.0,
        bpm_std=0.0 if math.isclose(deviation, 0.0, abs_tol=1e-9) else deviation,
        last_archive=str(entry.midi_path.parent / entry.session_id),
    )
    update_performance_score(snapshot)
    return snapshot
