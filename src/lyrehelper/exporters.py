from __future__ import annotations

import csv
import os
import struct
import wave
from pathlib import Path

import numpy as np

from .models import AnalysisResult, TempoPoint

ARCHIVE_SUFFIXES = (
    "_transcription.mid",
    "_tempo.mid",
    "_chords.csv",
    "_audio.wav",
)


def _variable_length(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 0x7F
    result = bytearray([buffer])
    while value >> 7:
        value >>= 7
        buffer = (value & 0x7F) | 0x80
        result.insert(0, buffer)
    return bytes(result)


def _meta(delta: int, kind: int, payload: bytes) -> bytes:
    return _variable_length(delta) + bytes((0xFF, kind)) + _variable_length(len(payload)) + payload


def _tempo_at(time: float, result: AnalysisResult) -> float:
    points = [point for point in result.tempo if point.time <= time]
    if points:
        return max(1.0, points[-1].bpm)
    return max(1.0, result.average_bpm or 120.0)


def _seconds_to_ticks(time: float, result: AnalysisResult, ppq: int) -> int:
    if not result.tempo:
        return round(time * (result.average_bpm or 120.0) * ppq / 60.0)
    points = sorted(result.tempo, key=lambda item: item.time)
    ticks = 0.0
    cursor = 0.0
    bpm = points[0].bpm
    for point in points[1:]:
        if point.time >= time:
            break
        ticks += (point.time - cursor) * bpm * ppq / 60.0
        cursor = point.time
        bpm = point.bpm
    ticks += max(0.0, time - cursor) * bpm * ppq / 60.0
    return round(ticks)


def build_transcription_midi(result: AnalysisResult, ppq: int = 480) -> bytes:
    events: list[tuple[int, int, bytes]] = []
    events.append((0, 0, _meta(0, 0x03, b"LyreHelper transcription")))
    events.append((0, 0, _meta(0, 0x58, bytes((4, 2, 24, 8)))))
    if result.tempo:
        tempo_points = list(result.tempo)
        if tempo_points[0].time > 0:
            tempo_points.insert(0, TempoPoint(0.0, tempo_points[0].bpm, tempo_points[0].confidence))
        for point in tempo_points:
            tick = _seconds_to_ticks(point.time, result, ppq)
            microseconds = round(60_000_000 / max(point.bpm, 1.0))
            events.append((tick, 1, _meta(0, 0x51, microseconds.to_bytes(3, "big"))))
    else:
        bpm = result.average_bpm or 120.0
        events.append((0, 1, _meta(0, 0x51, round(60_000_000 / bpm).to_bytes(3, "big"))))
    for beat in result.beats:
        tick = _seconds_to_ticks(beat.time, result, ppq)
        label = f"DOWNBEAT {beat.beat_in_bar}" if beat.is_downbeat else f"BEAT {beat.beat_in_bar}"
        events.append((tick, 2, _meta(0, 0x06, label.encode("ascii"))))
    for note in result.notes:
        start_tick = _seconds_to_ticks(note.start, result, ppq)
        end_tick = max(start_tick + 1, _seconds_to_ticks(note.end, result, ppq))
        midi_note = int(max(0, min(127, note.midi_note)))
        velocity = int(max(1, min(127, note.velocity)))
        events.append((start_tick, 4, b"\x00" + bytes((0x90, midi_note, velocity))))
        events.append((end_tick, 3, b"\x00" + bytes((0x80, midi_note, 0))))
    events.sort(key=lambda item: (item[0], item[1]))
    track = bytearray()
    previous_tick = 0
    for tick, _, event in events:
        delta = tick - previous_tick
        # Replace the event's zero delta with the actual sorted delta.
        zero_delta_length = len(_variable_length(0))
        track.extend(_variable_length(delta))
        track.extend(event[zero_delta_length:])
        previous_tick = tick
    track.extend(_meta(0, 0x2F, b""))
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def build_tempo_midi(result: AnalysisResult, ppq: int = 480) -> bytes:
    """Backward-compatible name for callers from the tempo-map-only release."""
    return build_transcription_midi(result, ppq)


def export_session(result: AnalysisResult, output_dir: Path, session_id: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / f"{session_id}_transcription.mid"
    csv_path = output_dir / f"{session_id}_chords.csv"
    midi_temp = midi_path.with_suffix(".mid.tmp")
    csv_temp = csv_path.with_suffix(".csv.tmp")
    midi_temp.write_bytes(build_transcription_midi(result))
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("start_time", "end_time", "chord_name", "key"))
        for chord in result.chords:
            writer.writerow((f"{chord.start:.3f}", f"{chord.end:.3f}", chord.chord, chord.key))
    os.replace(midi_temp, midi_path)
    os.replace(csv_temp, csv_path)
    return midi_path, csv_path


def export_audio(audio: np.ndarray, sample_rate: int, output_dir: Path, session_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{session_id}_audio.wav"
    wav_temp = wav_path.with_suffix(".wav.tmp")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(wav_temp), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    os.replace(wav_temp, wav_path)
    return wav_path


def prune_archives(output_dir: Path, keep: int = 10) -> list[str]:
    if keep < 0:
        raise ValueError("keep must be non-negative")
    if not output_dir.exists():
        return []
    sessions: dict[str, list[Path]] = {}
    modified: dict[str, float] = {}
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        suffix = next((item for item in ARCHIVE_SUFFIXES if path.name.endswith(item)), None)
        if suffix is None:
            continue
        session_id = path.name.removesuffix(suffix)
        sessions.setdefault(session_id, []).append(path)
        modified[session_id] = max(modified.get(session_id, 0.0), path.stat().st_mtime)
    newest = sorted(sessions, key=lambda item: (modified[item], item), reverse=True)
    removed: list[str] = []
    for session_id in newest[keep:]:
        for path in sessions[session_id]:
            path.unlink(missing_ok=True)
        removed.append(session_id)
    return removed
