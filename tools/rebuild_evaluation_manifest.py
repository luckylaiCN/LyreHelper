from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from lyrehelper.analysis import (
    _beat_track,
    _constant_tempo_fit,
    _fallback_dynamic_beats,
    _fold_tempo,
    _note_onset_envelope,
    _repeated_tempo_levels,
    _segment_tempi_consistent,
    _tempo_track,
    _timing_grid_errors,
    _timing_grid_statistics,
    update_performance_score,
)
from lyrehelper.history import HistoryEntry, load_history_snapshot
from lyrehelper.models import BeatMarker, NoteEvent, TempoPoint
from lyrehelper.pipeline import (
    _ANALYSIS_CONTEXT_SECONDS,
    _METER_CHANGE_CONFIRMATIONS,
    _METER_INITIAL_CONFIRMATIONS,
    _apply_tempo_summary,
    _detected_meter,
    _merge_incremental_beats,
    _merge_incremental_tempo,
    _repair_short_tempo_excursions,
    _viterbi_tempo,
)


FIELDS = (
    "session_id",
    "label",
    "duration_s",
    "notes",
    "onsets",
    "onsets_per_s",
    "bpm_average",
    "bpm_std",
    "grid_accuracy",
    "timing_rms_ms",
    "current_human_score",
    "error_median_ms",
    "error_p75_ms",
    "error_p90_ms",
    "within_10ms_pct",
    "window_median_error_ms",
    "bpm_high_frequency_std",
    "timing_rebuild",
    "user_label_source",
)

_CACHE_NAME = "timing-rebuild-cache.json"
_SAMPLE_RATE = 22_050
_DEFAULT_ANALYSIS_STEP = 1.5


def _entry(midi_path: Path) -> HistoryEntry:
    session_id = midi_path.name.removesuffix("_transcription.mid")
    return HistoryEntry(
        session_id,
        midi_path,
        midi_path.parent / f"{session_id}_chords.csv",
        midi_path.parent / f"{session_id}_audio.wav",
        datetime.fromtimestamp(midi_path.stat().st_mtime),
    )


def _onset_count(notes: list) -> int:
    groups: list[list[float]] = []
    for start in sorted(note.start for note in notes):
        if groups and start - groups[-1][0] <= 0.035:
            groups[-1].append(start)
        else:
            groups.append([start])
    return len(groups)


def _algorithm_fingerprint(analysis_step: float) -> str:
    digest = hashlib.sha256()
    timing_functions = (
        _note_onset_envelope,
        _fold_tempo,
        _constant_tempo_fit,
        _segment_tempi_consistent,
        _repeated_tempo_levels,
        _tempo_track,
        _fallback_dynamic_beats,
        _beat_track,
        _viterbi_tempo,
        _repair_short_tempo_excursions,
        _merge_incremental_tempo,
        _detected_meter,
        _merge_incremental_beats,
        _resolve_meter,
        _visible_notes,
        _analyze_recent_timing,
        _rebuild_incremental_timing,
    )
    for function in timing_functions:
        digest.update(inspect.getsource(function).encode("utf-8"))
    digest.update(f"sample_rate={_SAMPLE_RATE};step={analysis_step:.6f}".encode("ascii"))
    return digest.hexdigest()


def _load_cache(path: Path, fingerprint: str) -> dict[str, Any]:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        cache = {}
    if cache.get("algorithm_fingerprint") != fingerprint:
        return {"algorithm_fingerprint": fingerprint, "sessions": {}}
    if not isinstance(cache.get("sessions"), dict):
        cache["sessions"] = {}
    return cache


def _resolve_meter(
    detected: int | None,
    stable: int | None,
    candidate: int | None,
    candidate_runs: int,
) -> tuple[int, bool, int | None, int | None, int]:
    if detected not in {3, 4, 6}:
        return stable or candidate or 4, False, stable, candidate, candidate_runs
    if stable is None:
        if detected == candidate:
            candidate_runs += 1
        else:
            candidate = detected
            candidate_runs = 1
        if candidate_runs >= _METER_INITIAL_CONFIRMATIONS:
            stable = detected
            candidate = None
            candidate_runs = 0
        return stable or detected, False, stable, candidate, candidate_runs
    if detected == stable:
        return stable, False, stable, None, 0
    if detected == candidate:
        candidate_runs += 1
    else:
        candidate = detected
        candidate_runs = 1
    if candidate_runs < _METER_CHANGE_CONFIRMATIONS:
        return stable, False, stable, candidate, candidate_runs
    return detected, True, detected, None, 0


def _visible_notes(notes: list[NoteEvent], elapsed: float) -> list[NoteEvent]:
    return [
        NoteEvent(
            note.start,
            min(note.end, elapsed),
            note.midi_note,
            note.frequency,
            note.name,
            note.velocity,
            note.confidence,
        )
        for note in notes
        if note.start <= elapsed and note.end > note.start
    ]


def _analyze_recent_timing(
    notes: list[NoteEvent], duration: float
) -> tuple[list[TempoPoint], list[BeatMarker]]:
    context_start = max(0.0, duration - _ANALYSIS_CONTEXT_SECONDS)
    local_notes = [
        NoteEvent(
            max(0.0, note.start - context_start),
            max(0.0, note.end - context_start),
            note.midi_note,
            note.frequency,
            note.name,
            note.velocity,
            note.confidence,
        )
        for note in notes
        if note.end > context_start
    ]
    local_duration = duration - context_start
    onset = _note_onset_envelope(local_notes, local_duration, _SAMPLE_RATE)
    local_tempo, _ = _tempo_track(onset, local_notes, _SAMPLE_RATE)
    local_beats = _beat_track(onset, _SAMPLE_RATE, local_tempo, local_notes)
    tempo = [
        TempoPoint(point.time + context_start, point.bpm, point.confidence)
        for point in local_tempo
    ]
    beats = [
        BeatMarker(
            beat.time + context_start,
            beat.strength,
            beat.is_downbeat,
            beat.beat_in_bar,
        )
        for beat in local_beats
    ]
    return tempo, beats


def _rebuild_incremental_timing(
    notes: list[NoteEvent],
    duration: float,
    analysis_step: float,
) -> tuple[list[TempoPoint], list[BeatMarker]]:
    tempo: list[TempoPoint] = []
    beats: list[BeatMarker] = []
    stable_meter: int | None = None
    meter_candidate: int | None = None
    meter_candidate_runs = 0
    refreshes = list(np.arange(analysis_step, duration, analysis_step, dtype=float))
    refreshes.append(duration)
    for elapsed in refreshes:
        detected_tempo, detected_beats = _analyze_recent_timing(
            _visible_notes(notes, float(elapsed)), float(elapsed)
        )
        tempo = _merge_incremental_tempo(tempo, detected_tempo, float(elapsed))
        meter, changed, stable_meter, meter_candidate, meter_candidate_runs = _resolve_meter(
            _detected_meter(detected_beats),
            stable_meter,
            meter_candidate,
            meter_candidate_runs,
        )
        beats = _merge_incremental_beats(
            beats,
            detected_beats,
            float(elapsed),
            meter,
            align_to_existing=not changed,
        )
    return tempo, beats


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _timing_from_cache(
    midi_path: Path,
    notes: list[NoteEvent],
    duration: float,
    analysis_step: float,
    cache: dict[str, Any],
) -> tuple[list[TempoPoint], list[BeatMarker], bool]:
    stat = midi_path.stat()
    sessions = cache["sessions"]
    cache_key = f"{midi_path.parent.name}/{midi_path.stem}"
    stored = sessions.get(cache_key)
    if (
        isinstance(stored, dict)
        and stored.get("size") == stat.st_size
        and stored.get("mtime_ns") == stat.st_mtime_ns
    ):
        tempo = [TempoPoint(*point) for point in stored.get("tempo", [])]
        beats = [BeatMarker(*beat) for beat in stored.get("beats", [])]
        return tempo, beats, True
    tempo, beats = _rebuild_incremental_timing(notes, duration, analysis_step)
    sessions[cache_key] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "tempo": [[point.time, point.bpm, point.confidence] for point in tempo],
        "beats": [
            [beat.time, beat.strength, beat.is_downbeat, beat.beat_in_bar]
            for beat in beats
        ],
    }
    return tempo, beats, False


def _feature_row(
    midi_path: Path,
    label: str,
    source: str,
    analysis_step: float,
    cache: dict[str, Any],
) -> tuple[dict[str, str | int], bool]:
    snapshot = load_history_snapshot(_entry(midi_path))
    rebuilt_tempo, rebuilt_beats, cache_hit = _timing_from_cache(
        midi_path,
        snapshot.notes,
        snapshot.elapsed,
        analysis_step,
        cache,
    )
    _apply_tempo_summary(snapshot, rebuilt_tempo)
    snapshot.beats = rebuilt_beats
    update_performance_score(snapshot)
    _, _, errors = _timing_grid_errors(snapshot.notes, snapshot.beats)
    if len(errors):
        errors -= float(np.median(errors))
        absolute_errors = np.abs(errors) * 1000.0
        error_median = float(np.median(absolute_errors))
        error_p75 = float(np.percentile(absolute_errors, 75))
        error_p90 = float(np.percentile(absolute_errors, 90))
        within_10ms = float(100.0 * np.mean(absolute_errors <= 10.0))
    else:
        error_median = error_p75 = error_p90 = within_10ms = 0.0
    _, _, local_error_ms = _timing_grid_statistics(snapshot.notes, snapshot.beats)
    bpm_values = np.asarray([point.bpm for point in snapshot.tempo], dtype=float)
    if len(bpm_values) >= 5:
        size = min(11, (len(bpm_values) // 2) * 2 - 1)
        baseline = median_filter(bpm_values, size=size, mode="nearest")
        high_frequency_std = float(np.std(bpm_values - baseline))
    else:
        high_frequency_std = 0.0
    onset_count = _onset_count(snapshot.notes)
    duration = max(snapshot.elapsed, 1e-9)
    return {
        "session_id": snapshot.session_id or midi_path.stem,
        "label": label,
        "duration_s": f"{snapshot.elapsed:.1f}",
        "notes": len(snapshot.notes),
        "onsets": onset_count,
        "onsets_per_s": f"{onset_count / duration:.2f}",
        "bpm_average": f"{snapshot.average_bpm:.2f}",
        "bpm_std": f"{snapshot.bpm_std:.3f}",
        "grid_accuracy": f"{snapshot.grid_accuracy:.1f}",
        "timing_rms_ms": f"{snapshot.timing_deviation_ms:.2f}",
        "current_human_score": f"{snapshot.human_score:.1f}",
        "error_median_ms": f"{error_median:.2f}",
        "error_p75_ms": f"{error_p75:.2f}",
        "error_p90_ms": f"{error_p90:.2f}",
        "within_10ms_pct": f"{within_10ms:.1f}",
        "window_median_error_ms": f"{(local_error_ms or 0.0):.2f}",
        "bpm_high_frequency_std": f"{high_frequency_std:.3f}",
        "timing_rebuild": "midi-note-window-simulation",
        "user_label_source": source,
    }, cache_hit


def rebuild(root: Path, analysis_step: float = _DEFAULT_ANALYSIS_STEP) -> Path:
    if analysis_step <= 0:
        raise ValueError("analysis_step must be positive")
    cache_path = root / _CACHE_NAME
    cache = _load_cache(cache_path, _algorithm_fingerprint(analysis_step))
    rows: list[dict[str, str | int]] = []
    layouts = (
        ("human", "human", "user"),
        ("non_human", "non_human", "user"),
        ("uncertain", "human_candidate", "user_uncertain"),
    )
    midi_entries = [
        (midi_path, label, source)
        for directory, label, source in layouts
        for midi_path in sorted((root / directory).rglob("*_transcription.mid"))
    ]
    for index, (midi_path, label, source) in enumerate(midi_entries, start=1):
        row_source = "floating_panel" if midi_path.name.startswith("20260723_") else source
        row, cache_hit = _feature_row(
            midi_path,
            label,
            row_source,
            analysis_step,
            cache,
        )
        rows.append(row)
        state = "cached" if cache_hit else "rebuilt"
        print(f"[{index:02d}/{len(midi_entries):02d}] {midi_path.stem}: {state}")
        if not cache_hit:
            _save_cache(cache_path, cache)
    destination = root / "manifest.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _save_cache(cache_path, cache)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the protected score corpus manifest")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("artifacts/evaluation-samples"),
    )
    parser.add_argument(
        "--analysis-step",
        type=float,
        default=_DEFAULT_ANALYSIS_STEP,
        help="Seconds between simulated realtime refreshes (default: 1.5)",
    )
    arguments = parser.parse_args()
    print(rebuild(arguments.root.resolve(), arguments.analysis_step))


if __name__ == "__main__":
    main()
