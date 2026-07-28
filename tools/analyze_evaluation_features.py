from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from lyrehelper.analysis import (
    _chord_onset_metrics,
    _tempo_score_features,
    _timing_grid_statistics,
)
from lyrehelper.history import load_history_snapshot
from lyrehelper.models import AnalysisSnapshot, BeatMarker, NoteEvent, TempoPoint

from rebuild_evaluation_manifest import _apply_tempo_summary, _entry


_GRID_PHASES = np.asarray((0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0))
_RHYTHM_VALUES = np.asarray(
    (0.125, 1.0 / 6.0, 0.25, 1.0 / 3.0, 0.375, 0.5, 2.0 / 3.0, 0.75,
     1.0, 4.0 / 3.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    dtype=float,
)


def _onset_groups(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda item: (item.start, item.midi_note)):
        if groups and note.start - groups[-1][0].start <= 0.035:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def _group_times(groups: list[list[NoteEvent]]) -> np.ndarray:
    return np.asarray([np.median([note.start for note in group]) for group in groups])


def _beat_coordinates(
    times: np.ndarray, beats: list[BeatMarker]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beat_times = np.asarray(sorted({round(beat.time, 4) for beat in beats}), dtype=float)
    positions = np.full(len(times), np.nan, dtype=float)
    periods = np.full(len(times), np.nan, dtype=float)
    if len(beat_times) < 3:
        return positions, periods, np.zeros(len(times), dtype=bool)
    indices = np.searchsorted(beat_times, times, side="right") - 1
    valid = (indices >= 0) & (indices + 1 < len(beat_times))
    for index in np.flatnonzero(valid):
        beat_index = int(indices[index])
        period = beat_times[beat_index + 1] - beat_times[beat_index]
        if 0.2 <= period <= 2.0:
            periods[index] = period
            positions[index] = beat_index + (times[index] - beat_times[beat_index]) / period
        else:
            valid[index] = False
    valid &= np.isfinite(positions)
    return positions, periods, valid


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 4 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(left, right)[0, 1], nan=0.0))


def _weighted_group_residuals(
    grouped: dict[tuple, list[float]], minimum_repeats: int
) -> tuple[np.ndarray, int]:
    residuals: list[float] = []
    retained = 0
    for values in grouped.values():
        if len(values) < minimum_repeats:
            continue
        data = np.asarray(values, dtype=float)
        residuals.extend(np.abs(data - np.median(data)))
        retained += len(data)
    return np.asarray(residuals, dtype=float), retained


def _tempo_features(snapshot: AnalysisSnapshot) -> dict[str, float]:
    values = np.asarray([point.bpm for point in snapshot.tempo], dtype=float)
    average = max(snapshot.average_bpm, 1.0)
    base = _tempo_score_features(snapshot)
    output = {
        "tempo_step_median": base[0],
        "tempo_step_p90": base[1],
        "tempo_stable_ratio": base[2],
        "tempo_high_frequency": base[3],
        "tempo_low_frequency": base[4],
    }
    if len(values) < 8:
        output.update(
            tempo_smooth_motion=0.0,
            tempo_noise=0.0,
            tempo_motion_coherence=0.0,
            tempo_direction_persistence=0.5,
            tempo_segment_microvariation=0.0,
            tempo_stage_variation=0.0,
        )
        return output
    short_size = min(5, (len(values) // 2) * 2 - 1)
    long_size = min(21, (len(values) // 2) * 2 - 1)
    short = median_filter(values, size=short_size, mode="nearest")
    long = median_filter(values, size=long_size, mode="nearest")
    smooth_motion = float(np.std(short - long) / average)
    noise = float(np.std(values - short) / average)
    differences = np.diff(short) / average
    moving = np.abs(differences) >= 0.001
    moving_differences = differences[moving]
    persistence = (
        float(np.mean(np.sign(moving_differences[1:]) == np.sign(moving_differences[:-1])))
        if len(moving_differences) >= 3
        else 0.5
    )
    segments: list[np.ndarray] = []
    start = 0
    for index, change in enumerate(np.abs(np.diff(long)), start=1):
        if change > max(4.0, average * 0.04):
            if index - start >= 8:
                segments.append(values[start:index])
            start = index
    if len(values) - start >= 8:
        segments.append(values[start:])
    segment_micro = (
        float(np.median([np.std(segment) for segment in segments]) / average)
        if segments
        else 0.0
    )
    stage_std = [
        float(np.std(values[start : start + 15]) / average)
        for start in range(0, max(1, len(values) - 14), 8)
        if len(values[start : start + 15]) >= 8
    ]
    output.update(
        tempo_smooth_motion=smooth_motion,
        tempo_noise=noise,
        tempo_motion_coherence=smooth_motion / max(smooth_motion + noise, 1e-9),
        tempo_direction_persistence=persistence,
        tempo_segment_microvariation=segment_micro,
        tempo_stage_variation=float(np.subtract(*np.percentile(stage_std, (75, 25))))
        if len(stage_std) >= 2
        else 0.0,
    )
    return output


def _grid_features(
    groups: list[list[NoteEvent]], beats: list[BeatMarker]
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = _group_times(groups)
    positions, periods, valid = _beat_coordinates(times, beats)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 8:
        return {
            "grid_error_mad_beats": 0.0,
            "grid_error_p90_beats": 0.0,
            "grid_error_lag1": 0.0,
            "grid_error_allan_ratio": 0.0,
            "grid_error_low_frequency": 0.0,
            "grid_stage_median_iqr": 0.0,
            "grid_stage_mad_iqr": 0.0,
        }, times, positions, periods, valid
    phases = positions[valid] - np.floor(positions[valid])
    nearest = np.argmin(np.abs(phases[:, None] - _GRID_PHASES[None, :]), axis=1)
    errors = phases - _GRID_PHASES[nearest]
    errors -= float(np.median(errors))
    local_errors = np.zeros(len(errors), dtype=float)
    local_medians: list[float] = []
    local_mads: list[float] = []
    valid_times = times[valid]
    for index, time in enumerate(valid_times):
        nearby = errors[np.abs(valid_times - time) <= 5.0]
        center = float(np.median(nearby)) if len(nearby) >= 4 else 0.0
        local_errors[index] = errors[index] - center
    for start in np.arange(valid_times[0], max(valid_times[0], valid_times[-1] - 9.99), 5.0):
        local = errors[(valid_times >= start) & (valid_times < start + 10.0)]
        if len(local) < 8:
            continue
        local_medians.append(float(np.median(local)))
        local_mads.append(float(np.median(np.abs(local - np.median(local)))))
    low_frequency = float(np.std(local_medians)) if len(local_medians) >= 2 else 0.0
    median_iqr = (
        float(np.subtract(*np.percentile(local_medians, (75, 25))))
        if len(local_medians) >= 2
        else 0.0
    )
    mad_iqr = (
        float(np.subtract(*np.percentile(local_mads, (75, 25))))
        if len(local_mads) >= 2
        else 0.0
    )
    allan = np.median(np.abs(np.diff(local_errors))) if len(local_errors) >= 2 else 0.0
    return {
        "grid_error_mad_beats": float(np.median(np.abs(local_errors))),
        "grid_error_p90_beats": float(np.percentile(np.abs(local_errors), 90)),
        "grid_error_lag1": _safe_correlation(local_errors[:-1], local_errors[1:]),
        "grid_error_allan_ratio": float(allan / max(np.median(np.abs(local_errors)), 1e-6)),
        "grid_error_low_frequency": low_frequency,
        "grid_stage_median_iqr": median_iqr,
        "grid_stage_mad_iqr": mad_iqr,
    }, times, positions, periods, valid


def _repetition_features(
    groups: list[list[NoteEvent]],
    times: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    indices = np.flatnonzero(valid)
    if len(indices) < 12:
        return {
            "transition_repeat_mad": 0.0,
            "transition_repeat_p75": 0.0,
            "transition_repeat_coverage": 0.0,
            "phrase_repeat_mad": 0.0,
            "phrase_repeat_coverage": 0.0,
            "pitch_grid_repeat_mad": 0.0,
            "pitch_grid_repeat_coverage": 0.0,
        }
    valid_groups = [groups[index] for index in indices]
    valid_positions = positions[valid]
    tops = np.asarray([max(note.midi_note for note in group) for group in valid_groups])
    beat_intervals = np.diff(valid_positions)
    pitch_intervals = np.diff(tops)
    transition_groups: dict[tuple, list[float]] = defaultdict(list)
    for index, interval in enumerate(beat_intervals):
        if not 0.08 <= interval <= 8.5:
            continue
        rhythm_index = int(np.argmin(np.abs(_RHYTHM_VALUES - interval)))
        pitch_interval = int(np.clip(pitch_intervals[index], -24, 24))
        transition_groups[(pitch_interval, rhythm_index)].append(float(interval))
    transition_residuals, transition_retained = _weighted_group_residuals(
        transition_groups, 5
    )

    phrase_groups: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for index in range(len(beat_intervals) - 2):
        rhythm = beat_intervals[index : index + 3]
        if np.any((rhythm < 0.08) | (rhythm > 8.5)):
            continue
        rhythm_indices = tuple(
            int(np.argmin(np.abs(_RHYTHM_VALUES - value))) for value in rhythm
        )
        melodic = tuple(int(np.clip(value, -24, 24)) for value in pitch_intervals[index:index + 3])
        phrase_groups[(melodic, rhythm_indices)].append(rhythm)
    phrase_residuals: list[float] = []
    phrase_retained = 0
    for values in phrase_groups.values():
        if len(values) < 3:
            continue
        data = np.vstack(values)
        phrase_residuals.extend(np.abs(data - np.median(data, axis=0)).reshape(-1))
        phrase_retained += len(data)

    phase_groups: dict[tuple, list[float]] = defaultdict(list)
    phases = valid_positions - np.floor(valid_positions)
    nearest = np.argmin(np.abs(phases[:, None] - _GRID_PHASES[None, :]), axis=1)
    phase_errors = phases - _GRID_PHASES[nearest]
    for group, phase_index, error in zip(valid_groups, nearest, phase_errors):
        pitch_classes = tuple(sorted({note.midi_note % 12 for note in group}))
        phase_groups[(pitch_classes, int(phase_index))].append(float(error))
    pitch_residuals, pitch_retained = _weighted_group_residuals(phase_groups, 4)

    phrase_array = np.asarray(phrase_residuals, dtype=float)
    return {
        "transition_repeat_mad": float(np.median(transition_residuals))
        if len(transition_residuals)
        else 0.0,
        "transition_repeat_p75": float(np.percentile(transition_residuals, 75))
        if len(transition_residuals)
        else 0.0,
        "transition_repeat_coverage": transition_retained / max(1, len(beat_intervals)),
        "phrase_repeat_mad": float(np.median(phrase_array)) if len(phrase_array) else 0.0,
        "phrase_repeat_coverage": phrase_retained / max(1, len(beat_intervals) - 2),
        "pitch_grid_repeat_mad": float(np.median(pitch_residuals))
        if len(pitch_residuals)
        else 0.0,
        "pitch_grid_repeat_coverage": pitch_retained / max(1, len(valid_groups)),
    }


def _articulation_features(
    groups: list[list[NoteEvent]], periods: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    chord_spreads: list[float] = []
    chord_times: list[float] = []
    roll_linearity: list[float] = []
    pair_offsets: dict[tuple[int, int], list[float]] = defaultdict(list)
    chord_shape_spreads: dict[tuple[int, ...], list[float]] = defaultdict(list)
    pair_total = 0
    for index, group in enumerate(groups):
        if not valid[index] or len(group) < 2:
            continue
        period = periods[index]
        starts = np.asarray([note.start for note in group])
        spread = float((starts.max() - starts.min()) / period)
        chord_spreads.append(spread)
        chord_times.append(float(np.median(starts)))
        ordered = sorted(group, key=lambda note: note.midi_note)
        chord_shape_spreads[tuple(note.midi_note for note in ordered)].append(spread)
        if len(ordered) >= 3 and np.ptp(starts) > 1e-5:
            ordered_starts = np.asarray([note.start for note in ordered])
            roll_linearity.append(
                abs(_safe_correlation(np.arange(len(ordered), dtype=float), ordered_starts))
            )
        for left_index, left in enumerate(ordered[:-1]):
            for right in ordered[left_index + 1 :]:
                pair_offsets[(left.midi_note, right.midi_note)].append(
                    float((right.start - left.start) / period)
                )
                pair_total += 1
    pair_residuals, pair_retained = _weighted_group_residuals(pair_offsets, 4)
    shape_residuals, shape_retained = _weighted_group_residuals(chord_shape_spreads, 4)
    consistencies: list[float] = []
    for offsets in pair_offsets.values():
        if len(offsets) < 4:
            continue
        signs = np.sign(np.asarray(offsets))
        counts = [np.mean(signs < 0), np.mean(signs == 0), np.mean(signs > 0)]
        consistencies.append(float(max(counts)))
    spreads = np.asarray(chord_spreads, dtype=float)
    spread_times = np.asarray(chord_times, dtype=float)
    stage_medians: list[float] = []
    if len(spreads):
        for start in np.arange(spread_times[0], max(spread_times[0], spread_times[-1] - 19.99), 10.0):
            local = spreads[(spread_times >= start) & (spread_times < start + 20.0)]
            if len(local) >= 6:
                stage_medians.append(float(np.median(local)))
    return {
        "chord_spread_p90": float(np.percentile(spreads, 90)) if len(spreads) else 0.0,
        "chord_spread_iqr": float(np.subtract(*np.percentile(spreads, (75, 25))))
        if len(spreads) >= 2
        else 0.0,
        "chord_pair_repeat_mad": float(np.median(pair_residuals))
        if len(pair_residuals)
        else 0.0,
        "chord_pair_order_consistency": float(np.mean(consistencies))
        if consistencies
        else 0.5,
        "chord_pair_repeat_coverage": pair_retained / max(1, pair_total),
        "chord_pair_evidence_count": float(pair_retained),
        "chord_shape_repeat_mad": float(np.median(shape_residuals))
        if len(shape_residuals)
        else 0.0,
        "chord_shape_repeat_coverage": shape_retained / max(1, len(spreads)),
        "chord_spread_lag1": _safe_correlation(spreads[:-1], spreads[1:]),
        "chord_spread_stage_iqr": float(
            np.subtract(*np.percentile(stage_medians, (75, 25)))
        )
        if len(stage_medians) >= 2
        else 0.0,
        "chord_roll_linearity": float(np.median(roll_linearity))
        if roll_linearity
        else 0.0,
    }


def _duration_features(
    notes: list[NoteEvent], beats: list[BeatMarker]
) -> dict[str, float]:
    starts = np.asarray([note.start for note in notes])
    _, periods, valid = _beat_coordinates(starts, beats)
    grid_errors: list[float] = []
    repeat_groups: dict[tuple, list[float]] = defaultdict(list)
    for note, period, usable in zip(notes, periods, valid):
        if not usable:
            continue
        duration = (note.end - note.start) / period
        if not 0.05 <= duration <= 8.5:
            continue
        rhythm_index = int(np.argmin(np.abs(_RHYTHM_VALUES - duration)))
        grid_errors.append(float(abs(duration - _RHYTHM_VALUES[rhythm_index])))
        repeat_groups[(note.midi_note, rhythm_index)].append(float(duration))
    repeat_residuals, retained = _weighted_group_residuals(repeat_groups, 4)
    return {
        "duration_grid_mad": float(np.median(grid_errors)) if grid_errors else 0.0,
        "duration_repeat_mad": float(np.median(repeat_residuals))
        if len(repeat_residuals)
        else 0.0,
        "duration_repeat_coverage": retained / max(1, len(grid_errors)),
    }


def _load_rebuilt_snapshot(
    midi_path: Path, cache: dict
) -> AnalysisSnapshot:
    snapshot = load_history_snapshot(_entry(midi_path))
    cache_key = f"{midi_path.parent.name}/{midi_path.stem}"
    timing = cache["sessions"][cache_key]
    snapshot.tempo = [TempoPoint(*point) for point in timing["tempo"]]
    snapshot.beats = [BeatMarker(*beat) for beat in timing["beats"]]
    _apply_tempo_summary(snapshot, snapshot.tempo)
    return snapshot


def feature_row(midi_path: Path, label: str, cache: dict) -> dict[str, float | str]:
    snapshot = _load_rebuilt_snapshot(midi_path, cache)
    groups = _onset_groups(snapshot.notes)
    grid, times, positions, periods, valid = _grid_features(groups, snapshot.beats)
    chord = _chord_onset_metrics(snapshot.notes)
    grid_accuracy, timing_ms, local_ms = _timing_grid_statistics(
        snapshot.notes, snapshot.beats
    )
    row: dict[str, float | str] = {
        "session_id": snapshot.session_id or midi_path.stem,
        "label": label,
        "duration": snapshot.elapsed,
        "note_count": float(len(snapshot.notes)),
        "onset_count": float(len(groups)),
        "average_bpm": snapshot.average_bpm,
        "bpm_std_relative": snapshot.bpm_std / max(snapshot.average_bpm, 1.0),
        "grid_accuracy": float(grid_accuracy or 0.0) / 100.0,
        "timing_rms_beats": timing_ms * snapshot.average_bpm / 60_000.0,
        "local_timing_beats": (local_ms or 0.0) * snapshot.average_bpm / 60_000.0,
    }
    if chord is None:
        row.update(
            chord_exact_ratio=0.0,
            chord_polyphonic_ratio=0.0,
            chord_median_spread_beats=0.0,
            chord_evidence_available=0.0,
        )
    else:
        exact, polyphonic, spread = chord
        row.update(
            chord_exact_ratio=exact / 100.0,
            chord_polyphonic_ratio=polyphonic / 100.0,
            chord_median_spread_beats=spread * snapshot.average_bpm / 60.0,
            chord_evidence_available=1.0,
        )
    row.update(_tempo_features(snapshot))
    row.update(grid)
    row.update(_repetition_features(groups, times, positions, valid))
    row.update(_articulation_features(groups, periods, valid))
    row.update(_duration_features(snapshot.notes, snapshot.beats))
    return row


def analyze(root: Path, output: Path) -> Path:
    cache = json.loads((root / "timing-rebuild-cache.json").read_text(encoding="utf-8"))
    rows: list[dict[str, float | str]] = []
    for directory, label in (("human", "human"), ("non_human", "non_human")):
        for midi_path in sorted((root / directory).rglob("*_transcription.mid")):
            rows.append(feature_row(midi_path, label, cache))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timing-only evaluation features")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("artifacts/evaluation-samples"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation-samples/feature-analysis.csv"),
    )
    arguments = parser.parse_args()
    print(analyze(arguments.root.resolve(), arguments.output.resolve()))


if __name__ == "__main__":
    main()
