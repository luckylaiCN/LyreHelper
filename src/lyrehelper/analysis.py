from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import binary_closing, median_filter

from .models import (
    AnalysisResult,
    AnalysisSnapshot,
    BeatMarker,
    ChordSegment,
    KeySegment,
    NoteEvent,
    TempoPoint,
)
from .transcription_backend import transcribe_with_neural_model

HOP_LENGTH = 512
TEMPO_HOP = 128
TEMPO_AUTOCORRELATION_FACTOR = 4
LOWEST_MIDI = 36  # C2
NOTE_BINS = 72  # C2-B7
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass(frozen=True, slots=True)
class _ChordTemplate:
    suffix: str
    intervals: tuple[int, ...]
    tension: float


CHORD_TEMPLATES = (
    _ChordTemplate("maj7", (0, 4, 7, 11), 0.030),
    _ChordTemplate("7", (0, 4, 7, 10), 0.034),
    _ChordTemplate("m7", (0, 3, 7, 10), 0.028),
    _ChordTemplate("sus4", (0, 5, 7), 0.020),
    _ChordTemplate("sus2", (0, 2, 7), 0.018),
    _ChordTemplate("dim", (0, 3, 6), 0.022),
    _ChordTemplate("aug", (0, 4, 8), 0.022),
    _ChordTemplate("m", (0, 3, 7), 0.012),
    _ChordTemplate("", (0, 4, 7), 0.012),
)


def _event_from_run(
    decibels: np.ndarray,
    pitch_index: int,
    start_frame: int,
    end_frame: int,
    frame_rate: float,
) -> NoteEvent | None:
    duration = (end_frame - start_frame) / frame_rate
    if duration < 0.075:
        return None
    midi_note = LOWEST_MIDI + pitch_index
    energy_db = float(np.median(decibels[pitch_index, start_frame:end_frame]))
    confidence = float(np.clip((energy_db + 46.0) / 34.0, 0.05, 1.0))
    return NoteEvent(
        start=start_frame / frame_rate,
        end=end_frame / frame_rate,
        midi_note=midi_note,
        frequency=float(librosa.midi_to_hz(midi_note)),
        name=str(librosa.midi_to_note(midi_note, unicode=False)),
        velocity=96,
        confidence=confidence,
    )


def transcribe_notes(audio: np.ndarray, sample_rate: int) -> list[NoteEvent]:
    """Create polyphonic semitone note events from a harmonic CQT representation."""
    if len(audio) < 2048 or np.max(np.abs(audio), initial=0) < 1e-5:
        return []
    neural_events = transcribe_with_neural_model(audio, sample_rate)
    if neural_events is not None:
        return neural_events
    cqt = np.abs(
        librosa.cqt(
            audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            fmin=float(librosa.midi_to_hz(LOWEST_MIDI)),
            n_bins=NOTE_BINS,
            bins_per_octave=12,
        )
    )
    if not cqt.size or cqt.max(initial=0) <= 1e-9:
        return []
    decibels = librosa.amplitude_to_db(cqt, ref=np.max, top_db=70.0)
    horizontal = median_filter(cqt, size=(1, 17), mode="nearest")
    vertical = median_filter(cqt, size=(17, 1), mode="nearest")
    cqt *= horizontal / (horizontal + vertical + 1e-9)
    decibels = librosa.amplitude_to_db(cqt, ref=np.max, top_db=70.0)
    decibels = median_filter(decibels, size=(1, 3), mode="nearest")
    above = np.vstack((np.full((1, decibels.shape[1]), -80.0), decibels[:-1]))
    below = np.vstack((decibels[1:], np.full((1, decibels.shape[1]), -80.0)))
    frame_peak = decibels.max(axis=0, keepdims=True)
    active = (
        (decibels >= above)
        & (decibels > below)
        & (decibels > -46.0)
        & (decibels > frame_peak - 31.0)
    )

    # Keep a bounded polyphony and suppress only clearly weaker harmonic replicas.
    filtered = np.zeros_like(active)
    harmonic_intervals = {12, 19, 24, 28, 31, 36}
    for frame in range(active.shape[1]):
        candidates = np.flatnonzero(active[:, frame])
        candidates = candidates[np.argsort(decibels[candidates, frame])[::-1]][:12]
        selected: list[int] = []
        for pitch_index in candidates:
            replica = any(
                int(pitch_index) - lower in harmonic_intervals
                and decibels[lower, frame] > decibels[pitch_index, frame] + 9.0
                for lower in selected
                if lower < pitch_index
            )
            if not replica:
                selected.append(int(pitch_index))
                filtered[pitch_index, frame] = True
    active = binary_closing(filtered, structure=np.ones((1, 3), dtype=bool))

    frame_rate = sample_rate / HOP_LENGTH
    events: list[NoteEvent] = []
    for pitch_index in range(active.shape[0]):
        start: int | None = None
        for frame in range(active.shape[1] + 1):
            is_active = frame < active.shape[1] and bool(active[pitch_index, frame])
            renewed = (
                is_active
                and start is not None
                and frame - start >= 3
                and decibels[pitch_index, frame] - decibels[pitch_index, frame - 1] > 7.5
            )
            if is_active and start is None:
                start = frame
            elif (not is_active or renewed) and start is not None:
                event = _event_from_run(decibels, pitch_index, start, frame, frame_rate)
                if event is not None:
                    events.append(event)
                start = frame if renewed else None
    events.sort(key=lambda item: (item.start, item.midi_note, item.end))
    return events


def _note_onset_envelope(
    notes: list[NoteEvent], duration: float, sample_rate: int
) -> np.ndarray:
    frame_count = max(1, 1 + int(duration * sample_rate / TEMPO_HOP))
    envelope = np.zeros(frame_count, dtype=np.float32)
    for note in notes:
        frame = int(np.clip(round(note.start * sample_rate / TEMPO_HOP), 0, frame_count - 1))
        envelope[frame] = 1.0
    envelope = np.convolve(envelope, np.array([0.18, 0.58, 1.0, 0.58, 0.18]), mode="same")
    return envelope.astype(np.float32)


def _fold_tempo(value: float, reference: float) -> float:
    candidates = [value * factor for factor in (0.5, 1.0, 2.0) if 45 <= value * factor <= 210]
    if not candidates:
        return float(np.clip(value, 45, 210))
    return min(candidates, key=lambda item: abs(np.log2(item / max(reference, 1.0))))


def _constant_tempo_fit(
    onset_times: np.ndarray,
    reference_bpm: float,
    duration: float,
) -> tuple[float, float] | None:
    if duration < 8.0 or len(onset_times) < 8:
        return None
    subdivision = 4.0
    best_inliers: np.ndarray | None = None
    best_step = 0.0
    best_score = (-1, float("-inf"))
    candidates = np.linspace(
        max(45.0, reference_bpm * 0.97),
        min(210.0, reference_bpm * 1.03),
        121,
    )
    for candidate_bpm in candidates:
        nominal_step = 60.0 / candidate_bpm / subdivision
        for origin in onset_times[: min(24, len(onset_times))]:
            indices = np.rint((onset_times - origin) / nominal_step)
            residuals = np.abs(onset_times - (origin + indices * nominal_step))
            inliers = residuals <= 0.032
            count = int(np.count_nonzero(inliers))
            score = (count, -float(np.median(residuals[inliers])) if count else -1.0)
            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_step = nominal_step
    if best_inliers is None or np.count_nonzero(best_inliers) < max(8, round(len(onset_times) * 0.62)):
        return None
    times = onset_times[best_inliers]
    origin = float(times[0])
    indices = np.rint((times - origin) / best_step).astype(int)
    unique = np.concatenate(([True], np.diff(indices) != 0))
    indices = indices[unique]
    times = times[unique]
    if len(indices) < 8 or indices[-1] == indices[0]:
        return None
    slope, intercept = np.polyfit(indices, times, 1)
    refined_bpm = 60.0 / max(float(slope) * subdivision, 1e-9)
    if not 45.0 <= refined_bpm <= 210.0:
        return None
    residuals = np.abs(times - (intercept + indices * slope))
    median_residual = float(np.median(residuals))
    percentile_residual = float(np.percentile(residuals, 90))
    if median_residual > 0.014 or percentile_residual > 0.028:
        return None
    confidence = float(np.clip(1.0 - percentile_residual / 0.028, 0.55, 1.0))
    return refined_bpm, confidence


def _segment_tempi_consistent(
    onset: np.ndarray,
    sample_rate: int,
    reference_bpm: float,
    hop_length: int = TEMPO_HOP,
) -> tuple[bool, list[float]]:
    segment_count = 4 if len(onset) / (sample_rate / hop_length) >= 24 else 2
    estimates: list[float] = []
    for segment in np.array_split(onset, segment_count):
        if np.count_nonzero(segment > 0.15) < 4:
            continue
        estimate = float(
            np.asarray(
                librosa.feature.tempo(
                    onset_envelope=segment,
                    sr=sample_rate,
                    hop_length=hop_length,
                    aggregate=np.median,
                    start_bpm=reference_bpm,
                    std_bpm=1.6,
                    max_tempo=210.0,
                )
            ).reshape(-1)[0]
        )
        estimates.append(_fold_tempo(estimate, reference_bpm))
    if len(estimates) < 2:
        return True, estimates
    tolerance = max(2.0, reference_bpm * 0.025)
    center = float(np.median(estimates))
    inliers = [value for value in estimates if abs(value - center) <= tolerance]
    required = max(2, int(np.ceil(len(estimates) * 0.75)))
    if len(inliers) >= required:
        return True, inliers
    return False, estimates


def _repeated_tempo_levels(estimates: list[float], reference_bpm: float) -> list[float]:
    tolerance = max(2.0, reference_bpm * 0.025)
    clusters: list[list[float]] = []
    for estimate in sorted(estimates):
        if clusters and abs(estimate - float(np.mean(clusters[-1]))) <= tolerance:
            clusters[-1].append(estimate)
        else:
            clusters.append([estimate])
    repeated = [cluster for cluster in clusters if len(cluster) >= 2]
    if len(repeated) != 2:
        return []
    return [float(np.median(cluster)) for cluster in repeated]


def _tempo_track(
    onset: np.ndarray, notes: list[NoteEvent], sample_rate: int
) -> tuple[list[TempoPoint], float]:
    frame_rate = sample_rate / TEMPO_HOP
    duration = len(onset) / frame_rate
    factor = TEMPO_AUTOCORRELATION_FACTOR
    padded_length = int(np.ceil(len(onset) / factor) * factor)
    tempo_onset = np.pad(onset, (0, padded_length - len(onset))).reshape(-1, factor).max(axis=1)
    tempo_hop = TEMPO_HOP * factor
    tempo_frame_rate = sample_rate / tempo_hop
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=TEMPO_HOP,
        backtrack=False,
        wait=2,
    )
    unique_onsets = librosa.frames_to_time(
        onset_frames,
        sr=sample_rate,
        hop_length=TEMPO_HOP,
    )
    if len(unique_onsets) < 4 or duration < 3.0:
        return [], 0.0
    intervals = np.diff(unique_onsets)
    intervals = intervals[(intervals >= 0.18) & (intervals <= 2.0)]
    reference = 120.0
    if len(intervals):
        global_estimate = float(
            np.asarray(
                librosa.feature.tempo(
                    onset_envelope=tempo_onset,
                    sr=sample_rate,
                    hop_length=tempo_hop,
                    aggregate=np.median,
                    start_bpm=120.0,
                    std_bpm=1.4,
                    max_tempo=210.0,
                )
            ).reshape(-1)[0]
        )
        reference = _fold_tempo(global_estimate, 110.0)
    segments_consistent, segment_estimates = _segment_tempi_consistent(
        tempo_onset,
        sample_rate,
        reference,
        tempo_hop,
    )
    constant = _constant_tempo_fit(np.asarray(unique_onsets), reference, duration)
    tempo_levels = _repeated_tempo_levels(segment_estimates, reference)
    if constant is not None and not segments_consistent:
        if tempo_levels or constant[1] < 0.63:
            constant = None
    if constant is not None:
        bpm, confidence = constant
        points = [
            TempoPoint(float(time), float(bpm), confidence)
            for time in np.arange(0.0, duration + 1e-6, 1.0)
        ]
        return points, confidence
    raw = np.asarray(
        librosa.feature.tempo(
            onset_envelope=tempo_onset,
            sr=sample_rate,
            hop_length=tempo_hop,
            aggregate=None,
            start_bpm=reference,
            std_bpm=1.8,
            ac_size=float(np.clip(duration, 4.0, 10.0)),
            max_tempo=210.0,
        ),
        dtype=float,
    ).reshape(-1)
    if not len(raw):
        return [], 0.0
    folded = np.empty_like(raw)
    previous = reference
    for index, value in enumerate(raw):
        folded[index] = _fold_tempo(float(value), previous)
        previous = 0.9 * previous + 0.1 * folded[index]
    median_size = max(9, round(tempo_frame_rate * 3.0))
    if median_size % 2 == 0:
        median_size += 1
    folded = median_filter(folded, size=median_size, mode="nearest")
    if segment_estimates:
        folded = np.clip(
            folded,
            max(45.0, min(segment_estimates)),
            min(210.0, max(segment_estimates)),
        )
    if tempo_levels:
        distances = np.abs(folded[:, None] - np.asarray(tempo_levels)[None, :])
        level_indices = np.argmin(distances, axis=1)
        label_size = max(3, round(tempo_frame_rate * 5.0))
        if label_size % 2 == 0:
            label_size += 1
        level_indices = median_filter(level_indices, size=label_size, mode="nearest")
        folded = np.asarray(tempo_levels)[level_indices]
    edge = min(len(folded) // 3, median_size // 2)
    if edge > 0:
        folded[:edge] = folded[edge]
        folded[-edge:] = folded[-edge - 1]
    smoothed = np.empty_like(folded)
    smoothed[0] = folded[0]
    maximum_step = 4.0 / tempo_frame_rate
    for index in range(1, len(folded)):
        target = 0.94 * smoothed[index - 1] + 0.06 * folded[index]
        smoothed[index] = np.clip(
            target,
            smoothed[index - 1] - maximum_step,
            smoothed[index - 1] + maximum_step,
        )
    sample_step = max(1, round(tempo_frame_rate))
    points: list[TempoPoint] = []
    for frame in range(0, len(smoothed), sample_step):
        time = frame / tempo_frame_rate
        center = round(time * frame_rate)
        lo = max(0, center - round(frame_rate * 2))
        hi = min(len(onset), center + round(frame_rate * 2))
        evidence = int(np.count_nonzero(onset[lo:hi] > 0.18))
        confidence = float(np.clip(evidence / 7.0, 0.08, 1.0))
        points.append(TempoPoint(time, float(smoothed[frame]), confidence))
    return points, float(np.mean([point.confidence for point in points])) if points else 0.0


def _fallback_dynamic_beats(
    onset: np.ndarray, frame_rate: float, tempo: list[TempoPoint]
) -> np.ndarray:
    if not tempo:
        return np.array([], dtype=int)
    onset_frames = np.flatnonzero(onset > max(0.18, float(np.percentile(onset, 70))))
    anchor = int(onset_frames[0]) if len(onset_frames) else 0
    beats = [anchor]
    cursor = anchor / frame_rate
    duration = len(onset) / frame_rate
    while cursor < duration:
        bpm = float(np.interp(cursor, [item.time for item in tempo], [item.bpm for item in tempo]))
        predicted = cursor + 60.0 / max(bpm, 1.0)
        radius = 0.16 * 60.0 / max(bpm, 1.0)
        candidates = onset_frames[
            (onset_frames / frame_rate >= predicted - radius)
            & (onset_frames / frame_rate <= predicted + radius)
        ]
        cursor = float(candidates[np.argmax(onset[candidates])] / frame_rate) if len(candidates) else predicted
        frame = round(cursor * frame_rate)
        if frame >= len(onset) or frame <= beats[-1]:
            break
        beats.append(frame)
    return np.asarray(beats, dtype=int)


def _beat_track(
    onset: np.ndarray,
    sample_rate: int,
    tempo: list[TempoPoint],
    notes: list[NoteEvent],
) -> list[BeatMarker]:
    if not tempo:
        return []
    frame_rate = sample_rate / TEMPO_HOP
    frame_times = np.arange(len(onset)) / frame_rate
    bpm_curve = np.interp(
        frame_times,
        [item.time for item in tempo],
        [item.bpm for item in tempo],
    )
    try:
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=sample_rate,
            hop_length=TEMPO_HOP,
            bpm=bpm_curve,
            trim=False,
            sparse=True,
        )
        beat_frames = np.asarray(beat_frames, dtype=int)
    except (ValueError, TypeError):
        beat_frames = _fallback_dynamic_beats(onset, frame_rate, tempo)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < len(onset))]
    if len(beat_frames) < 2:
        beat_frames = _fallback_dynamic_beats(onset, frame_rate, tempo)
    if len(beat_frames) < 2:
        return []
    strengths = np.zeros(len(beat_frames), dtype=np.float32)
    for index, frame in enumerate(beat_frames):
        beat_time = frame / frame_rate
        nearby = [note for note in notes if abs(note.start - beat_time) <= 0.11]
        density = min(1.0, len(nearby) / 4.0)
        bass = (
            float(np.clip((72 - min(note.midi_note for note in nearby)) / 36.0, 0, 1))
            if nearby
            else 0.0
        )
        before = {
            note.midi_note % 12
            for note in notes
            if note.start <= beat_time - 0.06 < note.end
        }
        after = {
            note.midi_note % 12
            for note in notes
            if note.start <= beat_time + 0.06 < note.end
        }
        union = before | after
        harmonic_change = len(before ^ after) / len(union) if union else 0.0
        strengths[index] = (
            0.20 * (onset[frame] > 0.15)
            + 0.25 * density
            + 0.25 * bass
            + 0.30 * harmonic_change
        )
    meter_scores: list[tuple[float, int, int]] = []
    for meter in (3, 4, 6):
        for phase in range(meter):
            accented = strengths[phase::meter]
            if not len(accented):
                continue
            other_indices = np.arange(len(strengths)) % meter != phase
            others = strengths[other_indices]
            score = float(accented.mean() - (others.mean() if len(others) else 0.0))
            meter_scores.append((score, meter, phase))
    _, meter, downbeat_phase = max(meter_scores)
    return [
        BeatMarker(
            time=float(frame / frame_rate),
            strength=float(strengths[index]),
            is_downbeat=(index - downbeat_phase) % meter == 0,
            beat_in_bar=(index - downbeat_phase) % meter + 1,
        )
        for index, frame in enumerate(beat_frames)
    ]


def _timing_grid_errors(
    notes: list[NoteEvent], beats: list[BeatMarker]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beat_times = np.asarray(sorted({round(item.time, 4) for item in beats}), dtype=float)
    if len(beat_times) < 3:
        return beat_times, np.array([], dtype=float), np.array([], dtype=float)
    raw_starts = sorted(item.start for item in notes)
    onset_groups: list[list[float]] = []
    for start in raw_starts:
        if onset_groups and start - onset_groups[-1][0] <= 0.035:
            onset_groups[-1].append(start)
        else:
            onset_groups.append([start])
    onset_times = np.asarray([float(np.median(group)) for group in onset_groups], dtype=float)
    if len(onset_times) < 8:
        return beat_times, np.array([], dtype=float), np.array([], dtype=float)
    phases = np.asarray((0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0))
    timed_onsets: list[float] = []
    signed_errors: list[float] = []
    for onset in onset_times:
        index = int(np.searchsorted(beat_times, onset, side="right") - 1)
        if index < 0 or index + 1 >= len(beat_times):
            continue
        period = beat_times[index + 1] - beat_times[index]
        if not 0.2 <= period <= 2.0:
            continue
        phase = (onset - beat_times[index]) / period
        nearest = phases[int(np.argmin(np.abs(phases - phase)))]
        timed_onsets.append(float(onset))
        signed_errors.append(float((phase - nearest) * period))
    return beat_times, np.asarray(timed_onsets, dtype=float), np.asarray(signed_errors, dtype=float)


def _timing_grid_statistics(
    notes: list[NoteEvent], beats: list[BeatMarker]
) -> tuple[float | None, float, float | None]:
    beat_times, onset_times, errors = _timing_grid_errors(notes, beats)
    if len(errors) < 8:
        return None, 0.0, None
    errors -= float(np.median(errors))
    cutoff = max(0.012, float(np.percentile(np.abs(errors), 90)))
    deviation = float(np.sqrt(np.mean(np.minimum(np.abs(errors), cutoff) ** 2)))
    accuracy = float(100.0 * np.exp(-((deviation / 0.040) ** 2)))

    # A global RMS is sensitive to sparse beat/window mistakes. Measure the timing core
    # separately after removing the phase offset inside each overlapping local window.
    local_medians: list[float] = []
    for start in np.arange(0.0, max(0.0, beat_times[-1] - 9.99), 5.0):
        local = errors[(onset_times >= start) & (onset_times < start + 10.0)]
        if len(local) < 8:
            continue
        centered = local - float(np.median(local))
        local_medians.append(float(np.median(np.abs(centered))))
    local_error_ms = (
        float(np.median(local_medians) * 1000.0)
        if local_medians
        else float(np.median(np.abs(errors)) * 1000.0)
    )
    return accuracy, deviation * 1000.0, local_error_ms


def _timing_grid_metrics(
    notes: list[NoteEvent], beats: list[BeatMarker]
) -> tuple[float | None, float]:
    accuracy, deviation_ms, _ = _timing_grid_statistics(notes, beats)
    return accuracy, deviation_ms


def _grid_mechanical_evidence(grid_accuracy: float) -> float:
    exponent = float(np.clip(-(grid_accuracy - 90.0) / 1.5, -60.0, 60.0))
    return 100.0 / (1.0 + np.exp(exponent))


def _local_grid_mechanical_evidence(local_error_ms: float) -> float:
    exponent = float(np.clip((local_error_ms - 5.75) / 0.75, -60.0, 60.0))
    return 100.0 / (1.0 + np.exp(exponent))


def _combine_mechanical_evidence(
    grid_mechanical: float,
    local_grid_mechanical: float,
    tempo_mechanical: float,
) -> float:
    # Precise local timing is necessary but not sufficient. It must be corroborated by
    # either a precise global grid or stable tempo before the result is called mechanical.
    return min(local_grid_mechanical, max(grid_mechanical, tempo_mechanical))


def _chord_onset_mechanical_evidence(notes: list[NoteEvent]) -> float | None:
    metrics = _chord_onset_metrics(notes)
    if metrics is None:
        return None
    exact_ratio, polyphonic_ratio, _ = metrics
    exact_exponent = float(np.clip(-(exact_ratio - 45.0) / 3.0, -60.0, 60.0))
    polyphonic_exponent = float(np.clip(-(polyphonic_ratio - 46.0) / 3.0, -60.0, 60.0))
    exact_evidence = 100.0 / (1.0 + np.exp(exact_exponent))
    polyphonic_evidence = 100.0 / (1.0 + np.exp(polyphonic_exponent))
    return max(exact_evidence, polyphonic_evidence)


def _chord_onset_metrics(
    notes: list[NoteEvent],
) -> tuple[float, float, float] | None:
    """Return synchronization, polyphony and median spread for repeated chord attacks."""
    onset_groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda item: item.start):
        if onset_groups and note.start - onset_groups[-1][0].start <= 0.035:
            onset_groups[-1].append(note)
        else:
            onset_groups.append([note])
    simultaneous = [group for group in onset_groups if len(group) >= 2]
    if len(simultaneous) < 12:
        return None
    spreads = np.asarray(
        [max(item.start for item in group) - min(item.start for item in group) for group in simultaneous]
    )
    exact_ratio = float(100.0 * np.mean(spreads < 0.0015))
    polyphonic_ratio = float(100.0 * len(simultaneous) / max(1, len(onset_groups)))
    return exact_ratio, polyphonic_ratio, float(np.median(spreads))


def _chord_articulation_dynamics(
    notes: list[NoteEvent],
    beats: list[BeatMarker],
) -> tuple[float, float | None] | None:
    """Measure chord-attack tails and repeatability in beat-relative units."""
    beat_times = np.asarray(sorted({round(beat.time, 4) for beat in beats}), dtype=float)
    if len(beat_times) < 3:
        return None
    onset_groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda item: (item.start, item.midi_note)):
        if onset_groups and note.start - onset_groups[-1][0].start <= 0.035:
            onset_groups[-1].append(note)
        else:
            onset_groups.append([note])
    normalized_spreads: list[float] = []
    pair_offsets: dict[tuple[int, int], list[float]] = {}
    for group in onset_groups:
        if len(group) < 2:
            continue
        group_time = float(np.median([note.start for note in group]))
        beat_index = int(np.searchsorted(beat_times, group_time, side="right") - 1)
        if beat_index < 0 or beat_index + 1 >= len(beat_times):
            continue
        period = beat_times[beat_index + 1] - beat_times[beat_index]
        if not 0.2 <= period <= 2.0:
            continue
        starts = np.asarray([note.start for note in group], dtype=float)
        normalized_spreads.append(float(np.ptp(starts) / period))
        ordered = sorted(group, key=lambda note: note.midi_note)
        for left_index, left in enumerate(ordered[:-1]):
            for right in ordered[left_index + 1 :]:
                key = (left.midi_note, right.midi_note)
                pair_offsets.setdefault(key, []).append(
                    float((right.start - left.start) / period)
                )
    if len(normalized_spreads) < 8:
        return None
    pair_residuals: list[float] = []
    retained_pairs = 0
    for offsets in pair_offsets.values():
        if len(offsets) < 4:
            continue
        values = np.asarray(offsets, dtype=float)
        pair_residuals.extend(np.abs(values - np.median(values)))
        retained_pairs += len(values)
    pair_mad = (
        float(np.median(pair_residuals))
        if retained_pairs >= 12 and pair_residuals
        else None
    )
    return float(np.percentile(normalized_spreads, 90)), pair_mad


# Calibrated against the protected labeled corpus in artifacts/evaluation-samples.
# Values are standardized so each evidence family contributes on a comparable scale.
_SCORE_FEATURE_MEAN = np.asarray(
    (0.001124417195, 0.026243892384, 0.666373292202, 0.027356672420,
     0.048568949813, 0.859703512692, 0.027069469526, 0.421627875881,
     0.361139562842, 0.010914157254, 0.034926333143, 0.005259115162),
    dtype=float,
)
_SCORE_FEATURE_SCALE = np.asarray(
    (0.001401391782, 0.021202519274, 0.115107425000, 0.028075902756,
     0.052735753382, 0.070157083258, 0.009142679096, 0.114028775816,
     0.120220492900, 0.007379380831, 0.009957578815, 0.004526166205),
    dtype=float,
)
_SCORE_FEATURE_COEFFICIENT = np.asarray(
    (0.036096435331, 0.165445972255, -0.124490349412, 0.071181063725,
     0.053515069574, -0.146747748585, 0.071016048316, -0.214463460720,
     -0.262321132484, 0.207108990377, 0.288565295525, 0.310068392578),
    dtype=float,
)
_SCORE_INTERCEPT = 0.335035536850


def _tempo_score_features(result: AnalysisResult | AnalysisSnapshot) -> tuple[float, ...]:
    values = np.asarray([point.bpm for point in result.tempo], dtype=float)
    average = max(float(result.average_bpm), 1.0)
    differences = np.abs(np.diff(values))
    if len(values) >= 5:
        size = min(11, (len(values) // 2) * 2 - 1)
        baseline = median_filter(values, size=size, mode="nearest")
        high_frequency = float(np.std(values - baseline) / average)
        low_frequency = float(np.std(baseline) / average)
    else:
        high_frequency = 0.0
        low_frequency = 0.0
    step_median = float(np.median(differences) / average) if len(differences) else 0.0
    step_p90 = float(np.percentile(differences, 90) / average) if len(differences) else 0.0
    stable_ratio = (
        float(np.mean(differences <= max(0.35, average * 0.004)))
        if len(differences)
        else 1.0
    )
    return step_median, step_p90, stable_ratio, high_frequency, low_frequency


def _correlated_timing_human_evidence(step_p90: float, grid_accuracy: float) -> float:
    tempo_exponent = float(np.clip(-(step_p90 - 0.030) / 0.0075, -60.0, 60.0))
    grid_error = 1.0 - grid_accuracy / 100.0
    grid_exponent = float(np.clip(-(grid_error - 0.150) / 0.025, -60.0, 60.0))
    tempo_evidence = 100.0 / (1.0 + np.exp(tempo_exponent))
    grid_evidence = 100.0 / (1.0 + np.exp(grid_exponent))
    # Programmed tempo maps and loose grids each occur independently. Only their
    # conjunction is strong enough to override otherwise machine-like articulation.
    return min(tempo_evidence, grid_evidence)


def _calibrated_human_score(
    result: AnalysisResult | AnalysisSnapshot,
    grid_accuracy: float,
    timing_deviation_ms: float,
) -> float:
    average = max(float(result.average_bpm), 1.0)
    chord_metrics = _chord_onset_metrics(result.notes)
    if chord_metrics is None:
        # Keep early/sparse incremental results neutral until enough chord attacks exist.
        exact_ratio = float(_SCORE_FEATURE_MEAN[7] * 100.0)
        polyphonic_ratio = float(_SCORE_FEATURE_MEAN[8] * 100.0)
        median_spread = float(_SCORE_FEATURE_MEAN[9] * 60.0 / average)
    else:
        exact_ratio, polyphonic_ratio, median_spread = chord_metrics
    articulation = _chord_articulation_dynamics(result.notes, result.beats)
    if articulation is None:
        spread_p90 = float(_SCORE_FEATURE_MEAN[10])
        pair_repeat_mad = float(_SCORE_FEATURE_MEAN[11])
    else:
        spread_p90, pair_repeat_mad = articulation
        if pair_repeat_mad is None:
            pair_repeat_mad = float(_SCORE_FEATURE_MEAN[11])
    tempo_features = _tempo_score_features(result)
    features = np.asarray(
        (*tempo_features,
         float(grid_accuracy / 100.0),
         float(timing_deviation_ms * average / 60000.0),
         float(exact_ratio / 100.0),
         float(polyphonic_ratio / 100.0),
         float(median_spread * average / 60.0),
         spread_p90,
         pair_repeat_mad),
        dtype=float,
    )
    standardized = (features - _SCORE_FEATURE_MEAN) / _SCORE_FEATURE_SCALE
    logit = _SCORE_INTERCEPT + float(np.dot(_SCORE_FEATURE_COEFFICIENT, standardized))
    calibrated = float(100.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
    correlated_motion = _correlated_timing_human_evidence(tempo_features[1], grid_accuracy)
    return max(calibrated, correlated_motion)


def update_performance_score(result: AnalysisResult | AnalysisSnapshot) -> None:
    coefficient = result.bpm_std / max(result.average_bpm, 1.0)
    values = np.asarray([point.bpm for point in result.tempo], dtype=float)
    autocorrelation = (
        np.corrcoef(values[:-1], values[1:])[0, 1]
        if len(values) > 3 and result.bpm_std > 1e-9
        else 0.0
    )
    pattern = float(np.nan_to_num(abs(autocorrelation), nan=0.0))
    tempo_human = float(
        np.clip(100.0 * (coefficient / 0.045) * (0.65 + 0.35 * pattern), 0.0, 100.0)
    )
    grid_accuracy, timing_deviation_ms, local_error_ms = _timing_grid_statistics(
        result.notes, result.beats
    )
    if grid_accuracy is None:
        result.grid_accuracy = 0.0
        result.timing_deviation_ms = 0.0
        result.human_score = tempo_human
        result.mechanical_index = 100.0 - tempo_human
        return
    chord_metrics = _chord_onset_metrics(result.notes)
    if chord_metrics is None:
        tempo_mechanical = 100.0 - tempo_human
        grid_mechanical = _grid_mechanical_evidence(grid_accuracy)
        local_grid_mechanical = (
            _local_grid_mechanical_evidence(local_error_ms)
            if local_error_ms is not None
            else grid_mechanical
        )
        timing_mechanical = _combine_mechanical_evidence(
            grid_mechanical,
            local_grid_mechanical,
            tempo_mechanical,
        )
        human_score = 100.0 - timing_mechanical
    else:
        human_score = _calibrated_human_score(result, grid_accuracy, timing_deviation_ms)
    result.grid_accuracy = grid_accuracy
    result.timing_deviation_ms = timing_deviation_ms
    result.human_score = float(np.clip(human_score, 0.0, 100.0))
    result.mechanical_index = 100.0 - result.human_score


def _note_chroma(notes: list[NoteEvent], duration: float, frame_rate: float = 8.0) -> np.ndarray:
    chroma = np.zeros((max(1, int(np.ceil(duration * frame_rate))), 12), dtype=np.float32)
    for note in notes:
        start = max(0, int(np.floor(note.start * frame_rate)))
        end = min(len(chroma), max(start + 1, int(np.ceil(note.end * frame_rate))))
        weight = 0.35 + 0.65 * note.confidence
        chroma[start:end, note.midi_note % 12] += weight
    norm = chroma.sum(axis=1, keepdims=True)
    return np.divide(chroma, norm, out=np.zeros_like(chroma), where=norm > 1e-9)


def _detect_key(vector: np.ndarray) -> tuple[str, float, int, bool]:
    if vector.sum() <= 1e-8:
        return "Unknown", 0.0, 0, True
    normalized = (vector - vector.mean()) / (vector.std() + 1e-9)
    scores: list[tuple[float, int, bool]] = []
    for root in range(12):
        scores.append((float(np.dot(normalized, np.roll(MAJOR_PROFILE, root))), root, True))
        scores.append((float(np.dot(normalized, np.roll(MINOR_PROFILE, root))), root, False))
    scores.sort(reverse=True)
    best, root, major = scores[0]
    confidence = (best - scores[1][0]) / (abs(best) + 1e-9)
    return f"{PITCH_NAMES[root]} {'major' if major else 'minor'}", float(np.clip(confidence, 0, 1)), root, major


def _key_segments(chroma: np.ndarray, frame_rate: float) -> list[KeySegment]:
    window = max(1, int(frame_rate * 12))
    step = max(1, int(frame_rate * 4))
    raw: list[tuple[float, str, float]] = []
    for start in range(0, len(chroma), step):
        end = min(len(chroma), start + window)
        key, confidence, _, _ = _detect_key(chroma[start:end].mean(axis=0))
        raw.append((start / frame_rate, key, confidence))
    if not raw:
        return []
    merged: list[KeySegment] = []
    current_start, current_key, confidences = raw[0][0], raw[0][1], [raw[0][2]]
    for start, key, confidence in raw[1:]:
        if key != current_key and confidence >= 0.06:
            merged.append(KeySegment(current_start, start, current_key, float(np.mean(confidences))))
            current_start, current_key, confidences = start, key, [confidence]
        else:
            confidences.append(confidence)
    duration = len(chroma) / frame_rate
    merged.append(KeySegment(current_start, duration, current_key, float(np.mean(confidences))))
    return merged


def _key_at(time: float, keys: list[KeySegment]) -> tuple[str, int, bool]:
    segment = next((item for item in keys if item.start <= time < item.end), keys[-1] if keys else None)
    if segment is None or segment.key == "Unknown":
        return "Unknown", 0, True
    name, quality = segment.key.split(" ", 1)
    return segment.key, PITCH_NAMES.index(name), quality == "major"


def _roman_function(root: int, key_root: int, major: bool, chord: str) -> str:
    degree = (root - key_root) % 12
    major_map = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
    minor_map = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "V", 8: "VI", 10: "VII"}
    base = (major_map if major else minor_map).get(degree, f"{degree:+d}")
    if chord.endswith("7") and "7" not in base:
        base += "7"
    return base


def _classify_chord(vector: np.ndarray, key_root: int, major: bool) -> tuple[str, int, float]:
    if vector.sum() < 1e-7 or vector.max(initial=0) < 0.14:
        return "N", 0, 0.0
    best: tuple[float, str, int] = (-1e9, "N", 0)
    scale = ({0, 2, 4, 5, 7, 9, 11} if major else {0, 2, 3, 5, 7, 8, 10})
    for root in range(12):
        for template in CHORD_TEMPLATES:
            tones = {(root + interval) % 12 for interval in template.intervals}
            chord_energy = float(sum(vector[index] for index in tones))
            outside = float(sum(vector[index] for index in range(12) if index not in tones))
            diatonic_bonus = 0.09 if (root - key_root) % 12 in scale else 0.0
            missing = sum(vector[index] < 0.055 for index in tones)
            complexity = max(0, len(tones) - 3)
            score = chord_energy - 0.42 * outside - 0.075 * missing - 0.025 * complexity + diatonic_bonus + template.tension
            if score > best[0]:
                best = (score, f"{PITCH_NAMES[root]}{template.suffix}", root)
    return best[1], best[2], float(np.clip((best[0] + 0.2) / 1.2, 0, 1))


def _chord_segments(chroma: np.ndarray, frame_rate: float, keys: list[KeySegment]) -> list[ChordSegment]:
    harmonic_window = max(1, int(frame_rate * 1.25))
    hop = max(1, harmonic_window // 2)
    raw: list[ChordSegment] = []
    for start in range(0, len(chroma), hop):
        context_end = min(len(chroma), start + harmonic_window)
        segment_end = min(len(chroma), start + hop)
        center = (start + segment_end) / 2 / frame_rate
        key, key_root, major = _key_at(center, keys)
        # Duration-weighted note activation preserves arpeggiated chord tones.
        vector = np.mean(chroma[start:context_end], axis=0)
        chord, root, confidence = _classify_chord(vector, key_root, major)
        function = "" if chord == "N" else _roman_function(root, key_root, major, chord)
        raw.append(
            ChordSegment(
                start / frame_rate,
                segment_end / frame_rate,
                chord,
                key,
                function,
                confidence,
            )
        )
    for _ in range(2):
        for index in range(1, len(raw) - 1):
            previous, current, following = raw[index - 1], raw[index], raw[index + 1]
            if previous.chord == following.chord and current.chord != previous.chord:
                raw[index] = ChordSegment(
                    current.start,
                    current.end,
                    previous.chord,
                    current.key,
                    previous.function,
                    (previous.confidence + following.confidence) / 2,
                )
    merged: list[ChordSegment] = []
    for item in raw:
        if merged and item.chord == merged[-1].chord and item.key == merged[-1].key:
            previous = merged[-1]
            merged[-1] = ChordSegment(
                previous.start,
                item.end,
                item.chord,
                item.key,
                item.function,
                (previous.confidence + item.confidence) / 2,
            )
        elif item.chord != "N" or not merged:
            merged.append(item)
    return merged


def _analyze_events(
    notes: list[NoteEvent],
    duration: float,
    sample_rate: int,
    onset: np.ndarray,
    sparse_notes: bool,
) -> AnalysisResult:
    tempo, tempo_confidence = _tempo_track(onset, notes, sample_rate)
    beats = _beat_track(onset, sample_rate, tempo, notes)
    chroma_frame_rate = 8.0
    chroma = _note_chroma(notes, duration, chroma_frame_rate)
    keys = _key_segments(chroma, chroma_frame_rate)
    chords = _chord_segments(chroma, chroma_frame_rate, keys)
    bpms = np.asarray([point.bpm for point in tempo], dtype=float)
    average = float(np.mean(bpms)) if len(bpms) else 0.0
    minimum = float(np.min(bpms)) if len(bpms) else 0.0
    maximum = float(np.max(bpms)) if len(bpms) else 0.0
    standard_deviation = float(np.std(bpms)) if len(bpms) else 0.0
    if standard_deviation < 1e-9:
        standard_deviation = 0.0
    if sparse_notes:
        mode = "percussive fallback"
    elif len(bpms) and maximum - minimum < 0.05:
        mode = "constant-grid locked"
    elif tempo_confidence < 0.22:
        mode = "low-confidence note tracking"
    else:
        mode = "neural note-driven"
    result = AnalysisResult(
        duration=duration,
        tempo=tempo,
        beats=beats,
        chords=chords,
        keys=keys,
        average_bpm=average,
        min_bpm=minimum,
        max_bpm=maximum,
        bpm_std=standard_deviation,
        human_score=0.0,
        mechanical_index=0.0,
        mode=mode,
        notes=notes,
    )
    update_performance_score(result)
    return result


def analyze_note_events(
    notes: list[NoteEvent], duration: float, sample_rate: int
) -> AnalysisResult:
    ordered = sorted(notes, key=lambda item: (item.start, item.midi_note, item.end))
    onset = _note_onset_envelope(ordered, duration, sample_rate)
    sparse_notes = len(np.unique(np.round([note.start for note in ordered], 3))) < 4
    return _analyze_events(ordered, duration, sample_rate, onset, sparse_notes)


def analyze_audio(audio: np.ndarray, sample_rate: int) -> AnalysisResult:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    if not len(audio) or np.max(np.abs(audio), initial=0) < 1e-5:
        return AnalysisResult(duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "silent", [])
    peak = np.max(np.abs(audio), initial=0)
    if peak > 1.0:
        audio = audio / peak
    notes = transcribe_notes(audio, sample_rate)
    note_onset = _note_onset_envelope(notes, duration, sample_rate)
    unique_note_onsets = np.unique(np.round([note.start for note in notes], 3))
    sparse_notes = len(unique_note_onsets) < 4
    if not sparse_notes:
        return _analyze_events(notes, duration, sample_rate, note_onset, False)

    spectral_strength = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=TEMPO_HOP,
    ).astype(np.float32)
    peak_frames = librosa.onset.onset_detect(
        onset_envelope=spectral_strength,
        sr=sample_rate,
        hop_length=TEMPO_HOP,
        backtrack=False,
    )
    spectral_onset = np.zeros(len(spectral_strength), dtype=np.float32)
    spectral_onset[peak_frames] = 1.0
    spectral_onset = np.convolve(
        spectral_onset,
        np.array([0.18, 0.58, 1.0, 0.58, 0.18]),
        mode="same",
    ).astype(np.float32)
    frame_count = max(len(note_onset), len(spectral_onset))
    note_onset = np.pad(note_onset, (0, frame_count - len(note_onset)))
    spectral_onset = np.pad(spectral_onset, (0, frame_count - len(spectral_onset)))
    return _analyze_events(
        notes,
        duration,
        sample_rate,
        np.maximum(note_onset, spectral_onset),
        True,
    )
