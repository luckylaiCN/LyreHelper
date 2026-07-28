from __future__ import annotations

import copy
import logging
import os
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .analysis import (
    analyze_audio,
    analyze_note_events,
    transcribe_notes,
    update_performance_score,
)
from .config import AppConfig
from .exporters import export_audio, export_session, prune_archives
from .models import (
    AnalysisResult,
    AnalysisSnapshot,
    BeatMarker,
    ChordSegment,
    KeySegment,
    MonitorState,
    NoteEvent,
    TempoPoint,
)

logger = logging.getLogger(__name__)
_MIN_ANALYZABLE_NOTE_ONSETS = 4
_AUTO_VALIDATION_SECONDS = 5.0
_AUTO_NOTE_PREROLL_SECONDS = 0.2
_AUTO_MIN_NOTE_COVERAGE = 0.15
_AUTO_MIN_ARCHIVE_SECONDS = 20.0
_AUTO_PAUSE_NOTE_COVERAGE = 0.10
_ANALYSIS_CONTEXT_SECONDS = 24.0
_INCREMENTAL_MUTABLE_SECONDS = 12.0
_METER_INITIAL_CONFIRMATIONS = 3
_METER_CHANGE_CONFIRMATIONS = 8


@dataclass(frozen=True, slots=True)
class _AudioBlock:
    samples: np.ndarray
    captured_at: float


@dataclass(frozen=True, slots=True)
class _PauseRecordingCommand:
    requested_at: float


@dataclass(frozen=True, slots=True)
class _RecordingModeCommand:
    mode: str


def _merge_incremental_notes(
    existing: list[NoteEvent],
    detected: list[NoteEvent],
    window_start: float,
    context_seconds: float = 2.0,
) -> list[NoteEvent]:
    replace_from = window_start + (context_seconds if window_start > 0 else 0.0)
    stable = [item for item in existing if item.end <= replace_from]
    replacement = [
        NoteEvent(
            item.start + window_start,
            item.end + window_start,
            item.midi_note,
            item.frequency,
            item.name,
            item.velocity,
            item.confidence,
        )
        for item in detected
        if item.end + window_start > replace_from
    ]
    return sorted(stable + replacement, key=lambda item: (item.start, item.midi_note, item.end))


def _valid_note_window(
    notes: list[NoteEvent],
    duration: float,
    *,
    require_full_window: bool,
) -> tuple[float | None, float]:
    if not notes or duration <= 0:
        return None, 0.0
    if require_full_window and duration < _AUTO_VALIDATION_SECONDS:
        return None, 0.0
    window_size = min(duration, _AUTO_VALIDATION_SECONDS)
    candidate_ends = {duration, window_size}
    for item in notes:
        candidate_ends.add(float(np.clip(item.end, window_size, duration)))
        candidate_ends.add(float(np.clip(item.start + window_size, window_size, duration)))
    best_coverage = 0.0
    for window_end in sorted(candidate_ends):
        window_start = window_end - window_size
        intervals = sorted(
            (max(window_start, item.start), min(window_end, item.end))
            for item in notes
            if item.end > window_start and item.start < window_end
        )
        if not intervals:
            continue
        covered = 0.0
        merged_start, merged_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= merged_end:
                merged_end = max(merged_end, end)
            else:
                covered += merged_end - merged_start
                merged_start, merged_end = start, end
        covered += merged_end - merged_start
        coverage = covered / window_size
        best_coverage = max(best_coverage, coverage)
        if coverage >= _AUTO_MIN_NOTE_COVERAGE:
            first_note = min(
                item.start
                for item in notes
                if item.end > window_start and item.start < window_end
            )
            return first_note, coverage
    return None, best_coverage


def _note_coverage(notes: list[NoteEvent], start: float, end: float) -> float:
    if end <= start:
        return 0.0
    intervals = sorted(
        (max(start, item.start), min(end, item.end))
        for item in notes
        if item.end > start and item.start < end
    )
    if not intervals:
        return 0.0
    covered = 0.0
    merged_start, merged_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= merged_end:
            merged_end = max(merged_end, interval_end)
        else:
            covered += merged_end - merged_start
            merged_start, merged_end = interval_start, interval_end
    covered += merged_end - merged_start
    return covered / (end - start)


def _analyze_recent_notes(
    notes: list[NoteEvent], duration: float, sample_rate: int
) -> tuple[AnalysisResult, float]:
    context_start = max(0.0, duration - _ANALYSIS_CONTEXT_SECONDS)
    local_notes = [
        NoteEvent(
            max(0.0, item.start - context_start),
            max(0.0, item.end - context_start),
            item.midi_note,
            item.frequency,
            item.name,
            item.velocity,
            item.confidence,
        )
        for item in notes
        if item.end > context_start
    ]
    result = analyze_note_events(local_notes, duration - context_start, sample_rate)
    result.duration = duration
    result.notes = list(notes)
    result.tempo = [
        TempoPoint(item.time + context_start, item.bpm, item.confidence)
        for item in result.tempo
    ]
    result.beats = [
        BeatMarker(
            item.time + context_start,
            item.strength,
            item.is_downbeat,
            item.beat_in_bar,
        )
        for item in result.beats
    ]
    result.chords = [
        ChordSegment(
            item.start + context_start,
            item.end + context_start,
            item.chord,
            item.key,
            item.function,
            item.confidence,
        )
        for item in result.chords
    ]
    result.keys = [
        KeySegment(
            item.start + context_start,
            item.end + context_start,
            item.key,
            item.confidence,
        )
        for item in result.keys
    ]
    return result, context_start


def _viterbi_tempo(points: list[TempoPoint], anchor_bpm: float | None) -> list[TempoPoint]:
    if not points:
        return []
    raw_values = np.asarray([point.bpm for point in points], dtype=float)
    reference = anchor_bpm or float(np.median(raw_values[: min(3, len(raw_values))]))
    metric_ratios = np.asarray((0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0))
    output = np.full(len(points), reference, dtype=float)
    pending_indices: list[int] = []
    pending_values: list[float] = []
    for index, point in enumerate(points):
        observed_ratio = point.bpm / max(reference, 1.0)
        ratio_index = int(np.argmin(np.abs(np.log2(observed_ratio / metric_ratios))))
        metric_ratio = float(metric_ratios[ratio_index])
        ratio_error = abs(np.log2(observed_ratio / metric_ratio))
        ratio_tolerance = 1.12 if metric_ratio in {0.5, 2.0} else 1.04
        folded = point.bpm / metric_ratio
        observed = (
            folded
            if ratio_error <= np.log2(ratio_tolerance) and 45.0 <= folded <= 210.0
            else point.bpm
        )
        local_tolerance = max(4.0, reference * 0.04)
        if abs(observed - reference) <= local_tolerance:
            output[index] = observed
            reference = 0.85 * reference + 0.15 * observed
            pending_indices.clear()
            pending_values.clear()
            continue
        pending_center = float(np.median(pending_values)) if pending_values else observed
        pending_tolerance = max(3.0, pending_center * 0.03)
        if pending_values and abs(observed - pending_center) <= pending_tolerance:
            pending_indices.append(index)
            pending_values.append(observed)
        else:
            pending_indices = [index]
            pending_values = [observed]
        output[index] = reference
        if len(pending_values) >= 3:
            for pending_index, pending_value in zip(pending_indices, pending_values):
                output[pending_index] = pending_value
            reference = float(np.median(pending_values))
            pending_indices.clear()
            pending_values.clear()
    for index in range(2, len(output) - 2):
        neighbors = np.asarray(
            (output[index - 2], output[index - 1], output[index + 1], output[index + 2])
        )
        baseline = float(np.median(neighbors))
        if np.ptp(neighbors) <= 1.5 and abs(output[index] - baseline) <= 8.0:
            output[index] = baseline
    groups: list[list[int]] = []
    for index, value in enumerate(output):
        if groups:
            center = float(np.median(output[groups[-1]]))
            if abs(value - center) <= max(2.0, center * 0.025):
                groups[-1].append(index)
                continue
        groups.append([index])
    for group_index in range(1, len(groups) - 1):
        group = groups[group_index]
        duration = points[group[-1]].time - points[group[0]].time
        if len(group) > 2 and duration > 2.1:
            continue
        previous = float(np.median(output[groups[group_index - 1]]))
        following = float(np.median(output[groups[group_index + 1]]))
        if abs(previous - following) > max(3.0, 0.03 * (previous + following) / 2.0):
            continue
        output[group] = (previous + following) / 2.0
    return [
        TempoPoint(point.time, float(output[index]), point.confidence)
        for index, point in enumerate(points)
    ]


def _merge_incremental_tempo(
    existing: list[TempoPoint],
    detected: list[TempoPoint],
    elapsed: float,
    mutable_seconds: float = _INCREMENTAL_MUTABLE_SECONDS,
) -> list[TempoPoint]:
    cutoff = max(0.0, elapsed - mutable_seconds)
    committed = [point for point in existing if point.time < cutoff]
    recent = [point for point in detected if point.time >= cutoff]
    if not recent:
        return list(existing)
    anchor = (
        float(np.median([point.bpm for point in committed[-12:]]))
        if committed
        else None
    )
    combined = committed + _viterbi_tempo(recent, anchor)
    deduplicated: dict[float, TempoPoint] = {}
    for point in combined:
        key = round(point.time, 4)
        current = deduplicated.get(key)
        if current is None:
            deduplicated[key] = point
            continue
        reference = anchor or current.bpm
        current_distance = abs(np.log2(current.bpm / max(reference, 1.0)))
        candidate_distance = abs(np.log2(point.bpm / max(reference, 1.0)))
        if candidate_distance < current_distance or (
            np.isclose(candidate_distance, current_distance)
            and point.confidence >= current.confidence
        ):
            deduplicated[key] = point
    merged = [deduplicated[key] for key in sorted(deduplicated)]
    return _repair_short_tempo_excursions(merged)


def _repair_short_tempo_excursions(
    points: list[TempoPoint],
    max_duration: float = 3.5,
) -> list[TempoPoint]:
    if len(points) < 3:
        return list(points)
    values = np.asarray([point.bpm for point in points], dtype=float)
    for _ in range(2):
        groups: list[list[int]] = []
        for index, value in enumerate(values):
            if groups:
                center = float(np.median(values[groups[-1]]))
                if abs(value - center) <= max(2.0, center * 0.025):
                    groups[-1].append(index)
                    continue
            groups.append([index])
        changed = False
        for group_index in range(1, len(groups) - 1):
            group = groups[group_index]
            duration = points[group[-1]].time - points[group[0]].time
            surrounding_span = (
                points[groups[group_index + 1][0]].time
                - points[groups[group_index - 1][-1]].time
            )
            if duration > max_duration or surrounding_span > max_duration + 4.0:
                continue
            previous = float(np.median(values[groups[group_index - 1]]))
            following = float(np.median(values[groups[group_index + 1]]))
            tolerance = max(3.0, 0.03 * (previous + following) / 2.0)
            if abs(previous - following) > tolerance:
                continue
            values[group] = (previous + following) / 2.0
            changed = True
        if not changed:
            break
    return [
        TempoPoint(point.time, float(values[index]), point.confidence)
        for index, point in enumerate(points)
    ]


def _detected_meter(beats: list[BeatMarker]) -> int | None:
    meter = max((beat.beat_in_bar for beat in beats), default=0)
    return meter if meter in {3, 4, 6} else None


def _merge_incremental_beats(
    existing: list[BeatMarker],
    detected: list[BeatMarker],
    elapsed: float,
    meter: int,
    *,
    mutable_seconds: float = _INCREMENTAL_MUTABLE_SECONDS,
    align_to_existing: bool = True,
) -> list[BeatMarker]:
    cutoff = max(0.0, elapsed - mutable_seconds)
    committed = [beat for beat in existing if beat.time < cutoff]
    recent = sorted(
        (beat for beat in detected if beat.time >= cutoff),
        key=lambda beat: beat.time,
    )
    if not recent:
        return list(existing)

    if committed and align_to_existing:
        nearby_times = [beat.time for beat in committed[-8:]] + [beat.time for beat in recent[:8]]
        intervals = np.diff(nearby_times)
        intervals = intervals[intervals > 0.05]
        period = float(np.median(intervals)) if len(intervals) else 0.5
        steps = max(1, round((recent[0].time - committed[-1].time) / period))
        first_number = (committed[-1].beat_in_bar - 1 + steps) % meter + 1
    else:
        first_downbeat = next(
            (index for index, beat in enumerate(recent) if beat.is_downbeat),
            0,
        )
        first_number = (-first_downbeat) % meter + 1

    relabeled = [
        BeatMarker(
            beat.time,
            beat.strength,
            beat_in_bar == 1,
            beat_in_bar,
        )
        for index, beat in enumerate(recent)
        for beat_in_bar in ((first_number - 1 + index) % meter + 1,)
    ]
    return committed + relabeled


def _apply_tempo_summary(
    result: AnalysisResult | AnalysisSnapshot, tempo: list[TempoPoint]
) -> None:
    values = np.asarray([point.bpm for point in tempo], dtype=float)
    result.tempo = tempo
    result.average_bpm = float(values.mean()) if len(values) else 0.0
    result.min_bpm = float(values.min()) if len(values) else 0.0
    result.max_bpm = float(values.max()) if len(values) else 0.0
    result.bpm_std = float(values.std()) if len(values) else 0.0
    if result.bpm_std < 1e-9:
        result.bpm_std = 0.0
    if len(values) and result.max_bpm - result.min_bpm < 0.05:
        result.mode = "constant-grid locked"
    elif result.mode == "constant-grid locked":
        result.mode = "incremental segmented"


def _spectrogram_preview(audio: np.ndarray) -> np.ndarray:
    if len(audio) < 1024:
        return np.zeros((64, 128), dtype=np.float32)
    frame_count = 128
    starts = np.linspace(0, max(0, len(audio) - 1024), frame_count).astype(int)
    window = np.hanning(1024)
    columns = []
    for start in starts:
        spectrum = np.abs(np.fft.rfft(audio[start : start + 1024] * window))
        bands = np.array_split(np.log1p(spectrum), 64)
        columns.append([float(item.mean()) for item in bands])
    image = np.asarray(columns, dtype=np.float32).T
    image -= image.min(initial=0)
    image /= image.max(initial=1e-9)
    return image


class AnalysisPipeline(threading.Thread):
    """Owns song boundary detection, analysis scheduling and isolated archives."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(name="analysis-pipeline", daemon=True)
        self.config = config
        # The backing queue is unbounded so recording commands can never be evicted by audio.
        # Audio submissions enforce the realtime backlog limit independently.
        self._queue: queue.Queue[
            _AudioBlock | _RecordingModeCommand | _PauseRecordingCommand | None
        ] = queue.Queue()
        self._audio_backlog_limit = 48
        self._lock = threading.Lock()
        self._recording_mode = (
            config.recording_mode if config.recording_mode in {"on", "off", "auto"} else "auto"
        )
        self._snapshot = AnalysisSnapshot(recording_mode=self._recording_mode)
        self._stopping = threading.Event()
        self._cooldown_until = 0.0
        self._auto_rearm_after_pause = False
        self._analysis_reference_capture_at: float | None = None
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=max(1, round(2.0 / self.config.block_duration))
        )
        self._blocks: list[np.ndarray] = []
        self._active = False
        self._signal_duration = 0.0
        self._silence_duration = 0.0
        self._analysis_clock = 0.0
        self._analysis_interval = 1.5
        self._session_counter = 0
        self._session_recording_mode: str | None = None
        self._session_label = "none"
        self._auto_session_validated = False
        self._auto_valid_start: float | None = None
        self._stable_meter: int | None = None
        self._meter_candidate: int | None = None
        self._meter_candidate_runs = 0
        self._total_samples = 0
        local_app_data = os.environ.get("LOCALAPPDATA")
        fallback_root = Path(local_app_data) if local_app_data else Path.home() / ".lyrehelper"
        self._pending_directory = fallback_root / "LyreHelper" / "pending"
        self._labels_directory = Path.cwd() / ".LyreHelper" / "labels"

    def submit(self, block: np.ndarray) -> None:
        if self._queue.qsize() >= self._audio_backlog_limit:
            with self._lock:
                self._snapshot.quality = "reduced"
            return
        self._queue.put_nowait(
            _AudioBlock(np.asarray(block, dtype=np.float32), time.monotonic())
        )

    def set_capture_status(self, state: MonitorState, device_name: str) -> None:
        with self._lock:
            self._snapshot.device_name = device_name
            if not self._active:
                self._snapshot.state = state

    def stop(self) -> None:
        self._stopping.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def set_recording_mode(self, mode: str) -> None:
        normalized = mode if mode in {"on", "off", "auto"} else "auto"
        self.config.recording_mode = normalized
        with self._lock:
            self._snapshot.recording_mode = normalized
        self._queue.put_nowait(_RecordingModeCommand(normalized))

    def pause_recording(self) -> None:
        self._queue.put_nowait(_PauseRecordingCommand(time.monotonic()))

    def set_session_label(self, label: str) -> None:
        normalized = label if label in {"human", "non_human"} else "none"
        with self._lock:
            self._session_label = normalized
            self._snapshot.session_label = normalized

    def _resolve_meter(self, detected: int | None) -> tuple[int, bool]:
        if detected not in {3, 4, 6}:
            return self._stable_meter or self._meter_candidate or 4, False
        if self._stable_meter is None:
            if detected == self._meter_candidate:
                self._meter_candidate_runs += 1
            else:
                self._meter_candidate = detected
                self._meter_candidate_runs = 1
            if self._meter_candidate_runs >= _METER_INITIAL_CONFIRMATIONS:
                self._stable_meter = detected
                self._meter_candidate = None
                self._meter_candidate_runs = 0
            return self._stable_meter or detected, False
        if detected == self._stable_meter:
            self._meter_candidate = None
            self._meter_candidate_runs = 0
            return self._stable_meter, False
        if detected == self._meter_candidate:
            self._meter_candidate_runs += 1
        else:
            self._meter_candidate = detected
            self._meter_candidate_runs = 1
        if self._meter_candidate_runs < _METER_CHANGE_CONFIRMATIONS:
            return self._stable_meter, False
        self._stable_meter = detected
        self._meter_candidate = None
        self._meter_candidate_runs = 0
        return self._stable_meter, True

    def manual_cut(self) -> None:
        self.pause_recording()

    def get_snapshot(self) -> AnalysisSnapshot:
        with self._lock:
            result = copy.copy(self._snapshot)
            result.waveform = self._snapshot.waveform.copy()
            result.spectrum = self._snapshot.spectrum.copy()
            result.notes = list(self._snapshot.notes)
            result.tempo = list(self._snapshot.tempo)
            result.beats = list(self._snapshot.beats)
            result.chords = list(self._snapshot.chords)
            result.keys = list(self._snapshot.keys)
            result.cooldown_remaining = max(0.0, self._cooldown_until - time.monotonic())
            if self._active and self._analysis_reference_capture_at is not None:
                result.analysis_latency = max(
                    0.0, time.monotonic() - self._analysis_reference_capture_at
                )
            return result

    def get_display_snapshot(self) -> AnalysisSnapshot:
        result = self.get_snapshot()
        with self._lock:
            active = self._active
            auto_candidate = self._snapshot.auto_candidate
            valid_start = self._auto_valid_start
        if active and auto_candidate:
            return AnalysisSnapshot(
                state=MonitorState.STANDBY,
                device_name=result.device_name,
                quality=result.quality,
                mode="waiting",
                signal_db=result.signal_db,
                last_archive=result.last_archive,
                recording_mode=result.recording_mode,
                session_label=result.session_label,
                cooldown_remaining=result.cooldown_remaining,
            )
        if not active or valid_start is None:
            return result
        offset = max(0.0, valid_start - _AUTO_NOTE_PREROLL_SECONDS)
        if offset <= 0:
            return result
        result.elapsed = max(0.0, result.elapsed - offset)
        result.playhead = max(0.0, result.playhead - offset)
        result.waveform_start -= offset
        result.waveform_end -= offset
        result.spectrum_start -= offset
        result.spectrum_end -= offset
        result.notes = [
            NoteEvent(
                max(0.0, item.start - offset),
                max(0.0, item.end - offset),
                item.midi_note,
                item.frequency,
                item.name,
                item.velocity,
                item.confidence,
            )
            for item in result.notes
            if item.end > offset
        ]
        result.tempo = [
            TempoPoint(item.time - offset, item.bpm, item.confidence)
            for item in result.tempo
            if item.time >= offset
        ]
        result.beats = [
            BeatMarker(
                item.time - offset,
                item.strength,
                item.is_downbeat,
                item.beat_in_bar,
            )
            for item in result.beats
            if item.time >= offset
        ]
        result.chords = [
            ChordSegment(
                max(0.0, item.start - offset),
                max(0.0, item.end - offset),
                item.chord,
                item.key,
                item.function,
                item.confidence,
            )
            for item in result.chords
            if item.end > offset
        ]
        result.keys = [
            KeySegment(
                max(0.0, item.start - offset),
                max(0.0, item.end - offset),
                item.key,
                item.confidence,
            )
            for item in result.keys
            if item.end > offset
        ]
        return result

    def run(self) -> None:
        try:
            self.config.output_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.info("Output directory will be retried on archive: %s", error)
        self._flush_pending_archives()
        self._prune_archives()
        while not self._stopping.is_set():
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if block is None:
                break
            if isinstance(block, _PauseRecordingCommand):
                self._handle_manual_cut(block.requested_at)
                continue
            if isinstance(block, _RecordingModeCommand):
                self._handle_recording_mode(block.mode)
                continue
            try:
                self._process(block.samples, block.captured_at)
            except Exception as error:
                logger.exception("Analysis block failed: %s", error)
                with self._lock:
                    self._snapshot.state = MonitorState.DEGRADED
                    self._snapshot.quality = "reduced"
            if self._queue.qsize() <= 1:
                with self._lock:
                    self._snapshot.quality = "full"
        if self._active and self._blocks:
            self._finalize()

    def _process(self, block: np.ndarray, captured_at: float | None = None) -> None:
        rms = float(np.sqrt(np.mean(np.square(block.astype(np.float64))) + 1e-12))
        signal_db = 20 * np.log10(max(rms, 1e-6))
        duration = len(block) / self.config.sample_rate
        has_signal = signal_db >= self.config.signal_threshold_db
        with self._lock:
            self._snapshot.signal_db = signal_db
            recording_mode = "auto" if self._auto_rearm_after_pause else self._recording_mode
        if recording_mode == "off":
            if self._active and self._blocks:
                self._finalize()
            self._pre_roll.clear()
            self._signal_duration = 0.0
            with self._lock:
                self._snapshot.state = MonitorState.STANDBY
                self._snapshot.cooldown_remaining = 0.0
            return
        block_time = captured_at if captured_at is not None else time.monotonic()
        cooldown_remaining = self._cooldown_until - block_time
        if cooldown_remaining > 0:
            self._pre_roll.clear()
            self._signal_duration = 0.0
            with self._lock:
                self._snapshot.state = MonitorState.STANDBY
                self._snapshot.cooldown_remaining = max(
                    0.0, self._cooldown_until - time.monotonic()
                )
            return
        if not self._active:
            self._pre_roll.append(block)
            if recording_mode == "on":
                self._begin(block_time)
            else:
                self._signal_duration = self._signal_duration + duration if has_signal else 0.0
                if self._signal_duration >= self.config.trigger_seconds:
                    self._auto_rearm_after_pause = False
                    self._begin(block_time)
            return
        self._blocks.append(block)
        self._total_samples += len(block)
        self._silence_duration = 0.0 if has_signal else self._silence_duration + duration
        elapsed = self._total_samples / self.config.sample_rate
        self._analysis_clock += duration
        with self._lock:
            self._snapshot.elapsed = elapsed
            self._snapshot.playhead = elapsed
            self._snapshot.state = MonitorState.ANALYZING
        if self._analysis_clock >= self._analysis_interval and self._queue.qsize() <= 2:
            self._analysis_clock = 0.0
            analysis_started = time.perf_counter()
            self._refresh(block_time)
            analysis_duration = time.perf_counter() - analysis_started
            if not self._active:
                return
            self._analysis_interval = float(np.clip(analysis_duration * 1.25, 1.5, 5.0))
        if recording_mode == "auto" and self._silence_duration >= self.config.end_silence_seconds:
            self._finalize()

    def _handle_manual_cut(self, requested_at: float | None = None) -> None:
        self._recording_mode = "auto"
        self.config.recording_mode = "auto"
        self._auto_rearm_after_pause = True
        request_time = time.monotonic() if requested_at is None else requested_at
        self._cooldown_until = request_time + 3.0
        if self._active and self._blocks:
            self._finalize()
        elif self._active:
            self._reset_session_runtime()
        self._pre_roll.clear()
        self._signal_duration = 0.0
        with self._lock:
            self._snapshot.state = MonitorState.STANDBY
            self._snapshot.recording_mode = "auto"
            self._snapshot.cooldown_remaining = max(
                0.0, self._cooldown_until - time.monotonic()
            )

    def _handle_recording_mode(self, mode: str) -> None:
        self._recording_mode = mode
        self.config.recording_mode = mode
        self._cooldown_until = 0.0
        self._auto_rearm_after_pause = False
        with self._lock:
            self._snapshot.recording_mode = mode
            self._snapshot.cooldown_remaining = 0.0
        if mode == "off" and self._active and self._blocks:
            self._finalize()
        if mode == "off":
            if self._active:
                self._reset_session_runtime()
            self._pre_roll.clear()
            self._signal_duration = 0.0

    def _begin(self, captured_at: float | None = None) -> None:
        self._active = True
        self._session_recording_mode = self._recording_mode
        self._auto_session_validated = self._session_recording_mode != "auto"
        self._auto_valid_start = None
        self._session_counter += 1
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{now}_{self._session_counter:03d}"
        self._blocks = list(self._pre_roll)
        self._total_samples = sum(len(item) for item in self._blocks)
        self._pre_roll.clear()
        self._silence_duration = 0.0
        self._analysis_clock = 0.0
        capture_time = time.monotonic() if captured_at is None else captured_at
        self._analysis_reference_capture_at = capture_time - (
            self._total_samples / self.config.sample_rate
        )
        with self._lock:
            last_archive = self._snapshot.last_archive
            device = self._snapshot.device_name
            self._snapshot = AnalysisSnapshot(
                state=MonitorState.ANALYZING,
                device_name=device,
                session_id=session_id,
                last_archive=last_archive,
                recording_mode=self._recording_mode,
                session_label=self._session_label,
                auto_candidate=not self._auto_session_validated,
            )

    def _refresh(self, captured_at: float | None = None) -> None:
        preview_seconds = 12.0
        preview_samples = int(preview_seconds * self.config.sample_rate)
        selected: list[np.ndarray] = []
        selected_samples = 0
        for block in reversed(self._blocks):
            if selected_samples >= preview_samples:
                break
            remaining = preview_samples - selected_samples
            chosen = block[-remaining:]
            selected.append(chosen)
            selected_samples += len(chosen)
        window = np.concatenate(list(reversed(selected))) if selected else np.zeros(1, dtype=np.float32)
        offset_samples = max(0, self._total_samples - len(window))
        with self._lock:
            reduced = self._snapshot.quality == "reduced"
        analysis_audio = window[::2] if reduced else window
        analysis_rate = self.config.sample_rate // 2 if reduced else self.config.sample_rate
        window_start = offset_samples / self.config.sample_rate
        detected_notes = transcribe_notes(analysis_audio, analysis_rate)
        waveform_source = window
        waveform_indices = np.linspace(0, max(0, len(waveform_source) - 1), 1024).astype(int)
        waveform = waveform_source[waveform_indices] if len(waveform_source) else np.zeros(1024)
        spectrum_audio = window
        with self._lock:
            existing_notes = list(self._snapshot.notes)
            existing_tempo = list(self._snapshot.tempo)
            existing_beats = list(self._snapshot.beats)
            existing_chords = list(self._snapshot.chords)
            existing_keys = list(self._snapshot.keys)
        merged_notes = _merge_incremental_notes(existing_notes, detected_notes, window_start)
        elapsed = self._total_samples / self.config.sample_rate
        if self._session_recording_mode == "auto" and not self._auto_session_validated:
            valid_start, _ = _valid_note_window(
                merged_notes, elapsed, require_full_window=True
            )
            if valid_start is not None:
                self._auto_session_validated = True
                self._auto_valid_start = valid_start
        result, analysis_start = _analyze_recent_notes(
            merged_notes, elapsed, self.config.sample_rate
        )
        stable_tempo = _merge_incremental_tempo(existing_tempo, result.tempo, elapsed)
        _apply_tempo_summary(result, stable_tempo)
        meter, meter_changed = self._resolve_meter(_detected_meter(result.beats))
        stable_beats = _merge_incremental_beats(
            existing_beats,
            result.beats,
            elapsed,
            meter,
            align_to_existing=not meter_changed,
        )
        result.notes = merged_notes
        result.beats = stable_beats
        update_performance_score(result)
        beat_cutoff = max(0.0, elapsed - _INCREMENTAL_MUTABLE_SECONDS)
        segment_cutoff = max(analysis_start, beat_cutoff)
        stable_chords = [item for item in existing_chords if item.end <= segment_cutoff]
        stable_chords.extend(item for item in result.chords if item.end > segment_cutoff)
        stable_keys = [item for item in existing_keys if item.end <= segment_cutoff]
        stable_keys.extend(item for item in result.keys if item.end > segment_cutoff)
        latest_coverage = _note_coverage(
            merged_notes,
            max(0.0, elapsed - _AUTO_VALIDATION_SECONDS),
            elapsed,
        )
        should_auto_pause = (
            self._session_recording_mode == "auto"
            and self._auto_session_validated
            and elapsed >= _AUTO_VALIDATION_SECONDS
            and latest_coverage < _AUTO_PAUSE_NOTE_COVERAGE
        )
        with self._lock:
            self._snapshot.tempo = list(result.tempo)
            self._snapshot.beats = stable_beats
            self._snapshot.notes = list(result.notes)
            self._snapshot.chords = stable_chords
            self._snapshot.keys = stable_keys
            self._snapshot.average_bpm = result.average_bpm
            self._snapshot.min_bpm = result.min_bpm
            self._snapshot.max_bpm = result.max_bpm
            self._snapshot.bpm_std = result.bpm_std
            self._snapshot.human_score = result.human_score
            self._snapshot.mechanical_index = result.mechanical_index
            self._snapshot.grid_accuracy = result.grid_accuracy
            self._snapshot.timing_deviation_ms = result.timing_deviation_ms
            self._snapshot.mode = result.mode
            self._snapshot.auto_candidate = not self._auto_session_validated
            self._snapshot.waveform = waveform.astype(np.float32)
            self._snapshot.waveform_start = window_start
            self._snapshot.waveform_end = self._total_samples / self.config.sample_rate
            self._snapshot.spectrum = _spectrogram_preview(spectrum_audio)
            self._snapshot.spectrum_start = window_start
            self._snapshot.spectrum_end = self._total_samples / self.config.sample_rate
            if captured_at is not None:
                self._analysis_reference_capture_at = captured_at
                self._snapshot.analysis_latency = max(0.0, time.monotonic() - captured_at)
        if should_auto_pause:
            logger.info(
                "AUTO note coverage %.1f%% is below 10.0%%; pausing session",
                latest_coverage * 100.0,
            )
            self._handle_manual_cut(captured_at)

    def _discard_auto_candidate(self, reason: str) -> None:
        logger.info(
            "Discarding AUTO candidate %s: %s",
            self._snapshot.session_id,
            reason,
        )
        with self._lock:
            device = self._snapshot.device_name
            last_archive = self._snapshot.last_archive
            self._snapshot = AnalysisSnapshot(
                state=MonitorState.STANDBY,
                device_name=device,
                last_archive=last_archive,
                recording_mode=self._recording_mode,
            )
        self._reset_session_runtime()

    def _finalize(self) -> None:
        with self._lock:
            self._snapshot.state = MonitorState.FINALIZING
            accumulated_notes = list(self._snapshot.notes)
            accumulated_tempo = list(self._snapshot.tempo)
            accumulated_beats = list(self._snapshot.beats)
            accumulated_chords = list(self._snapshot.chords)
            accumulated_keys = list(self._snapshot.keys)
        trim = int(self._silence_duration * self.config.sample_rate)
        audio = np.concatenate(self._blocks)
        if 0 < trim < len(audio):
            audio = audio[:-trim]
        duration = len(audio) / self.config.sample_rate
        if (
            self._session_recording_mode == "auto"
            and duration < _AUTO_MIN_ARCHIVE_SECONDS
        ):
            self._discard_auto_candidate(f"duration {duration:.1f}s is below 20.0s")
            return
        preview_samples = min(len(audio), int(12.0 * self.config.sample_rate))
        window_start = (len(audio) - preview_samples) / self.config.sample_rate
        tail_notes = transcribe_notes(audio[-preview_samples:], self.config.sample_rate)
        final_notes = _merge_incremental_notes(accumulated_notes, tail_notes, window_start)
        unique_starts = {round(item.start, 3) for item in final_notes}
        if self._session_recording_mode == "auto":
            valid_start, coverage = _valid_note_window(
                final_notes, duration, require_full_window=False
            )
            if valid_start is None:
                self._discard_auto_candidate(f"best note coverage was {coverage * 100.0:.1f}%")
                return
            self._auto_session_validated = True
            self._auto_valid_start = valid_start
        if self._session_recording_mode == "auto" and final_notes:
            leading_trim = max(
                0.0, (self._auto_valid_start or 0.0) - _AUTO_NOTE_PREROLL_SECONDS
            )
            trim_samples = min(len(audio), round(leading_trim * self.config.sample_rate))
            if trim_samples:
                leading_trim = trim_samples / self.config.sample_rate
                audio = audio[trim_samples:]
                duration = len(audio) / self.config.sample_rate
                final_notes = [
                    NoteEvent(
                        max(0.0, item.start - leading_trim),
                        max(0.0, item.end - leading_trim),
                        item.midi_note,
                        item.frequency,
                        item.name,
                        item.velocity,
                        item.confidence,
                    )
                    for item in final_notes
                    if item.end > leading_trim
                ]
                accumulated_tempo = [
                    TempoPoint(item.time - leading_trim, item.bpm, item.confidence)
                    for item in accumulated_tempo
                    if item.time >= leading_trim
                ]
                accumulated_beats = [
                    BeatMarker(
                        item.time - leading_trim,
                        item.strength,
                        item.is_downbeat,
                        item.beat_in_bar,
                    )
                    for item in accumulated_beats
                    if item.time >= leading_trim
                ]
                accumulated_chords = [
                    ChordSegment(
                        max(0.0, item.start - leading_trim),
                        max(0.0, item.end - leading_trim),
                        item.chord,
                        item.key,
                        item.function,
                        item.confidence,
                    )
                    for item in accumulated_chords
                    if item.end > leading_trim
                ]
                accumulated_keys = [
                    KeySegment(
                        max(0.0, item.start - leading_trim),
                        max(0.0, item.end - leading_trim),
                        item.key,
                        item.confidence,
                    )
                    for item in accumulated_keys
                    if item.end > leading_trim
                ]
                if duration < _AUTO_MIN_ARCHIVE_SECONDS:
                    self._discard_auto_candidate(
                        f"trimmed duration {duration:.1f}s is below 20.0s"
                    )
                    return
        if len(unique_starts) >= _MIN_ANALYZABLE_NOTE_ONSETS:
            result, analysis_start = _analyze_recent_notes(
                final_notes, duration, self.config.sample_rate
            )
            segment_cutoff = max(
                analysis_start,
                duration - _INCREMENTAL_MUTABLE_SECONDS,
            )
            result.chords = [
                item for item in accumulated_chords if item.end <= segment_cutoff
            ] + [item for item in result.chords if item.end > segment_cutoff]
            result.keys = [
                item for item in accumulated_keys if item.end <= segment_cutoff
            ] + [item for item in result.keys if item.end > segment_cutoff]
        else:
            result = analyze_audio(audio, self.config.sample_rate)
        stable_tempo = _merge_incremental_tempo(accumulated_tempo, result.tempo, duration)
        _apply_tempo_summary(result, stable_tempo)
        meter = self._stable_meter or _detected_meter(result.beats) or 4
        result.beats = _merge_incremental_beats(
            accumulated_beats,
            result.beats,
            duration,
            meter,
        )
        update_performance_score(result)
        waveform_indices = np.linspace(0, max(0, len(audio) - 1), 4096).astype(int)
        final_waveform = audio[waveform_indices] if len(audio) else np.zeros(4096)
        final_spectrum = _spectrogram_preview(audio)
        with self._lock:
            session_id = self._snapshot.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            session_label = self._session_label
        try:
            midi_path, _ = export_session(result, self.config.output_path, session_id)
            export_audio(audio, self.config.sample_rate, self.config.output_path, session_id)
            last_archive = str(midi_path.parent / session_id)
        except OSError as error:
            logger.error("Primary archive failed; staging for silent retry: %s", error)
            try:
                export_session(result, self._pending_directory, session_id)
                export_audio(audio, self.config.sample_rate, self._pending_directory, session_id)
                last_archive = f"Pending automatic transfer: {session_id}"
            except OSError as staging_error:
                logger.error("Local staging also failed: %s", staging_error)
                last_archive = f"Archive unavailable: {session_id}"
        if session_label in {"human", "non_human"}:
            try:
                label_directory = self._labels_directory / session_label
                export_session(result, label_directory, session_id)
                export_audio(audio, self.config.sample_rate, label_directory, session_id)
            except OSError as error:
                logger.error("Labeled archive copy failed for %s: %s", session_id, error)
        with self._lock:
            device = self._snapshot.device_name
            self._snapshot = AnalysisSnapshot(
                state=MonitorState.STANDBY,
                device_name=device,
                last_archive=last_archive,
                elapsed=result.duration,
                playhead=result.duration,
                waveform=final_waveform.astype(np.float32),
                waveform_start=0.0,
                waveform_end=result.duration,
                spectrum=final_spectrum,
                spectrum_start=0.0,
                spectrum_end=result.duration,
                notes=list(result.notes),
                tempo=list(result.tempo),
                beats=list(result.beats),
                chords=list(result.chords),
                keys=list(result.keys),
                average_bpm=result.average_bpm,
                min_bpm=result.min_bpm,
                max_bpm=result.max_bpm,
                bpm_std=result.bpm_std,
                human_score=result.human_score,
                mechanical_index=result.mechanical_index,
                grid_accuracy=result.grid_accuracy,
                timing_deviation_ms=result.timing_deviation_ms,
                recording_mode=self._recording_mode,
            )
        self._reset_session_runtime()
        self._flush_pending_archives()
        self._prune_archives()

    def _reset_session_runtime(self) -> None:
        self._pre_roll.clear()
        self._blocks = []
        self._active = False
        self._session_recording_mode = None
        self._auto_session_validated = False
        self._auto_valid_start = None
        self._stable_meter = None
        self._meter_candidate = None
        self._meter_candidate_runs = 0
        self._signal_duration = 0.0
        self._silence_duration = 0.0
        self._analysis_clock = 0.0
        self._analysis_interval = 1.5
        self._total_samples = 0
        self._analysis_reference_capture_at = None
        with self._lock:
            self._session_label = "none"
            self._snapshot.session_label = "none"

    def _flush_pending_archives(self) -> None:
        if not self._pending_directory.exists():
            return
        try:
            self.config.output_path.mkdir(parents=True, exist_ok=True)
            midi_sources = list(self._pending_directory.glob("*_transcription.mid"))
            midi_sources.extend(self._pending_directory.glob("*_tempo.mid"))
            for source in midi_sources:
                suffix = "_transcription.mid" if source.name.endswith("_transcription.mid") else "_tempo.mid"
                session_id = source.name.removesuffix(suffix)
                partner = self._pending_directory / f"{session_id}_chords.csv"
                if not partner.exists():
                    continue
                audio = self._pending_directory / f"{session_id}_audio.wav"
                items = [source, partner]
                if audio.exists():
                    items.append(audio)
                for item in items:
                    destination = self.config.output_path / item.name
                    temp = destination.with_suffix(destination.suffix + ".retry")
                    shutil.copy2(item, temp)
                    os.replace(temp, destination)
                source.unlink(missing_ok=True)
                partner.unlink(missing_ok=True)
                audio.unlink(missing_ok=True)
        except OSError as error:
            logger.info("Pending archives remain staged: %s", error)

    def _prune_archives(self) -> None:
        try:
            removed = prune_archives(self.config.output_path, keep=10)
            if removed:
                logger.info("Pruned %d old archive sessions", len(removed))
        except OSError as error:
            logger.info("Old archives could not be pruned: %s", error)
