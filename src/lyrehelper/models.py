from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MonitorState(str, Enum):
    CONNECTING = "connecting"
    STANDBY = "standby"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class TempoPoint:
    time: float
    bpm: float
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class BeatMarker:
    time: float
    strength: float
    is_downbeat: bool = False
    beat_in_bar: int = 0


@dataclass(frozen=True, slots=True)
class NoteEvent:
    start: float
    end: float
    midi_note: int
    frequency: float
    name: str
    velocity: int = 64
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ChordSegment:
    start: float
    end: float
    chord: str
    key: str
    function: str = ""
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class KeySegment:
    start: float
    end: float
    key: str
    confidence: float = 0.0


@dataclass(slots=True)
class AnalysisSnapshot:
    state: MonitorState = MonitorState.CONNECTING
    device_name: str = "Searching for output device"
    quality: str = "full"
    mode: str = "rhythmic"
    session_id: str | None = None
    elapsed: float = 0.0
    playhead: float = 0.0
    signal_db: float = -120.0
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    waveform_start: float = 0.0
    waveform_end: float = 0.0
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros((64, 128), dtype=np.float32))
    spectrum_start: float = 0.0
    spectrum_end: float = 0.0
    notes: list[NoteEvent] = field(default_factory=list)
    tempo: list[TempoPoint] = field(default_factory=list)
    beats: list[BeatMarker] = field(default_factory=list)
    chords: list[ChordSegment] = field(default_factory=list)
    keys: list[KeySegment] = field(default_factory=list)
    average_bpm: float = 0.0
    min_bpm: float = 0.0
    max_bpm: float = 0.0
    bpm_std: float = 0.0
    human_score: float = 0.0
    mechanical_index: float = 0.0
    grid_accuracy: float = 0.0
    timing_deviation_ms: float = 0.0
    last_archive: str | None = None
    recording_mode: str = "auto"
    session_label: str = "none"
    cooldown_remaining: float = 0.0
    analysis_latency: float = 0.0
    auto_candidate: bool = False


@dataclass(slots=True)
class AnalysisResult:
    duration: float
    tempo: list[TempoPoint]
    beats: list[BeatMarker]
    chords: list[ChordSegment]
    keys: list[KeySegment]
    average_bpm: float
    min_bpm: float
    max_bpm: float
    bpm_std: float
    human_score: float
    mechanical_index: float
    mode: str
    notes: list[NoteEvent] = field(default_factory=list)
    grid_accuracy: float = 0.0
    timing_deviation_ms: float = 0.0
