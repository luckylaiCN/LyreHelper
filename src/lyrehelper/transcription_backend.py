from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

from .models import NoteEvent

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * 2 - FFT_HOP
ANNOTATION_FRAMES_PER_WINDOW = ANNOTATIONS_FPS * 2
MIDI_OFFSET = 21
OVERLAPPING_FRAMES = 30
LOWEST_MIDI = 36
HIGHEST_MIDI = 107


def _preferred_providers(available: list[str]) -> list[str]:
    priority = (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    return [provider for provider in priority if provider in available]


@lru_cache(maxsize=1)
def _session() -> object | None:
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = Path(__file__).parent / "assets" / "basic_pitch" / "nmp.onnx"
        available = ort.get_available_providers()
        providers = _preferred_providers(available)
        if "DmlExecutionProvider" in providers:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        for provider in providers:
            requested = [provider]
            if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
                requested.append("CPUExecutionProvider")
            try:
                session = ort.InferenceSession(
                    str(model),
                    sess_options=options,
                    providers=requested,
                )
                logger.info("Neural transcription provider: %s", session.get_providers()[0])
                return session
            except (RuntimeError, ValueError) as error:
                logger.warning("Neural provider %s unavailable: %s", provider, error)
        return None
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.warning("Neural transcription backend unavailable: %s", error)
        return None


def transcription_execution_device() -> tuple[str, bool]:
    """Return the active inference-device label and whether it is accelerated."""
    session = _session()
    if session is None:
        return "CPU · FALLBACK", False
    provider = str(session.get_providers()[0])
    labels = {
        "CUDAExecutionProvider": ("GPU · CUDA", True),
        "DmlExecutionProvider": ("GPU · DIRECTML", True),
        "CoreMLExecutionProvider": ("ACCEL · COREML", True),
        "CPUExecutionProvider": ("CPU", False),
    }
    return labels.get(provider, (provider.removesuffix("ExecutionProvider").upper(), False))


def transcription_runtime_available() -> bool:
    return _session() is not None


def _model_output(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray] | None:
    session = _session()
    if session is None:
        return None
    if sample_rate != AUDIO_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE)
    audio = np.asarray(audio, dtype=np.float32)
    original_length = len(audio)
    overlap_samples = OVERLAPPING_FRAMES * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_samples
    padded = np.concatenate((np.zeros(overlap_samples // 2, dtype=np.float32), audio))
    note_outputs: list[np.ndarray] = []
    onset_outputs: list[np.ndarray] = []
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    for start in range(0, len(padded), hop_size):
        window = padded[start : start + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - len(window)))
        values = session.run(None, {input_name: window.reshape(1, AUDIO_N_SAMPLES, 1)})
        mapped = dict(zip(output_names, values))
        note_outputs.append(mapped["StatefulPartitionedCall:1"])
        onset_outputs.append(mapped["StatefulPartitionedCall:2"])
    trim = OVERLAPPING_FRAMES // 2

    def unwrap(chunks: list[np.ndarray]) -> np.ndarray:
        combined = np.concatenate(chunks, axis=0)[:, trim:-trim, :]
        flattened = combined.reshape(-1, combined.shape[-1])
        frame_count = int(np.floor(original_length * ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE))
        return flattened[:frame_count]

    return unwrap(note_outputs), unwrap(onset_outputs)


def _events_from_probabilities(notes: np.ndarray, onsets: np.ndarray) -> list[NoteEvent]:
    onset_threshold = 0.65
    frame_threshold = 0.30
    minimum_note_frames = 10
    energy_tolerance = 11
    frame_numbers = np.arange(len(notes) + 1)
    original_times = frame_numbers * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.floor(frame_numbers / ANNOTATION_FRAMES_PER_WINDOW)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOTATION_FRAMES_PER_WINDOW - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018
    frame_times = original_times - window_offset * window_numbers
    start_index = LOWEST_MIDI - MIDI_OFFSET
    end_index = HIGHEST_MIDI - MIDI_OFFSET + 1
    pitch_slice = slice(start_index, min(end_index, notes.shape[1]))
    remaining = notes.copy()
    local_peaks = np.zeros_like(onsets, dtype=bool)
    local_peaks[1:-1] = (onsets[1:-1] > onsets[:-2]) & (onsets[1:-1] > onsets[2:])
    if len(onsets) > 1:
        local_peaks[0] = onsets[0] > onsets[1]
        local_peaks[-1] = onsets[-1] > onsets[-2]
    peak_frames, peak_pitches = np.where(
        local_peaks[:, pitch_slice] & (onsets[:, pitch_slice] >= onset_threshold)
    )
    peak_pitches += start_index

    events: list[NoteEvent] = []
    onset_strengths: dict[int, float] = {}
    for start_frame, pitch_index in zip(peak_frames[::-1], peak_pitches[::-1]):
        if start_frame >= len(notes) - 1:
            continue
        frame = int(start_frame) + 1
        quiet_frames = 0
        while frame < len(notes) - 1 and quiet_frames < energy_tolerance:
            if remaining[frame, pitch_index] < frame_threshold:
                quiet_frames += 1
            else:
                quiet_frames = 0
            frame += 1
        end_frame = frame - quiet_frames
        if end_frame - start_frame <= minimum_note_frames:
            continue

        remaining[start_frame:end_frame, pitch_index] = 0.0
        if pitch_index > 0:
            remaining[start_frame:end_frame, pitch_index - 1] = 0.0
        if pitch_index + 1 < remaining.shape[1]:
            remaining[start_frame:end_frame, pitch_index + 1] = 0.0

        midi_note = int(pitch_index + MIDI_OFFSET)
        confidence = float(np.mean(notes[start_frame:end_frame, pitch_index]))
        event = NoteEvent(
            float(frame_times[start_frame]),
            float(frame_times[end_frame]),
            midi_note,
            float(librosa.midi_to_hz(midi_note)),
            str(librosa.midi_to_note(midi_note, unicode=False)),
            96,
            confidence,
        )
        events.append(event)
        onset_strengths[id(event)] = float(onsets[start_frame, pitch_index])
    if not events:
        for pitch_index in range(start_index, min(end_index, notes.shape[1])):
            active_frames = np.flatnonzero(notes[:, pitch_index] >= frame_threshold)
            if not len(active_frames):
                continue
            boundaries = np.flatnonzero(np.diff(active_frames) > energy_tolerance) + 1
            for run in np.split(active_frames, boundaries):
                start_frame = int(run[0])
                end_frame = int(run[-1]) + 1
                if end_frame - start_frame <= minimum_note_frames:
                    continue
                if np.max(notes[start_frame:end_frame, pitch_index], initial=0.0) < 0.58:
                    continue
                confidence = float(np.mean(notes[start_frame:end_frame, pitch_index]))
                midi_note = int(pitch_index + MIDI_OFFSET)
                event = NoteEvent(
                    float(frame_times[start_frame]),
                    float(frame_times[end_frame]),
                    midi_note,
                    float(librosa.midi_to_hz(midi_note)),
                    str(librosa.midi_to_note(midi_note, unicode=False)),
                    96,
                    confidence,
                )
                events.append(event)
                onset_strengths[id(event)] = 1.0
    events.sort(key=lambda item: (item.start, item.midi_note, item.end))
    filtered: list[NoteEvent] = []
    cursor = 0
    while cursor < len(events):
        group_end = cursor + 1
        while group_end < len(events) and events[group_end].start - events[cursor].start <= 0.035:
            group_end += 1
        group = events[cursor:group_end]
        strongest = max(event.confidence for event in group)
        relative = [event for event in group if event.confidence >= strongest * 0.62]
        filtered.extend(
            event
            for event in relative
            if not any(
                upper.midi_note == event.midi_note + 12
                and upper.confidence >= event.confidence * 1.4
                for upper in relative
            )
        )
        cursor = group_end

    merged: list[NoteEvent] = []
    root_onsets: list[float] = []
    for event in sorted(filtered, key=lambda item: (item.midi_note, item.start, item.end)):
        onset_strength = onset_strengths[id(event)]
        if (
            merged
            and event.midi_note == merged[-1].midi_note
            and abs(event.start - merged[-1].end) <= 0.02
            and onset_strength < root_onsets[-1] * 0.875
        ):
            previous = merged[-1]
            merged[-1] = NoteEvent(
                previous.start,
                max(previous.end, event.end),
                previous.midi_note,
                previous.frequency,
                previous.name,
                96,
                max(previous.confidence, event.confidence),
            )
        else:
            merged.append(event)
            root_onsets.append(onset_strength)
    return sorted(merged, key=lambda item: (item.start, item.midi_note, item.end))


def transcribe_with_neural_model(
    audio: np.ndarray, sample_rate: int
) -> list[NoteEvent] | None:
    output = _model_output(audio, sample_rate)
    if output is None:
        return None
    return _events_from_probabilities(*output)
