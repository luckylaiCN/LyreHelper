from __future__ import annotations

import numpy as np
import pytest

from lyrehelper.analysis import (
    MAJOR_PROFILE,
    TEMPO_HOP,
    _beat_track,
    _chord_articulation_dynamics,
    _chord_onset_mechanical_evidence,
    _combine_mechanical_evidence,
    _correlated_timing_human_evidence,
    _grid_mechanical_evidence,
    _key_segments,
    _repeated_tempo_levels,
    _segment_tempi_consistent,
    tempo_standard_deviation,
    _tempo_track,
    _tempo_score_features,
    analyze_audio,
    analyze_note_events,
    update_performance_score,
)
from lyrehelper.models import AnalysisResult, BeatMarker, NoteEvent, TempoPoint


def tonal_click_track(duration: float = 18.0, bpm: float = 120.0, sample_rate: int = 22050) -> np.ndarray:
    time = np.arange(round(duration * sample_rate)) / sample_rate
    audio = (
        0.10 * np.sin(2 * np.pi * 261.63 * time)
        + 0.08 * np.sin(2 * np.pi * 329.63 * time)
        + 0.08 * np.sin(2 * np.pi * 392.00 * time)
    )
    for start in np.arange(0, duration, 60 / bpm):
        index = round(start * sample_rate)
        envelope_index = np.arange(min(1200, len(audio) - index))
        click = 0.8 * np.exp(-envelope_index / 130) * np.sin(
            2 * np.pi * 1000 * envelope_index / sample_rate
        )
        audio[index : index + len(click)] += click
    return audio.astype(np.float32)


def test_grid_accuracy_requires_high_precision_for_mechanical_evidence() -> None:
    assert _grid_mechanical_evidence(77.0) < 1.0
    assert 18.0 <= _grid_mechanical_evidence(88.0) <= 23.0
    assert _grid_mechanical_evidence(90.0) == 50.0
    assert _grid_mechanical_evidence(95.0) >= 96.0
    assert _grid_mechanical_evidence(98.0) >= 99.0


def test_mechanical_score_requires_correlated_timing_evidence() -> None:
    assert _combine_mechanical_evidence(35.0, 96.0, 47.0) == 47.0
    assert _combine_mechanical_evidence(84.0, 86.0, 0.0) == 0.0
    assert _combine_mechanical_evidence(2.0, 75.0, 63.0) == 63.0


def test_monophonic_grid_fit_requires_stable_tempo_before_calling_it_mechanical() -> None:
    beats = [BeatMarker(index * 0.5, 1.0) for index in range(41)]
    notes = [
        NoteEvent(index * 0.25, index * 0.25 + 0.1, 60, 261.63, "C4", 96, 1.0)
        for index in range(1, 80)
    ]
    stable_tempo = [TempoPoint(float(index), 120.0, 1.0) for index in range(20)]
    moving_values = [104.0 + index * 1.2 for index in range(20)]
    moving_tempo = [
        TempoPoint(float(index), bpm, 1.0)
        for index, bpm in enumerate(moving_values)
    ]
    machine = AnalysisResult(
        20.0, stable_tempo, beats, [], [], 120.0, 120.0, 120.0, 0.0,
        0.0, 0.0, "test", notes,
    )
    human = AnalysisResult(
        20.0,
        moving_tempo,
        beats,
        [],
        [],
        float(np.mean(moving_values)),
        min(moving_values),
        max(moving_values),
        float(np.std(moving_values)),
        0.0,
        0.0,
        "test",
        notes,
    )

    update_performance_score(machine)
    update_performance_score(human)

    assert machine.human_score < 5.0
    assert human.human_score > 90.0


def test_tempo_standard_deviation_pools_within_segment_variance() -> None:
    first = [79.0, 81.0] * 6
    second = [118.0, 122.0] * 6
    points = [
        TempoPoint(float(index), bpm, 1.0)
        for index, bpm in enumerate(first + second)
    ]

    deviation = tempo_standard_deviation(points)

    assert deviation == pytest.approx(np.sqrt(2.5))


def test_tempo_standard_deviation_discards_short_jump_segments() -> None:
    points = [
        TempoPoint(float(index), bpm, 1.0)
        for index, bpm in enumerate([90.0] * 11 + [120.0] * 3 + [90.0] * 11)
    ]

    assert tempo_standard_deviation(points) == 0.0


def test_piecewise_constant_tempo_does_not_become_human_evidence() -> None:
    beats = [BeatMarker(index * 0.5, 1.0) for index in range(45)]
    notes = [
        NoteEvent(index * 0.25, index * 0.25 + 0.1, 60, 261.63, "C4", 96, 1.0)
        for index in range(1, 88)
    ]
    tempo = [
        TempoPoint(float(index), 90.0 if index < 22 else 120.0, 1.0)
        for index in range(44)
    ]
    result = AnalysisResult(
        44.0,
        tempo,
        beats,
        [],
        [],
        105.0,
        90.0,
        120.0,
        15.0,
        0.0,
        0.0,
        "test",
        notes,
    )

    update_performance_score(result)

    assert result.bpm_std == 0.0
    assert result.human_score < 5.0


def test_chord_onsets_capture_synchronized_machine_articulation() -> None:
    def event(start: float, midi_note: int) -> NoteEvent:
        return NoteEvent(start, start + 0.2, midi_note, 440.0, "test", 80, 1.0)

    machine = []
    expressive = []
    for index in range(40):
        start = float(index)
        machine.extend((event(start, 60), event(start, 64)))
        expressive.append(event(start, 60))
        if index < 12:
            expressive.append(event(start + 0.012, 64))

    assert (_chord_onset_mechanical_evidence(machine) or 0.0) >= 95.0
    assert (_chord_onset_mechanical_evidence(expressive) or 100.0) <= 5.0


def test_chord_articulation_dynamics_detects_repeatable_machine_attacks() -> None:
    def event(start: float, midi_note: int) -> NoteEvent:
        return NoteEvent(start, start + 0.2, midi_note, 440.0, "test", 80, 1.0)

    beats = [BeatMarker(index * 0.5, 1.0) for index in range(28)]
    machine: list[NoteEvent] = []
    expressive: list[NoteEvent] = []
    for index in range(1, 25):
        start = index * 0.5
        machine.extend((event(start, 60), event(start, 64), event(start, 67)))
        expressive.extend(
            (
                event(start, 60),
                event(start + 0.004 + (index % 4) * 0.003, 64),
                event(start + 0.012 + (index % 3) * 0.005, 67),
            )
        )

    machine_tail, machine_repeat = _chord_articulation_dynamics(machine, beats) or (1.0, 1.0)
    expressive_tail, expressive_repeat = _chord_articulation_dynamics(
        expressive, beats
    ) or (0.0, 0.0)

    assert machine_tail == 0.0
    assert machine_repeat == 0.0
    assert expressive_tail > 0.03
    assert expressive_repeat is not None and expressive_repeat > 0.003


def test_chord_pair_repeatability_waits_for_enough_evidence() -> None:
    notes: list[NoteEvent] = []
    beats = [BeatMarker(index * 0.5, 1.0) for index in range(14)]
    for index in range(1, 9):
        start = index * 0.5
        notes.extend(
            (
                NoteEvent(start, start + 0.2, 60, 440.0, "test", 80, 1.0),
                NoteEvent(start + 0.01, start + 0.2, 64, 440.0, "test", 80, 1.0),
            )
        )

    metrics = _chord_articulation_dynamics(notes, beats)

    assert metrics is not None
    assert metrics[0] > 0.0
    assert metrics[1] is None


def test_short_beat_track_skips_empty_meter_phases(monkeypatch) -> None:
    monkeypatch.setattr(
        "lyrehelper.analysis.librosa.beat.beat_track",
        lambda **_kwargs: (120.0, np.asarray([10, 50])),
    )
    onset = np.zeros(80, dtype=np.float32)

    with np.errstate(invalid="raise"):
        beats = _beat_track(onset, 22050, [TempoPoint(0.0, 120.0, 1.0)], [])

    assert len(beats) == 2


def test_tempo_score_features_treat_a_step_change_as_locally_stable() -> None:
    values = [120.0] * 12 + [90.0] * 12
    result = AnalysisResult(
        24.0,
        [TempoPoint(float(index), bpm, 1.0) for index, bpm in enumerate(values)],
        [],
        [],
        [],
        105.0,
        90.0,
        120.0,
        15.0,
        0.0,
        0.0,
        "test",
    )

    step_median, step_p90, stable_ratio, _, low_frequency = _tempo_score_features(result)

    assert step_median == 0.0
    assert step_p90 == 0.0
    assert stable_ratio > 0.95
    assert low_frequency > 0.1


def test_correlated_timing_evidence_requires_tempo_and_grid_motion() -> None:
    assert _correlated_timing_human_evidence(0.040, 80.0) > 70.0
    assert _correlated_timing_human_evidence(0.040, 95.0) < 5.0
    assert _correlated_timing_human_evidence(0.005, 80.0) < 5.0


def test_calibrated_score_separates_correlated_timing_evidence() -> None:
    def event(start: float, midi_note: int) -> NoteEvent:
        return NoteEvent(start, start + 0.1, midi_note, 440.0, "test", 80, 1.0)

    beats = [
        BeatMarker(index * 0.5, 1.0, index % 4 == 0, index % 4 + 1)
        for index in range(49)
    ]
    machine_tempo = [TempoPoint(float(index), 120.0, 1.0) for index in range(25)]
    machine_notes = []
    for index in range(48):
        start = index * 0.25
        machine_notes.extend((event(start, 60), event(start, 64)))
    machine = AnalysisResult(
        12.0,
        machine_tempo,
        beats,
        [],
        [],
        120.0,
        120.0,
        120.0,
        0.0,
        0.0,
        0.0,
        "test",
        machine_notes,
    )

    human_values = (118.0, 119.0, 121.0, 120.0, 117.0, 122.0, 119.0,
                    121.0, 118.0, 120.0, 123.0, 119.0, 117.0)
    human_notes = []
    jitter = (0.025, -0.018, 0.032, -0.027)
    for index in range(48):
        start = index * 0.25 + jitter[index % len(jitter)]
        human_notes.append(event(start, 60))
        if index % 3 == 0:
            human_notes.append(event(start + 0.012, 64))
    human = AnalysisResult(
        12.0,
        [TempoPoint(float(index), bpm, 1.0) for index, bpm in enumerate(human_values)],
        beats,
        [],
        [],
        120.0,
        117.0,
        123.0,
        1.8,
        0.0,
        0.0,
        "test",
        human_notes,
    )

    update_performance_score(machine)
    update_performance_score(human)

    assert machine.human_score < 5.0
    assert human.human_score > 90.0


def test_detects_tempo_beats_key_and_unambiguous_chord() -> None:
    result = analyze_audio(tonal_click_track(), 22050)
    assert 115 <= result.average_bpm <= 125
    assert len(result.beats) >= 28
    assert any(beat.is_downbeat for beat in result.beats)
    assert result.keys[0].key == "C major"
    assert any(chord.chord == "C" and chord.function == "I" for chord in result.chords)
    detected = {note.midi_note for note in result.notes}
    assert {60, 64, 67}.issubset(detected)
    assert all(note.end > note.start and note.frequency > 0 for note in result.notes)


def test_key_tracking_ignores_one_short_conflicting_window() -> None:
    frame_rate = 8.0
    chroma = np.tile(MAJOR_PROFILE / MAJOR_PROFILE.sum(), (round(48 * frame_rate), 1))
    short_change = np.roll(MAJOR_PROFILE / MAJOR_PROFILE.sum(), 7)
    chroma[round(20 * frame_rate) : round(24 * frame_rate)] = short_change

    keys = _key_segments(chroma.astype(np.float32), frame_rate)

    assert keys[0].start == 0.0
    assert keys[-1].end == 48.0
    assert all(left.end == right.start for left, right in zip(keys, keys[1:]))
    assert len(keys) == 1
    assert keys[0].key == "C major"


def test_monophonic_melody_does_not_invent_chord_progression() -> None:
    notes = [
        NoteEvent(index * 0.5, index * 0.5 + 0.3, 60 + index % 8, 440.0, "note", 96, 1.0)
        for index in range(32)
    ]

    result = analyze_note_events(notes, 16.0, 22050)

    assert result.chords
    assert {item.chord for item in result.chords} == {"N"}
    assert result.keys[0].start == 0.0
    assert result.keys[-1].end == 16.0


def test_silence_has_stable_empty_result() -> None:
    result = analyze_audio(np.zeros(22050, dtype=np.float32), 22050)
    assert result.average_bpm == 0
    assert result.tempo == []


def test_dynamic_tempo_produces_nonuniform_accelerating_beats() -> None:
    sample_rate = 22050
    frame_rate = sample_rate / TEMPO_HOP
    duration = 24.0
    tempo = [TempoPoint(float(time), 84 + time * 3.0, 0.9) for time in range(25)]
    onset = np.zeros(round(duration * frame_rate) + 1, dtype=np.float32)
    cursor = 0.0
    while cursor < duration:
        frame = min(len(onset) - 1, round(cursor * frame_rate))
        onset[frame] = 1.0
        bpm = np.interp(cursor, [item.time for item in tempo], [item.bpm for item in tempo])
        cursor += 60.0 / bpm
    beats = _beat_track(onset, sample_rate, tempo, [])
    intervals = np.diff([beat.time for beat in beats])
    assert len(intervals) > 20
    assert np.median(intervals[: len(intervals) // 3]) > np.median(intervals[-len(intervals) // 3 :])


def test_step_tempo_does_not_create_reverse_valley() -> None:
    sample_rate = 22050
    frame_rate = sample_rate / TEMPO_HOP
    duration = 120.0
    onset = np.zeros(round(duration * frame_rate) + 1, dtype=np.float32)
    cursor = 0.0
    while cursor < duration:
        onset[min(len(onset) - 1, round(cursor * frame_rate))] = 1.0
        cursor += 60.0 / (80.0 if cursor < 54.0 else 100.0)
    onset = np.convolve(onset, np.array([0.18, 0.58, 1.0, 0.58, 0.18]), mode="same")

    tempo, _ = _tempo_track(onset.astype(np.float32), [], sample_rate)
    before = [point.bpm for point in tempo if 10.0 <= point.time < 50.0]
    transition = [point.bpm for point in tempo if 50.0 <= point.time < 65.0]
    after = [point.bpm for point in tempo if 70.0 <= point.time < 115.0]

    assert 78.0 <= float(np.median(before)) <= 82.0
    assert 98.0 <= float(np.median(after)) <= 102.0
    assert min(transition) >= min(float(np.median(before)), float(np.median(after))) - 1.0


def test_single_segment_tempo_outlier_does_not_disable_constant_grid(monkeypatch) -> None:
    estimates = iter((139.67, 139.67, 186.23, 139.67))
    monkeypatch.setattr(
        "lyrehelper.analysis.librosa.feature.tempo",
        lambda **_kwargs: np.array([next(estimates)]),
    )
    onset = np.zeros(round(28.0 * 22050 / TEMPO_HOP), dtype=np.float32)
    onset[::100] = 1.0

    consistent, retained = _segment_tempi_consistent(onset, 22050, 139.67)

    assert consistent
    assert retained == [139.67, 139.67, 139.67]


def test_two_repeated_segment_tempi_form_piecewise_levels() -> None:
    levels = _repeated_tempo_levels([80.75, 80.12, 100.35, 100.35], 80.12)

    assert levels == [80.435, 100.35]
    assert _repeated_tempo_levels([82.0, 87.0, 93.0, 99.0], 90.0) == []
