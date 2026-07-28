from __future__ import annotations

import numpy as np

from lyrehelper.config import AppConfig
from lyrehelper.analysis import update_performance_score
from lyrehelper.models import AnalysisResult, BeatMarker, MonitorState, NoteEvent, TempoPoint
from lyrehelper.pipeline import (
    AnalysisPipeline,
    _analyze_recent_notes,
    _detected_meter,
    _merge_incremental_beats,
    _merge_incremental_notes,
    _merge_incremental_tempo,
    _valid_note_window,
    _viterbi_tempo,
)
from lyrehelper.transcription_backend import _preferred_providers


def empty_result(duration: float = 1.0) -> AnalysisResult:
    return AnalysisResult(duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "silent")


def note(start: float, end: float, pitch: int) -> NoteEvent:
    return NoteEvent(start, end, pitch, 440.0, "A4", 96, 0.9)


def test_incremental_note_merge_only_replaces_unstable_context() -> None:
    existing = [note(0.0, 1.0, 60), note(2.9, 3.2, 62), note(4.0, 5.0, 64)]
    detected = [note(0.5, 1.0, 65), note(1.8, 2.5, 67), note(4.0, 5.0, 69)]

    merged = _merge_incremental_notes(existing, detected, window_start=2.0)

    assert [(item.start, item.end, item.midi_note) for item in merged] == [
        (0.0, 1.0, 60),
        (2.9, 3.2, 62),
        (3.8, 4.5, 67),
        (6.0, 7.0, 69),
    ]


def test_auto_validation_uses_note_duration_coverage_not_note_count() -> None:
    fragments = [note(index * 0.2, index * 0.2 + 0.01, 60) for index in range(20)]
    sustained = [note(1.0, 1.8, 60)]

    fragment_start, fragment_coverage = _valid_note_window(
        fragments, 5.0, require_full_window=True
    )
    sustained_start, sustained_coverage = _valid_note_window(
        sustained, 5.0, require_full_window=True
    )

    assert fragment_start is None
    assert fragment_coverage < 0.15
    assert sustained_start == 1.0
    assert sustained_coverage >= 0.15


def test_gpu_execution_providers_are_preferred_over_cpu() -> None:
    available = ["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"]

    assert _preferred_providers(available) == [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_long_session_analysis_is_bounded_to_recent_context(monkeypatch) -> None:
    analyzed: list[tuple[list[NoteEvent], float]] = []

    def analyze(notes, duration, sample_rate):
        analyzed.append((notes, duration))
        return empty_result(duration)

    monkeypatch.setattr("lyrehelper.pipeline.analyze_note_events", analyze)
    notes = [note(1.0, 1.5, 60), note(90.0, 91.0, 64)]

    result, context_start = _analyze_recent_notes(notes, 100.0, 100)

    assert context_start == 76.0
    assert analyzed[0][1] == 24.0
    assert [(item.start, item.end) for item in analyzed[0][0]] == [(14.0, 15.0)]
    assert result.notes == notes


def test_committed_tempo_prefix_cannot_be_flattened_by_later_evidence() -> None:
    existing = [TempoPoint(float(time), 108.0 + time % 3) for time in range(31)]
    newly_flat = [TempoPoint(float(time), 110.0, 0.9) for time in range(31)]

    merged = _merge_incremental_tempo(
        existing,
        newly_flat,
        elapsed=30.0,
        mutable_seconds=12.0,
    )

    assert [point.bpm for point in merged if point.time < 18.0] == [
        point.bpm for point in existing if point.time < 18.0
    ]
    assert all(point.bpm == 110.0 for point in merged if point.time >= 18.0)


def test_incremental_merge_repairs_a_temporary_committed_tempo_segment() -> None:
    temporary = [TempoPoint(float(index), 150.0, 0.9) for index in range(31)]
    for index in (18, 19, 20):
        temporary[index] = TempoPoint(float(index), 86.0, 0.9)

    corrected = [TempoPoint(float(index), 150.0, 0.9) for index in range(21, 34)]
    merged = _merge_incremental_tempo(temporary, corrected, elapsed=33.0)

    assert all(point.bpm == 150.0 for point in merged if 18.0 <= point.time <= 20.0)


def test_meter_hysteresis_rejects_alternating_window_estimates() -> None:
    pipeline = AnalysisPipeline(AppConfig())
    for _ in range(3):
        meter, _ = pipeline._resolve_meter(4)
    assert meter == 4
    assert pipeline._stable_meter == 4

    for detected in (3, 6) * 8:
        meter, changed = pipeline._resolve_meter(detected)
        assert meter == 4
        assert not changed

    pipeline._resolve_meter(4)
    changes = [pipeline._resolve_meter(6) for _ in range(8)]
    assert changes[-1] == (6, True)


def test_incremental_beat_merge_keeps_meter_and_bar_phase_continuous() -> None:
    existing = [
        BeatMarker(float(index), 1.0, index % 4 == 0, index % 4 + 1)
        for index in range(10)
    ]
    detected = [
        BeatMarker(float(index), 1.0, index % 6 == 0, index % 6 + 1)
        for index in range(8, 13)
    ]

    assert _detected_meter(detected) == 6
    merged = _merge_incremental_beats(
        existing,
        detected,
        elapsed=12.0,
        meter=4,
        mutable_seconds=4.0,
    )

    assert [beat.beat_in_bar for beat in merged] == [
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1
    ]
    assert [beat.time for beat in merged if beat.is_downbeat] == [0.0, 4.0, 8.0, 12.0]


def test_incremental_tempo_uses_robust_anchor_and_removes_duplicate_times() -> None:
    existing = [TempoPoint(float(index), 100.0, 0.9) for index in range(14)]
    existing.extend((TempoPoint(13.0, 150.0, 0.4), TempoPoint(14.0, 150.0, 0.4)))
    detected = [TempoPoint(float(index), 50.0, 0.9) for index in range(14, 27)]

    merged = _merge_incremental_tempo(existing, detected, elapsed=26.0)

    assert len({round(point.time, 4) for point in merged}) == len(merged)
    assert all(95.0 <= point.bpm <= 105.0 for point in merged[-10:])


def test_viterbi_tempo_ignores_isolated_wobble_but_keeps_sustained_step() -> None:
    noisy = [TempoPoint(float(index), value, 0.9) for index, value in enumerate((110, 110, 111, 110, 110))]
    stepped = [
        TempoPoint(float(index), value, 0.9)
        for index, value in enumerate((110, 110, 110, 120, 120, 120, 120))
    ]

    stable = _viterbi_tempo(noisy, 110.0)
    changed = _viterbi_tempo(stepped, 110.0)

    assert {point.bpm for point in stable} == {110.0}
    assert [point.bpm for point in changed[:3]] == [110.0, 110.0, 110.0]
    assert [point.bpm for point in changed[-3:]] == [120.0, 120.0, 120.0]


def test_viterbi_tempo_folds_half_and_double_time_observations() -> None:
    points = [
        TempoPoint(0.0, 120.0, 0.9),
        TempoPoint(1.0, 60.0, 0.9),
        TempoPoint(2.0, 120.0, 0.9),
        TempoPoint(3.0, 240.0, 0.9),
        TempoPoint(4.0, 120.0, 0.9),
    ]

    normalized = _viterbi_tempo(points, 120.0)

    assert [point.bpm for point in normalized] == [120.0] * 5


def test_viterbi_tempo_rejects_short_four_thirds_metric_excursion() -> None:
    values = (120, 120, 131, 139, 147, 160, 160, 147, 139, 131, 120, 120)
    points = [TempoPoint(float(index), value, 0.9) for index, value in enumerate(values)]

    normalized = _viterbi_tempo(points, 120.0)

    assert max(point.bpm for point in normalized) - min(point.bpm for point in normalized) <= 4.0


def test_viterbi_tempo_preserves_coherent_human_scale_drift() -> None:
    values = (80.0, 81.0, 80.0, 79.0, 78.0)
    points = [TempoPoint(float(index), value, 0.9) for index, value in enumerate(values)]

    normalized = _viterbi_tempo(points, 80.0)

    assert [point.bpm for point in normalized] == list(values)


def test_viterbi_tempo_merges_two_second_false_segment_between_matching_levels() -> None:
    values = (120.0, 120.0, 116.0, 116.0, 120.0, 120.0)
    points = [TempoPoint(float(index), value, 0.9) for index, value in enumerate(values)]

    normalized = _viterbi_tempo(points, 120.0)

    assert [point.bpm for point in normalized] == [120.0] * len(values)


def test_performance_score_distinguishes_grid_accuracy_at_same_bpm() -> None:
    tempo = [TempoPoint(float(index), 120.0, 1.0) for index in range(9)]
    beats = [BeatMarker(index * 0.5, 1.0, index % 4 == 0, index % 4 + 1) for index in range(17)]
    machine_notes = [note(index * 0.125, index * 0.125 + 0.08, 60) for index in range(64)]
    jitter = (0.04, -0.04, 0.03, -0.03)
    human_notes = [
        note(index * 0.125 + jitter[index % len(jitter)], index * 0.125 + jitter[index % len(jitter)] + 0.08, 60)
        for index in range(1, 64)
    ]
    machine = AnalysisResult(8.0, tempo, beats, [], [], 120, 120, 120, 0, 0, 0, "test", machine_notes)
    human = AnalysisResult(8.0, tempo, beats, [], [], 120, 120, 120, 0, 0, 0, "test", human_notes)

    update_performance_score(machine)
    update_performance_score(human)

    assert machine.grid_accuracy > 95.0
    assert human.grid_accuracy < machine.grid_accuracy - 25.0
    assert machine.mechanical_index > human.mechanical_index + 10.0


def test_recording_modes_control_session_start() -> None:
    config = AppConfig(sample_rate=100, block_duration=0.25, trigger_seconds=0.5)
    pipeline = AnalysisPipeline(config)
    signal = np.full(25, 0.1, dtype=np.float32)

    pipeline._handle_recording_mode("off")
    pipeline._process(signal)
    pipeline._process(signal)
    assert not pipeline._active

    pipeline._handle_recording_mode("on")
    pipeline._process(signal)
    assert pipeline._active
    assert pipeline.get_snapshot().recording_mode == "on"


def test_analysis_latency_tracks_age_of_latest_analyzed_audio(monkeypatch) -> None:
    pipeline = AnalysisPipeline(AppConfig())
    pipeline._active = True
    pipeline._analysis_reference_capture_at = 97.5
    monkeypatch.setattr("lyrehelper.pipeline.time.monotonic", lambda: 100.0)

    assert pipeline.get_snapshot().analysis_latency == 2.5


def test_auto_candidate_display_stays_empty_and_waiting() -> None:
    pipeline = AnalysisPipeline(AppConfig(recording_mode="auto"))
    pipeline._active = True
    pipeline._snapshot.state = MonitorState.ANALYZING
    pipeline._snapshot.auto_candidate = True
    pipeline._snapshot.session_id = "candidate"
    pipeline._snapshot.elapsed = 4.0
    pipeline._snapshot.playhead = 4.0
    pipeline._snapshot.notes = [note(1.0, 2.0, 60)]

    display = pipeline.get_display_snapshot()

    assert display.state.value == "standby"
    assert display.session_id is None
    assert display.elapsed == 0.0
    assert display.playhead == 0.0
    assert display.notes == []
    assert pipeline.get_snapshot().elapsed == 4.0


def test_verified_auto_display_rebases_to_shared_archive_origin() -> None:
    pipeline = AnalysisPipeline(AppConfig(recording_mode="auto"))
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._auto_session_validated = True
    pipeline._auto_valid_start = 5.0
    pipeline._snapshot.auto_candidate = False
    pipeline._snapshot.session_id = "verified"
    pipeline._snapshot.elapsed = 10.0
    pipeline._snapshot.playhead = 10.0
    pipeline._snapshot.notes = [note(5.0, 5.5, 60), note(6.0, 6.5, 64)]

    display = pipeline.get_display_snapshot()

    assert round(display.elapsed, 1) == 5.2
    assert round(display.playhead, 1) == 5.2
    assert [round(item.start, 1) for item in display.notes] == [0.2, 1.2]
    assert pipeline.get_snapshot().notes[0].start == 5.0


def test_manual_cut_finalizes_and_starts_three_second_cooldown(monkeypatch) -> None:
    pipeline = AnalysisPipeline(AppConfig(sample_rate=100))
    pipeline._active = True
    pipeline._blocks = [np.full(25, 0.1, dtype=np.float32)]
    finalized: list[bool] = []

    def finalize() -> None:
        finalized.append(True)
        pipeline._active = False
        pipeline._blocks = []

    monkeypatch.setattr(pipeline, "_finalize", finalize)
    pipeline._handle_manual_cut()

    snapshot = pipeline.get_snapshot()
    assert finalized == [True]
    assert not pipeline._active
    assert snapshot.cooldown_remaining > 2.9
    assert snapshot.recording_mode == "auto"
    assert pipeline._recording_mode == "auto"
    assert pipeline.config.recording_mode == "auto"


def test_pause_without_audio_creates_no_archive_but_still_resumes_in_auto(monkeypatch) -> None:
    pipeline = AnalysisPipeline(AppConfig(recording_mode="off"))
    finalized: list[bool] = []
    monkeypatch.setattr(pipeline, "_finalize", lambda: finalized.append(True))

    pipeline._handle_manual_cut()

    snapshot = pipeline.get_snapshot()
    assert finalized == []
    assert snapshot.recording_mode == "auto"
    assert snapshot.cooldown_remaining > 2.9


def test_terminate_archives_active_session_and_remains_off(monkeypatch) -> None:
    pipeline = AnalysisPipeline(AppConfig(recording_mode="on", sample_rate=100))
    pipeline._active = True
    pipeline._blocks = [np.full(25, 0.1, dtype=np.float32)]
    finalized: list[bool] = []

    def finalize() -> None:
        finalized.append(True)
        pipeline._active = False
        pipeline._blocks = []

    monkeypatch.setattr(pipeline, "_finalize", finalize)
    pipeline._handle_recording_mode("off")

    assert finalized == [True]
    assert pipeline.get_snapshot().recording_mode == "off"
    assert pipeline._recording_mode == "off"
    assert pipeline.config.recording_mode == "off"


def test_explicit_start_cancels_pause_cooldown() -> None:
    pipeline = AnalysisPipeline(AppConfig(sample_rate=100, recording_mode="auto"))
    signal = np.full(25, 0.1, dtype=np.float32)
    pipeline._handle_manual_cut()

    pipeline._handle_recording_mode("on")
    pipeline._process(signal)

    assert pipeline._active
    assert pipeline.get_snapshot().cooldown_remaining == 0.0


def test_pause_uses_capture_time_and_next_signal_starts_a_new_session(monkeypatch) -> None:
    config = AppConfig(sample_rate=100, block_duration=0.25, trigger_seconds=0.5)
    pipeline = AnalysisPipeline(config)
    signal = np.full(25, 0.1, dtype=np.float32)
    finalized_sessions: list[str | None] = []

    pipeline._handle_recording_mode("on")
    pipeline._process(signal)
    first_session = pipeline.get_snapshot().session_id

    def finalize() -> None:
        finalized_sessions.append(pipeline.get_snapshot().session_id)
        pipeline._active = False
        pipeline._blocks = []
        pipeline._total_samples = 0

    monkeypatch.setattr(pipeline, "_finalize", finalize)
    requested_at = 100.0
    pipeline._handle_manual_cut(requested_at)
    pipeline._process(signal, captured_at=102.9)
    assert not pipeline._active

    pipeline._process(signal, captured_at=103.1)
    pipeline._process(signal, captured_at=103.35)

    assert finalized_sessions == [first_session]
    assert pipeline._active
    assert pipeline.get_snapshot().session_id != first_session


def test_pause_rearms_auto_without_starting_on_silence(monkeypatch) -> None:
    config = AppConfig(
        sample_rate=100,
        block_duration=0.25,
        trigger_seconds=0.5,
        signal_threshold_db=-30,
        recording_mode="on",
    )
    pipeline = AnalysisPipeline(config)
    signal = np.full(25, 0.1, dtype=np.float32)
    silence = np.zeros(25, dtype=np.float32)
    pipeline._active = True
    pipeline._blocks = [signal]

    def finalize() -> None:
        pipeline._active = False
        pipeline._blocks = []
        pipeline._total_samples = 0

    monkeypatch.setattr(pipeline, "_finalize", finalize)
    pipeline._handle_manual_cut(requested_at=100.0)

    for captured_at in (103.1, 103.35, 103.6, 103.85):
        pipeline._process(silence, captured_at=captured_at)
    assert not pipeline._active
    assert pipeline.get_snapshot().recording_mode == "auto"

    pipeline._process(signal, captured_at=104.1)
    assert not pipeline._active
    pipeline._process(signal, captured_at=104.35)
    assert pipeline._active


def test_sparse_auto_session_is_discarded_without_archive(monkeypatch, tmp_path) -> None:
    pipeline = AnalysisPipeline(
        AppConfig(output_directory=str(tmp_path), sample_rate=100, recording_mode="auto")
    )
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._blocks = [np.full(100, 0.1, dtype=np.float32)]
    pipeline._total_samples = 100
    pipeline._snapshot.session_id = "sparse-auto"
    pipeline._snapshot.notes = [note(0.1, 0.4, 60), note(0.5, 0.8, 64)]
    exported: list[bool] = []
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [])
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_session", lambda *args: exported.append(True)
    )

    pipeline._finalize()

    assert exported == []
    assert not pipeline._active
    assert pipeline.get_snapshot().last_archive is None
    assert not list(tmp_path.iterdir())


def test_auto_session_shorter_than_twenty_seconds_is_discarded_before_transcription(
    monkeypatch, tmp_path
) -> None:
    pipeline = AnalysisPipeline(
        AppConfig(output_directory=str(tmp_path), sample_rate=100, recording_mode="auto")
    )
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._blocks = [np.full(1999, 0.1, dtype=np.float32)]
    pipeline._total_samples = 1999
    pipeline._snapshot.session_id = "short-auto"
    monkeypatch.setattr(
        "lyrehelper.pipeline.transcribe_notes",
        lambda *args: (_ for _ in ()).throw(AssertionError("short AUTO was transcribed")),
    )

    pipeline._finalize()

    assert not pipeline._active
    assert not list(tmp_path.iterdir())


def test_sparse_on_session_is_still_archived(monkeypatch, tmp_path) -> None:
    pipeline = AnalysisPipeline(
        AppConfig(output_directory=str(tmp_path), sample_rate=100, recording_mode="on")
    )
    pipeline._active = True
    pipeline._session_recording_mode = "on"
    pipeline._blocks = [np.full(100, 0.1, dtype=np.float32)]
    pipeline._total_samples = 100
    pipeline._snapshot.session_id = "sparse-on"
    exported: list[str] = []
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [])
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_audio", lambda audio, rate: empty_result(len(audio) / rate)
    )
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_session",
        lambda result, output, session: (
            exported.append(session) or tmp_path / "x.mid",
            tmp_path / "x.csv",
        ),
    )
    monkeypatch.setattr("lyrehelper.pipeline.export_audio", lambda *args: tmp_path / "x.wav")

    pipeline._finalize()

    assert exported == ["sparse-on"]
    assert not pipeline._active


def test_labeled_session_writes_additional_training_copy(monkeypatch, tmp_path) -> None:
    pipeline = AnalysisPipeline(
        AppConfig(output_directory=str(tmp_path), sample_rate=100, recording_mode="on")
    )
    pipeline._labels_directory = tmp_path / "labels"
    pipeline._active = True
    pipeline._session_recording_mode = "on"
    pipeline._blocks = [np.full(100, 0.1, dtype=np.float32)]
    pipeline._total_samples = 100
    pipeline._snapshot.session_id = "labeled-on"
    pipeline.set_session_label("human")
    export_directories: list[object] = []
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [])
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_audio", lambda audio, rate: empty_result(len(audio) / rate)
    )

    def export_result(result, output, session):
        export_directories.append(output)
        return output / f"{session}_transcription.mid", output / f"{session}_chords.csv"

    monkeypatch.setattr("lyrehelper.pipeline.export_session", export_result)
    monkeypatch.setattr("lyrehelper.pipeline.export_audio", lambda *args: None)

    pipeline._finalize()

    assert export_directories == [tmp_path, tmp_path / "labels" / "human"]
    assert pipeline.get_snapshot().session_label == "none"


def test_auto_candidate_keeps_sliding_after_five_seconds_without_enough_notes(
    monkeypatch,
) -> None:
    pipeline = AnalysisPipeline(AppConfig(sample_rate=100, recording_mode="auto"))
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._blocks = [np.full(500, 0.1, dtype=np.float32)]
    pipeline._total_samples = 500
    pipeline._snapshot.session_id = "invalid-candidate"
    monkeypatch.setattr(
        "lyrehelper.pipeline.transcribe_notes",
        lambda audio, rate: [note(0.5, 0.8, 60), note(1.0, 1.3, 64)],
    )
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_note_events",
        lambda notes, duration, rate: AnalysisResult(
            duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "candidate", notes
        ),
    )

    pipeline._refresh(captured_at=100.0)

    assert pipeline._active
    assert pipeline.get_snapshot().auto_candidate


def test_auto_candidate_is_confirmed_by_four_notes_in_five_seconds(monkeypatch) -> None:
    pipeline = AnalysisPipeline(AppConfig(sample_rate=100, recording_mode="auto"))
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._blocks = [np.full(500, 0.1, dtype=np.float32)]
    pipeline._total_samples = 500
    pipeline._snapshot.session_id = "valid-candidate"
    detected = [
        note(0.5, 0.8, 60),
        note(1.0, 1.3, 62),
        note(1.5, 1.8, 64),
        note(2.0, 2.3, 65),
    ]
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: detected)
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_note_events",
        lambda notes, duration, rate: AnalysisResult(
            duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "note-driven", notes
        ),
    )

    pipeline._refresh(captured_at=100.0)

    assert pipeline._active
    assert pipeline._auto_session_validated
    assert not pipeline.get_snapshot().auto_candidate


def test_auto_candidate_window_validates_when_music_begins_at_4_9_seconds(
    monkeypatch,
) -> None:
    pipeline = AnalysisPipeline(AppConfig(sample_rate=100, recording_mode="auto"))
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._blocks = [np.full(600, 0.1, dtype=np.float32)]
    pipeline._total_samples = 600
    pipeline._snapshot.session_id = "boundary-candidate"
    detected = [
        note(4.9, 5.15, 60),
        note(5.2, 5.45, 62),
        note(5.5, 5.75, 64),
        note(5.8, 6.0, 65),
    ]
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: detected)
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_note_events",
        lambda notes, duration, rate: AnalysisResult(
            duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "note-driven", notes
        ),
    )

    pipeline._refresh(captured_at=100.0)

    assert pipeline._auto_session_validated
    assert pipeline._auto_valid_start == 4.9


def test_validated_auto_session_pauses_when_latest_note_coverage_falls_below_ten_percent(
    monkeypatch,
) -> None:
    pipeline = AnalysisPipeline(AppConfig(sample_rate=100, recording_mode="auto"))
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._auto_session_validated = True
    pipeline._blocks = [np.full(3000, 0.1, dtype=np.float32)]
    pipeline._total_samples = 3000
    pipeline._snapshot.session_id = "low-coverage"
    pipeline._snapshot.notes = [
        note(1.0, 1.3, 60),
        note(1.5, 1.8, 62),
        note(2.0, 2.3, 64),
        note(2.5, 2.8, 65),
        note(27.0, 27.4, 67),
    ]
    pauses: list[float | None] = []
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [])
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_note_events",
        lambda notes, duration, rate: empty_result(duration),
    )
    monkeypatch.setattr(pipeline, "_handle_manual_cut", lambda when=None: pauses.append(when))

    pipeline._refresh(captured_at=100.0)

    assert pauses == [100.0]


def test_auto_session_trims_non_note_lead_in_and_shifts_timeline(monkeypatch, tmp_path) -> None:
    pipeline = AnalysisPipeline(
        AppConfig(output_directory=str(tmp_path), sample_rate=100, recording_mode="auto")
    )
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._auto_session_validated = True
    pipeline._blocks = [np.full(2500, 0.1, dtype=np.float32)]
    pipeline._total_samples = 2500
    pipeline._snapshot.session_id = "trimmed-auto"
    detected = [
        note(2.0, 2.3, 60),
        note(2.5, 2.8, 62),
        note(3.0, 3.3, 64),
        note(3.5, 3.8, 65),
    ]
    pipeline._snapshot.notes = detected
    exported_results: list[AnalysisResult] = []
    exported_audio: list[np.ndarray] = []
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [])
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_note_events",
        lambda notes, duration, rate: AnalysisResult(
            duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "note-driven", notes
        ),
    )
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_session",
        lambda result, output, session: (
            exported_results.append(result) or tmp_path / "x.mid",
            tmp_path / "x.csv",
        ),
    )
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_audio",
        lambda audio, rate, output, session: (
            exported_audio.append(audio.copy()) or tmp_path / "x.wav"
        ),
    )

    pipeline._finalize()

    assert len(exported_audio[0]) == 2320
    assert exported_results[0].duration == 23.2
    assert [round(item.start, 1) for item in exported_results[0].notes] == [
        0.2,
        0.7,
        1.2,
        1.7,
    ]


def test_auto_session_below_twenty_seconds_after_lead_in_trim_is_discarded(
    monkeypatch, tmp_path
) -> None:
    pipeline = AnalysisPipeline(
        AppConfig(output_directory=str(tmp_path), sample_rate=100, recording_mode="auto")
    )
    pipeline._active = True
    pipeline._session_recording_mode = "auto"
    pipeline._auto_session_validated = True
    pipeline._blocks = [np.full(2100, 0.1, dtype=np.float32)]
    pipeline._total_samples = 2100
    pipeline._snapshot.session_id = "short-after-trim"
    pipeline._snapshot.notes = [
        note(2.0, 2.2, 60),
        note(2.25, 2.45, 62),
        note(2.5, 2.7, 64),
        note(2.75, 2.95, 65),
    ]
    exported: list[bool] = []
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [])
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_session", lambda *args: exported.append(True)
    )

    pipeline._finalize()

    assert exported == []
    assert not pipeline._active


def test_audio_backlog_cannot_evict_pause_command() -> None:
    pipeline = AnalysisPipeline(AppConfig())
    signal = np.ones(16, dtype=np.float32)
    for _ in range(pipeline._audio_backlog_limit):
        pipeline.submit(signal)
    pipeline.pause_recording()
    for _ in range(10):
        pipeline.submit(signal)

    queued = list(pipeline._queue.queue)
    assert any(item.__class__.__name__ == "_PauseRecordingCommand" for item in queued)


def test_short_dropout_continues_but_three_seconds_ends_auto_session(
    monkeypatch, tmp_path
) -> None:
    config = AppConfig(
        output_directory=str(tmp_path),
        sample_rate=100,
        block_duration=0.25,
        signal_threshold_db=-30,
        trigger_seconds=0.5,
        end_silence_seconds=3.0,
    )
    pipeline = AnalysisPipeline(config)
    exports: list[str] = []
    monkeypatch.setattr(pipeline, "_refresh", lambda captured_at=None: None)
    monkeypatch.setattr("lyrehelper.pipeline.analyze_audio", lambda audio, rate: empty_result(len(audio) / rate))
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_session",
        lambda result, output, session: (exports.append(session) or tmp_path / "x.mid", tmp_path / "x.csv"),
    )
    signal = np.full(25, 0.1, dtype=np.float32)
    silence = np.zeros(25, dtype=np.float32)
    pipeline._process(signal)
    pipeline._process(signal)
    assert pipeline._active
    session_id = pipeline.get_snapshot().session_id
    pipeline._snapshot.notes = [
        note(0.1, 0.3, 60),
        note(0.4, 0.6, 62),
        note(0.7, 0.9, 64),
        note(1.0, 1.2, 65),
    ]
    detected = list(pipeline._snapshot.notes)
    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: detected)
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_note_events",
        lambda notes, duration, rate: AnalysisResult(
            duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "note-driven", notes
        ),
    )
    for _ in range(8):
        pipeline._process(silence)
    pipeline._process(signal)
    assert pipeline._active
    assert pipeline.get_snapshot().session_id == session_id
    for _ in range(12):
        pipeline._process(silence)
    assert not pipeline._active
    assert exports == []


def test_finalize_preserves_incremental_notes_and_only_refreshes_tail(monkeypatch, tmp_path) -> None:
    config = AppConfig(output_directory=str(tmp_path), sample_rate=100)
    pipeline = AnalysisPipeline(config)
    pipeline._blocks = [np.full(2000, 0.1, dtype=np.float32)]
    pipeline._total_samples = 2000
    pipeline._active = True
    pipeline._snapshot.session_id = "incremental"
    pipeline._snapshot.notes = [
        note(1.0, 1.5, 60),
        note(2.0, 2.5, 62),
        note(3.0, 3.5, 64),
        note(4.0, 4.5, 65),
    ]
    exported: list[AnalysisResult] = []

    monkeypatch.setattr("lyrehelper.pipeline.transcribe_notes", lambda audio, rate: [note(3.0, 3.5, 67)])

    def analyze_notes(notes, duration, rate):
        return AnalysisResult(duration, [], [], [], [], 0, 0, 0, 0, 0, 0, "note-driven", notes)

    monkeypatch.setattr("lyrehelper.pipeline.analyze_note_events", analyze_notes)
    monkeypatch.setattr(
        "lyrehelper.pipeline.analyze_audio",
        lambda audio, rate: (_ for _ in ()).throw(AssertionError("full pass must not replace live notes")),
    )
    monkeypatch.setattr(
        "lyrehelper.pipeline.export_session",
        lambda result, output, session: (
            exported.append(result) or tmp_path / "x.mid",
            tmp_path / "x.csv",
        ),
    )

    pipeline._finalize()

    assert len(exported) == 1
    assert [(item.start, item.midi_note) for item in exported[0].notes] == [
        (1.0, 60),
        (2.0, 62),
        (3.0, 64),
        (4.0, 65),
        (11.0, 67),
    ]
