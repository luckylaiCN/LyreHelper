from __future__ import annotations

from lyrehelper.models import TempoPoint
from lyrehelper.ui import (
    _cadence_label,
    _smoothed_tempo_segments,
    _smoothed_tempo_values,
)


def test_thirty_second_tempo_line_is_causal() -> None:
    points = [TempoPoint(float(index * 10), 60.0 + index * 10, 1.0) for index in range(5)]

    smoothed = _smoothed_tempo_values(points, 30.0)

    assert smoothed[-1] == (40.0, 85.0)
    assert smoothed[1] == (10.0, 65.0)


def test_thirty_second_tempo_line_resets_and_breaks_at_confirmed_jumps() -> None:
    points = [
        TempoPoint(float(index), bpm, 1.0)
        for index, bpm in enumerate((90.0, 91.0, 90.0, 78.0, 79.0, 78.0, 124.0, 125.0))
    ]

    segments = _smoothed_tempo_segments(points)

    assert [[time for time, _ in segment] for segment in segments] == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0],
    ]
    assert segments[1][0] == (3.0, 78.0)
    assert segments[2][0] == (6.0, 124.0)


def test_thirty_second_tempo_line_keeps_human_scale_motion_continuous() -> None:
    points = [
        TempoPoint(float(index), bpm, 1.0)
        for index, bpm in enumerate((80.0, 81.0, 80.0, 79.0, 78.0))
    ]

    segments = _smoothed_tempo_segments(points)

    assert len(segments) == 1
    assert segments[0][-1] == (4.0, 79.6)


def test_harmony_summary_names_common_cadences() -> None:
    assert _cadence_label(["ii", "V7", "I"]) == "AUTHENTIC CADENCE"
    assert _cadence_label(["IV", "I"]) == "PLAGAL CADENCE"
