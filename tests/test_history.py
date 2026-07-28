from __future__ import annotations

import numpy as np

from lyrehelper.exporters import export_audio, export_session
from lyrehelper.history import list_history, load_history_snapshot
from tests.test_exporters import result_fixture


def test_history_restores_archived_audio_notes_and_tempo(tmp_path) -> None:
    result = result_fixture()
    export_session(result, tmp_path, "20260722_120000_001")
    export_audio(np.zeros(22050 * 4, dtype=np.float32), 22050, tmp_path, "20260722_120000_001")

    entries = list_history(tmp_path)
    snapshot = load_history_snapshot(entries[0])

    assert len(entries) == 1
    assert entries[0].audio_path is not None
    assert snapshot.session_id == "20260722_120000_001"
    assert snapshot.elapsed == 4.0
    assert [round(point.bpm) for point in snapshot.tempo] == [120, 123]
    assert [(note.midi_note, note.velocity) for note in snapshot.notes] == [(60, 96)]
    assert snapshot.chords[0].chord == "C"
