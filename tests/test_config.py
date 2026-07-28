from __future__ import annotations

from lyrehelper.config import AppConfig


def test_default_output_path_is_hidden_directory_under_runtime_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    assert AppConfig().output_path == tmp_path / ".LyreHelper" / "output"
