from __future__ import annotations

import sys

from lyrehelper.config import AppConfig, load_config, save_config


def test_default_output_path_is_hidden_directory_under_runtime_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    assert AppConfig().output_path == tmp_path / ".LyreHelper" / "output"


def test_frozen_build_uses_executable_directory(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "LyreHelper.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert AppConfig().output_path == tmp_path / ".LyreHelper" / "output"


def test_cpu_warning_acknowledgement_is_persisted(tmp_path) -> None:
    path = tmp_path / "settings.json"
    config = AppConfig(cpu_warning_shown=True)

    save_config(config, path)

    assert load_config(path).cpu_warning_shown is True
