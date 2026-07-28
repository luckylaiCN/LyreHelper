from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    device_name: str = ""
    output_directory: str = ""
    sample_rate: int = 22050
    block_duration: float = 0.25
    signal_threshold_db: float = -48.0
    trigger_seconds: float = 0.6
    end_silence_seconds: float = 3.0
    keep_running_in_tray: bool = True
    recording_mode: str = "auto"
    cpu_warning_shown: bool = False

    @property
    def output_path(self) -> Path:
        if self.output_directory:
            return Path(os.path.expandvars(self.output_directory)).expanduser()
        return runtime_directory() / ".LyreHelper" / "output"


def runtime_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home()))
    return base / "LyreHelper" / "settings.json"


def load_config(path: Path | None = None) -> AppConfig:
    target = path or config_path()
    defaults = AppConfig()
    if not target.exists():
        save_config(defaults, target)
        return defaults
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        known = {key: value for key, value in data.items() if hasattr(defaults, key)}
        return AppConfig(**known)
    except (OSError, ValueError, TypeError):
        return defaults


def save_config(config: AppConfig, path: Path | None = None) -> None:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)
