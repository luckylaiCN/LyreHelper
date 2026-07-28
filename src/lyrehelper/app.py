from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import QLockFile, QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from .audio_capture import AudioCaptureService, SyntheticCaptureService
from .config import load_config, save_config
from .pipeline import AnalysisPipeline
from .transcription_backend import (
    transcription_execution_device,
    transcription_runtime_available,
)
from .ui import MainWindow


def _configure_logging() -> None:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LyreHelper"
    base.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=base / "lyrehelper.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unattended Windows audio analysis monitor")
    parser.add_argument("--demo", action="store_true", help="run a generated studio signal")
    parser.add_argument("--no-tray", action="store_true", help="quit when the window is closed")
    parser.add_argument("--build-smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _configure_logging()
    if args.build_smoke_test:
        return 0 if transcription_runtime_available() else 2
    application = QApplication(sys.argv[:1])
    application.setApplicationName("LyreHelper")
    application.setOrganizationName("LyreHelper")
    application.setQuitOnLastWindowClosed(False)
    lock_path = Path(QStandardPaths.writableLocation(QStandardPaths.TempLocation)) / "lyrehelper.lock"
    instance_lock = QLockFile(str(lock_path))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(50):
        return 0
    config = load_config()
    if args.no_tray:
        config.keep_running_in_tray = False
        application.setQuitOnLastWindowClosed(True)
    analysis_device, accelerated = transcription_execution_device()
    pipeline = AnalysisPipeline(config)
    capture_type = SyntheticCaptureService if args.demo else AudioCaptureService
    capture = capture_type(config, pipeline.submit, pipeline.set_capture_status)

    def change_audio_source(device_name: str) -> None:
        config.device_name = device_name
        save_config(config)
        reconfigure = getattr(capture, "reconfigure", None)
        if reconfigure is not None:
            reconfigure()

    def change_recording_mode(recording_mode: str) -> None:
        config.recording_mode = recording_mode
        save_config(config)

    window = MainWindow(
        pipeline,
        config,
        change_audio_source,
        change_recording_mode,
        analysis_device,
    )
    application.aboutToQuit.connect(capture.stop)
    application.aboutToQuit.connect(pipeline.stop)
    pipeline.start()
    capture.start()
    window.show()
    if not accelerated and not config.cpu_warning_shown:
        def warn_about_cpu_analysis() -> None:
            QMessageBox.warning(
                window,
                "CPU analysis mode",
                "No GPU inference provider could be initialized. Neural transcription "
                "will use the CPU and may fall behind real-time audio. This warning will "
                "not be shown again.",
            )
            config.cpu_warning_shown = True
            save_config(config)

        QTimer.singleShot(0, warn_about_cpu_analysis)
    exit_code = application.exec()
    capture.stop()
    pipeline.stop()
    capture.join(timeout=2)
    pipeline.join(timeout=8)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
