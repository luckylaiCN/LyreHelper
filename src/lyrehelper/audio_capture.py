from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from .config import AppConfig
from .models import MonitorState

logger = logging.getLogger(__name__)


def list_output_devices() -> list[str]:
    try:
        import soundcard as sc

        return list(dict.fromkeys(speaker.name for speaker in sc.all_speakers()))
    except Exception as error:
        logger.warning("Unable to enumerate output devices: %s", error)
        return []


class AudioCaptureService(threading.Thread):
    """Continuously reconnecting WASAPI loopback capture with no modal failures."""

    def __init__(
        self,
        config: AppConfig,
        on_audio: Callable[[np.ndarray], None],
        on_status: Callable[[MonitorState, str], None],
    ) -> None:
        super().__init__(name="audio-capture", daemon=True)
        self.config = config
        self.on_audio = on_audio
        self.on_status = on_status
        self._stopping = threading.Event()
        self._reconfigure = threading.Event()

    def stop(self) -> None:
        self._stopping.set()
        self._reconfigure.set()

    def reconfigure(self) -> None:
        self.on_status(MonitorState.CONNECTING, "Switching audio source")
        self._reconfigure.set()

    def _select_speaker(self, soundcard: object) -> object:
        requested = self.config.device_name.strip().casefold()
        if requested:
            matches = [item for item in soundcard.all_speakers() if requested in item.name.casefold()]
            if matches:
                return matches[0]
        return soundcard.default_speaker()

    def run(self) -> None:
        retry_delay = 1.0
        while not self._stopping.is_set():
            self._reconfigure.clear()
            try:
                import soundcard as sc

                speaker = self._select_speaker(sc)
                if speaker is None:
                    raise RuntimeError("No Windows output device is available")
                loopback = sc.get_microphone(id=str(speaker.id), include_loopback=True)
                self.on_status(MonitorState.STANDBY, speaker.name)
                frames = max(256, int(self.config.sample_rate * self.config.block_duration))
                retry_delay = 1.0
                with loopback.recorder(
                    samplerate=self.config.sample_rate,
                    channels=2,
                    blocksize=frames,
                ) as recorder:
                    while not self._stopping.is_set() and not self._reconfigure.is_set():
                        block = recorder.record(numframes=frames)
                        if block.ndim == 2:
                            block = block.mean(axis=1)
                        self.on_audio(np.asarray(block, dtype=np.float32))
            except Exception as error:  # Device failures are deliberately non-interactive.
                logger.warning("Audio capture unavailable; retrying: %s", error)
                self.on_status(MonitorState.CONNECTING, "Reconnecting to Windows audio")
                self._reconfigure.wait(retry_delay)
                retry_delay = min(15.0, retry_delay * 1.7)


class SyntheticCaptureService(threading.Thread):
    """Deterministic demonstration source used by --demo and UI smoke tests."""

    def __init__(
        self,
        config: AppConfig,
        on_audio: Callable[[np.ndarray], None],
        on_status: Callable[[MonitorState, str], None],
    ) -> None:
        super().__init__(name="demo-capture", daemon=True)
        self.config = config
        self.on_audio = on_audio
        self.on_status = on_status
        self._stopping = threading.Event()

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        self.on_status(MonitorState.STANDBY, "Demo studio feed")
        sample_rate = self.config.sample_rate
        size = int(sample_rate * self.config.block_duration)
        cursor = 0
        while not self._stopping.is_set():
            time_axis = (np.arange(size) + cursor) / sample_rate
            bpm = 108 + 3.8 * np.sin(2 * np.pi * time_axis / 18)
            beat_phase = np.mod(time_axis * bpm / 60, 1.0)
            click = 0.72 * np.exp(-beat_phase * 55) * np.sin(2 * np.pi * 1100 * time_axis)
            chord_roots = np.array([261.63, 349.23, 392.00, 261.63])
            chord_index = (time_axis // 4).astype(int) % 4
            root = chord_roots[chord_index]
            harmony = 0.12 * (
                np.sin(2 * np.pi * root * time_axis)
                + np.sin(2 * np.pi * root * 1.25 * time_axis)
                + np.sin(2 * np.pi * root * 1.5 * time_axis)
            )
            self.on_audio((click + harmony).astype(np.float32))
            cursor += size
            self._stopping.wait(self.config.block_duration)
