from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import mido
import numpy as np
import soundfile as sf

PITCH_NAMES = ("c", "c_", "d", "d_", "e", "f", "f_", "g", "g_", "a", "a_", "b")
SAMPLE_DIRECTORY_ENV = "LYREHELPER_SAMPLE_DIR"


def midi_note_starts(path: Path) -> tuple[list[tuple[float, int]], float]:
    midi = mido.MidiFile(path)
    tempo = 500_000
    elapsed = 0.0
    events: list[tuple[float, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "note_on" and message.velocity > 0:
            events.append((elapsed, message.note))
    return events, elapsed


def sample_path(sample_directory: Path, midi_note: int) -> Path:
    resource_octave = midi_note // 12 - 2
    name = PITCH_NAMES[midi_note % 12]
    return sample_directory / f"{name}{resource_octave}.wav"


def render_midi(
    midi_path: Path,
    sample_directory: Path,
    output_path: Path,
    sample_rate: int = 22050,
    duration_limit: float | None = None,
) -> Path:
    events, midi_duration = midi_note_starts(midi_path)
    duration = min(midi_duration, duration_limit) if duration_limit else midi_duration
    cache: dict[int, np.ndarray] = {}
    audio = np.zeros(round((duration + 2.0) * sample_rate), dtype=np.float32)
    for start, midi_note in events:
        if start >= duration:
            continue
        if midi_note not in cache:
            source, _ = librosa.load(sample_path(sample_directory, midi_note), sr=sample_rate, mono=True)
            # Resource C3 and MIDI C4 name the same central-C frequency.
            cache[midi_note] = source.astype(np.float32)
        sample = cache[midi_note]
        offset = round(start * sample_rate)
        length = min(len(sample), len(audio) - offset)
        audio[offset : offset + length] += sample[:length]
    audio = audio[: round(duration * sample_rate)]
    peak = float(np.max(np.abs(audio), initial=0))
    if peak > 0:
        audio *= 0.92 / peak
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render MIDI with fixed-gain game instrument samples")
    parser.add_argument("midi", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help=(
            "External WAV sample directory. Defaults to the "
            f"{SAMPLE_DIRECTORY_ENV} environment variable."
        ),
    )
    parser.add_argument("--limit", type=float, default=None)
    args = parser.parse_args()
    sample_directory = args.samples or (
        Path(os.environ[SAMPLE_DIRECTORY_ENV])
        if os.environ.get(SAMPLE_DIRECTORY_ENV)
        else None
    )
    if sample_directory is None:
        parser.error(
            f"--samples is required, or set {SAMPLE_DIRECTORY_ENV} to an external WAV sample directory"
        )
    if not sample_directory.is_dir():
        parser.error(f"sample directory does not exist: {sample_directory}")
    render_midi(args.midi, sample_directory, args.output, duration_limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
