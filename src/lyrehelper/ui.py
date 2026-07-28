from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QActionGroup, QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSystemTrayIcon,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .audio_capture import list_output_devices
from .analysis import TEMPO_JUMP_MIN_BPM, TEMPO_JUMP_RATIO, _tempo_jump_segments
from .config import AppConfig
from .history import HistoryEntry, list_history, load_history_snapshot
from .models import AnalysisSnapshot, MonitorState, TempoPoint
from .pipeline import AnalysisPipeline

INK = QColor("#e7ebe9")
MUTED = QColor("#7e8b88")
GRID = QColor("#26312f")
PANEL = QColor("#111816")
GREEN = QColor("#76e6a5")
AMBER = QColor("#f1b95b")
RED = QColor("#ff6b62")
CYAN = QColor("#56c8d8")


def _clock(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _midi_note_name(midi_note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def _smoothed_tempo_values(
    points: list[TempoPoint], window_seconds: float = 30.0
) -> list[tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda point: point.time)
    times = np.asarray([point.time for point in ordered], dtype=float)
    values = np.asarray([point.bpm for point in ordered], dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.searchsorted(times, times - window_seconds, side="left")
    counts = np.arange(1, len(values) + 1) - starts
    averages = (prefix[1:] - prefix[starts]) / np.maximum(counts, 1)
    return [(float(time), float(value)) for time, value in zip(times, averages)]


def _smoothed_tempo_segments(
    points: list[TempoPoint],
    window_seconds: float = 30.0,
    jump_bpm: float = TEMPO_JUMP_MIN_BPM,
    jump_ratio: float = TEMPO_JUMP_RATIO,
) -> list[list[tuple[float, float]]]:
    raw_segments = _tempo_jump_segments(points, jump_bpm, jump_ratio)
    return [
        _smoothed_tempo_values(segment, window_seconds)
        for segment in raw_segments
    ]


def _tinted_icon(icon: QIcon, color: QColor, size: int = 18) -> QIcon:
    source = icon.pixmap(size, size)
    tinted = QPixmap(source.size())
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, source)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    return QIcon(tinted)


def _prohibited_icon(color: QColor, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap))
    painter.drawEllipse(2, 2, size - 4, size - 4)
    painter.drawLine(3, size - 3, size - 3, 3)
    painter.end()
    return QIcon(pixmap)


class SignalPulse(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(34, 34)
        self._phase = 0.0
        self._strength = 0.0

    def update_pulse(self, snapshot: AnalysisSnapshot) -> None:
        if snapshot.beats and snapshot.state == MonitorState.ANALYZING:
            last = snapshot.beats[-1]
            age = max(0.0, snapshot.playhead - last.time)
            self._strength = math.exp(-age * 6.5)
        else:
            self._strength *= 0.82
        self._phase += 0.12
        self.update()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = 5 + self._strength * 8
        halo = QColor(GREEN)
        halo.setAlphaF(0.08 + self._strength * 0.32)
        painter.setBrush(halo)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(GREEN if self._strength > 0.15 else MUTED)
        painter.drawEllipse(center, 3.5, 3.5)


class TimelineWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(430)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._snapshot = AnalysisSnapshot()
        self._span = 30.0
        self._view_end = 30.0
        self._follow = True
        self._drag_origin: QPoint | None = None
        self._drag_end: QPoint | None = None
        self._drag_view_end = 0.0
        self._last_session_id: str | None = None
        self._last_state = MonitorState.CONNECTING
        self.setMouseTracking(True)
        self.setToolTip(
            "Mouse wheel scrolls time; Ctrl+wheel zooms; drag pans; Shift-drag box-zooms"
        )

    def set_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        session_started = (
            snapshot.session_id is not None
            and snapshot.session_id != self._last_session_id
        )
        analysis_started = (
            snapshot.state == MonitorState.ANALYZING
            and self._last_state != MonitorState.ANALYZING
        )
        if session_started or analysis_started:
            self._follow = True
        self._snapshot = snapshot
        if self._follow:
            self._view_end = max(self._span, snapshot.playhead + 1.0)
        self._last_session_id = snapshot.session_id
        self._last_state = snapshot.state
        self.update()

    def follow_playhead(self) -> None:
        self._follow = True
        self._view_end = max(self._span, self._snapshot.playhead + 1.0)
        self.update()

    @property
    def _view_start(self) -> float:
        return max(0.0, self._view_end - self._span)

    def _x(self, time: float) -> float:
        return 56 + (time - self._view_start) / self._span * max(1, self.width() - 72)

    def _time(self, x: float) -> float:
        return self._view_start + (x - 56) / max(1, self.width() - 72) * self._span

    def wheelEvent(self, event: object) -> None:
        steps = event.angleDelta().y() / 120.0
        if event.modifiers() & Qt.ControlModifier:
            anchor = self._time(event.position().x())
            factor = 0.82 if steps > 0 else 1.22
            new_span = float(np.clip(self._span * factor, 6.0, 600.0))
            ratio = (anchor - self._view_start) / self._span
            self._span = new_span
            self._view_end = anchor + new_span * (1 - ratio)
        else:
            self._view_end -= steps * self._span * 0.12
        maximum_end = max(self._span, self._snapshot.elapsed + 1.0)
        self._view_end = float(np.clip(self._view_end, self._span, maximum_end))
        self._follow = False
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.position().toPoint()
            self._drag_end = self._drag_origin
            self._drag_view_end = self._view_end

    def mouseMoveEvent(self, event: object) -> None:
        if self._drag_origin is None:
            return
        self._drag_end = event.position().toPoint()
        if not event.modifiers() & Qt.ShiftModifier:
            delta = self._drag_origin.x() - self._drag_end.x()
            self._view_end = max(self._span, self._drag_view_end + delta / max(1, self.width() - 72) * self._span)
            self._follow = False
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        if self._drag_origin is not None and event.modifiers() & Qt.ShiftModifier:
            left, right = sorted((self._drag_origin.x(), event.position().x()))
            if right - left > 18:
                start, end = self._time(left), self._time(right)
                self._span = max(3.0, end - start)
                self._view_end = max(self._span, end)
                self._follow = False
        self._drag_origin = None
        self._drag_end = None
        self.update()

    def mouseDoubleClickEvent(self, event: object) -> None:
        self._span = 30.0
        self._follow = True
        self.update()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0c1211"))
        plot = QRectF(56, 26, self.width() - 72, self.height() - 80)
        self._draw_grid(painter, plot)
        self._draw_spectrum(painter, plot)
        self._draw_notes(painter, plot)
        self._draw_tempo(painter, plot)
        self._draw_beats(painter, plot)
        self._draw_chords(painter, plot)
        playhead_x = self._x(self._snapshot.playhead)
        if plot.left() <= playhead_x <= plot.right():
            painter.setPen(QPen(INK, 1.2))
            painter.drawLine(QPointF(playhead_x, plot.top()), QPointF(playhead_x, plot.bottom() + 36))
            painter.setBrush(INK)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(
                [QPointF(playhead_x - 4, plot.top()), QPointF(playhead_x + 4, plot.top()), QPointF(playhead_x, plot.top() + 7)]
            )
        if self._drag_origin and self._drag_end and QApplication.keyboardModifiers() & Qt.ShiftModifier:
            selection = QRect(self._drag_origin, self._drag_end).normalized()
            painter.fillRect(selection, QColor(86, 200, 216, 35))
            painter.setPen(QPen(CYAN, 1))
            painter.drawRect(selection)

    def _draw_grid(self, painter: QPainter, plot: QRectF) -> None:
        painter.setFont(QFont("Cascadia Mono", 8))
        tick = 1 if self._span <= 15 else 5 if self._span <= 90 else 15 if self._span <= 240 else 30
        first = math.ceil(self._view_start / tick) * tick
        for time in np.arange(first, self._view_end + tick, tick):
            x = self._x(float(time))
            painter.setPen(QPen(GRID, 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom() + 36))
            painter.setPen(MUTED)
            painter.drawText(QRectF(x + 4, 4, 48, 18), _clock(float(time)))
        for ratio, label in ((0.0, "NOTES"), (0.42, "TEMPO"), (0.72, "BEAT")):
            y = plot.top() + plot.height() * ratio
            painter.setPen(QPen(GRID, 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(MUTED)
            painter.drawText(QRectF(4, y + 4, 48, 16), label)

    def _draw_spectrum(self, painter: QPainter, plot: QRectF) -> None:
        data = self._snapshot.spectrum
        start = self._snapshot.spectrum_start
        end = self._snapshot.spectrum_end
        visible_start = max(self._view_start, start)
        visible_end = min(self._view_end, end)
        if data.size == 0 or end <= start or visible_end <= visible_start:
            return
        rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = (18 + data * 55).astype(np.uint8)
        rgba[..., 1] = (35 + data * 160).astype(np.uint8)
        rgba[..., 2] = (34 + data * 120).astype(np.uint8)
        rgba[..., 3] = (data * 54).astype(np.uint8)
        image = QImage(rgba.data, data.shape[1], data.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
        mirrored = image.mirrored(False, True)
        source_left = (visible_start - start) / (end - start) * mirrored.width()
        source_right = (visible_end - start) / (end - start) * mirrored.width()
        source = QRectF(source_left, 0, source_right - source_left, mirrored.height())
        target = QRectF(
            self._x(visible_start),
            plot.top(),
            self._x(visible_end) - self._x(visible_start),
            plot.height() * 0.42,
        )
        painter.drawImage(target, mirrored, source)

    def _draw_notes(self, painter: QPainter, plot: QRectF) -> None:
        top = plot.top()
        bottom = plot.top() + plot.height() * 0.42
        visible = [
            note
            for note in self._snapshot.notes
            if note.end >= self._view_start and note.start <= self._view_end
        ]
        if not visible:
            painter.setPen(QColor("#46514f"))
            painter.setFont(QFont("Bahnschrift", 10))
            painter.drawText(
                QRectF(plot.left(), top, plot.width(), bottom - top),
                Qt.AlignCenter,
                "TRANSCRIBING PITCH EVENTS",
            )
            return
        lowest = max(24, min(note.midi_note for note in visible) - 3)
        highest = min(108, max(note.midi_note for note in visible) + 4)
        lowest = (lowest // 12) * 12
        highest = max(lowest + 24, ((highest + 11) // 12) * 12)
        highest = min(120, highest)

        def note_y(midi_note: float) -> float:
            return bottom - (midi_note - lowest) / max(1, highest - lowest) * (bottom - top)

        painter.setFont(QFont("Cascadia Mono", 7))
        for midi_note in range(lowest, highest + 1):
            y = note_y(midi_note)
            if midi_note % 12 == 0:
                painter.setPen(QPen(QColor("#33403d"), 1))
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                painter.setPen(MUTED)
                painter.drawText(QRectF(4, y - 8, 46, 14), _midi_note_name(midi_note))
            elif midi_note % 12 in {1, 3, 6, 8, 10}:
                painter.fillRect(
                    QRectF(
                        plot.left(),
                        note_y(midi_note + 0.5),
                        plot.width(),
                        max(1, note_y(midi_note - 0.5) - note_y(midi_note + 0.5)),
                    ),
                    QColor(5, 10, 9, 34),
                )
        lane_height = max(9.0, (bottom - top) / max(1, highest - lowest) * 0.72)
        for note in visible:
            left = max(plot.left(), self._x(note.start))
            right = min(plot.right(), self._x(note.end))
            if right <= left:
                continue
            active = note.start <= self._snapshot.playhead < note.end
            fill = (
                QColor(241, 185, 91, 225)
                if active
                else QColor(79, 202, 151, 90 + round(note.confidence * 100))
            )
            border = QColor("#ffd88c") if active else QColor("#76e6a5")
            y = note_y(note.midi_note + 0.5) - lane_height / 2
            rect = QRectF(left + 1, y, max(2.0, right - left - 2), lane_height)
            painter.setBrush(fill)
            painter.setPen(QPen(border, 1.0))
            painter.drawRoundedRect(rect, 2, 2)
            if rect.width() > 42:
                painter.setPen(QColor("#eaf5ef"))
                painter.setFont(QFont("Cascadia Mono", 6))
                label = note.name if rect.width() <= 88 else f"{note.name}  {note.frequency:.1f} Hz"
                painter.drawText(rect.adjusted(4, 0, -3, 0), Qt.AlignVCenter, label)

    def _draw_tempo(self, painter: QPainter, plot: QRectF) -> None:
        visible = [point for point in self._snapshot.tempo if self._view_start <= point.time <= self._view_end]
        if not visible:
            painter.setPen(QColor("#46514f"))
            painter.setFont(QFont("Bahnschrift", 10))
            painter.drawText(QRectF(plot.left(), plot.top() + plot.height() * 0.48, plot.width(), 24), Qt.AlignCenter, "LISTENING FOR A STABLE PULSE")
            return
        low = min(point.bpm for point in visible) - 4
        high = max(point.bpm for point in visible) + 4
        if high - low < 12:
            midpoint = (high + low) / 2
            low, high = midpoint - 6, midpoint + 6
        top = plot.top() + plot.height() * 0.46
        bottom = plot.top() + plot.height() * 0.68
        path = QPainterPath()
        for index, point in enumerate(visible):
            x = self._x(point.time)
            y = bottom - (point.bpm - low) / (high - low) * (bottom - top)
            path.moveTo(x, y) if index == 0 else path.lineTo(x, y)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(AMBER, 2.0))
        painter.drawPath(path)
        for smooth_segment in _smoothed_tempo_segments(self._snapshot.tempo):
            smooth_visible = [
                (time, bpm)
                for time, bpm in smooth_segment
                if self._view_start <= time <= self._view_end
            ]
            if not smooth_visible:
                continue
            smooth_path = QPainterPath()
            for index, (time, bpm) in enumerate(smooth_visible):
                x = self._x(time)
                y = bottom - (bpm - low) / (high - low) * (bottom - top)
                smooth_path.moveTo(x, y) if index == 0 else smooth_path.lineTo(x, y)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(CYAN, 1.8))
            painter.drawPath(smooth_path)
        painter.setPen(MUTED)
        painter.setFont(QFont("Cascadia Mono", 8))
        painter.drawText(QRectF(plot.left(), top - 18, 80, 16), f"{high:.0f} BPM")
        painter.drawText(QRectF(plot.left(), bottom + 2, 80, 16), f"{low:.0f} BPM")
        painter.setPen(AMBER)
        painter.drawText(QRectF(plot.right() - 112, top - 18, 48, 16), "RAW")
        painter.setPen(CYAN)
        painter.drawText(QRectF(plot.right() - 64, top - 18, 64, 16), "30S AVG")

    def _draw_beats(self, painter: QPainter, plot: QRectF) -> None:
        top = plot.top() + plot.height() * 0.72
        bottom = plot.bottom()
        for beat in self._snapshot.beats:
            x = self._x(beat.time)
            if not plot.left() <= x <= plot.right():
                continue
            color = RED if beat.is_downbeat else QColor(118, 230, 165, 115)
            width = 2.8 if beat.is_downbeat else 1.0
            painter.setPen(QPen(color, width))
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            if beat.is_downbeat:
                painter.fillRect(QRectF(x - 3, top, 6, 6), RED)

    def _draw_chords(self, painter: QPainter, plot: QRectF) -> None:
        y = plot.bottom() + 10
        current_time = self._snapshot.playhead
        for chord in self._snapshot.chords:
            left = max(plot.left(), self._x(chord.start))
            right = min(plot.right(), self._x(chord.end))
            if right <= plot.left() or left >= plot.right() or right - left < 2:
                continue
            active = chord.start <= current_time < chord.end
            color = QColor("#244c3c") if active else QColor("#182421")
            painter.fillRect(QRectF(left + 1, y, right - left - 2, 31), color)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(GREEN if active else GRID, 1))
            painter.drawRect(QRectF(left + 1, y, right - left - 2, 31))
            painter.setPen(INK if active else QColor("#a4afac"))
            painter.setFont(QFont("Bahnschrift", 9, QFont.DemiBold))
            painter.drawText(QRectF(left + 5, y, right - left - 10, 31), Qt.AlignVCenter, f"{chord.chord}  {chord.function}")


class Metric(QWidget):
    def __init__(self, label: str, value: str = "--") -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        caption = QLabel(label.upper())
        caption.setObjectName("metricCaption")
        layout.addWidget(self.value)
        layout.addWidget(caption)


class SummaryPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("summaryPanel")
        self.setFixedWidth(310)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)
        title = QLabel("SESSION SUMMARY")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        metrics = QGridLayout()
        metrics.setSpacing(16)
        self.average = Metric("Average BPM")
        self.range = Metric("Tempo range")
        self.deviation = Metric("BPM deviation")
        self.duration = Metric("Elapsed")
        metrics.addWidget(self.average, 0, 0)
        metrics.addWidget(self.range, 0, 1)
        metrics.addWidget(self.deviation, 1, 0)
        metrics.addWidget(self.duration, 1, 1)
        layout.addLayout(metrics)
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("divider")
        layout.addWidget(divider)
        label = QLabel("PERFORMANCE CHARACTER")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        self.score = QLabel("--")
        self.score.setObjectName("score")
        self.score_caption = QLabel("Awaiting tempo evidence")
        self.score_caption.setObjectName("muted")
        self.score_caption.setWordWrap(True)
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setTextVisible(False)
        self.score_bar.setFixedHeight(7)
        layout.addWidget(self.score)
        layout.addWidget(self.score_caption)
        layout.addWidget(self.score_bar)
        keys_label = QLabel("DETECTED KEYS")
        keys_label.setObjectName("sectionTitle")
        layout.addWidget(keys_label)
        self.keys = QLabel("No stable key yet")
        self.keys.setObjectName("keyList")
        self.keys.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.keys.setWordWrap(False)
        self.keys.setTextInteractionFlags(Qt.NoTextInteraction)
        self.keys_area = QScrollArea()
        self.keys_area.setObjectName("keyListArea")
        self.keys_area.setWidgetResizable(True)
        self.keys_area.setFrameShape(QFrame.NoFrame)
        self.keys_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.keys_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.keys_area.setWidget(self.keys)
        self.keys_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.keys_area, 1)
        self.archive = QLabel("No archive created in this run")
        self.archive.setObjectName("archive")
        self.archive.setWordWrap(True)
        layout.addWidget(self.archive)

    def set_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        self.average.value.setText(f"{snapshot.average_bpm:.1f}" if snapshot.average_bpm else "--")
        self.range.value.setText(
            f"{snapshot.min_bpm:.0f}–{snapshot.max_bpm:.0f}" if snapshot.min_bpm else "--"
        )
        self.deviation.value.setText(f"σ {snapshot.bpm_std:.2f}" if snapshot.average_bpm else "--")
        self.duration.value.setText(_clock(snapshot.elapsed))
        self.score.setText(f"{snapshot.human_score:.0f}%")
        if snapshot.human_score >= 70:
            character = "Humanized timing"
        elif snapshot.human_score <= 45:
            character = "Mechanically stable"
        else:
            character = "Mixed timing evidence"
        grid = (
            f"\nGrid {snapshot.grid_accuracy:.0f}% · {snapshot.timing_deviation_ms:.1f} ms"
            if snapshot.grid_accuracy
            else ""
        )
        self.score_caption.setText(f"{character}{grid}")
        self.score_bar.setValue(round(snapshot.human_score))
        if snapshot.keys:
            self.keys.setText(
                "\n".join(
                    f"{item.key}  ·  {_clock(item.start)}–{_clock(item.end)}"
                    for item in snapshot.keys
                )
            )
        else:
            self.keys.setText("No stable key yet")
        if snapshot.last_archive:
            self.archive.setText(f"LAST AUTO-ARCHIVE\n{snapshot.last_archive}")


class HarmonyFlow(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("harmonyFlow")
        self.setFixedHeight(112)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(4)
        heading = QLabel("HARMONIC MOTION")
        heading.setObjectName("sectionTitle")
        self.heading = heading
        self.flow = QLabel("Waiting for a harmonic phrase")
        self.flow.setObjectName("flowText")
        self.roles = QLabel("")
        self.roles.setObjectName("flowDetail")
        layout.addWidget(heading)
        layout.addWidget(self.flow)
        layout.addWidget(self.roles)

    def set_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        current_time = snapshot.playhead if snapshot.playhead > 0 else snapshot.elapsed
        current_key = next(
            (
                item.key
                for item in reversed(snapshot.keys)
                if item.start <= current_time <= item.end + 0.01
            ),
            snapshot.keys[-1].key if snapshot.keys else "Unknown",
        )
        recent = [
            item
            for item in snapshot.chords
            if item.chord != "N"
            and item.end >= max(0.0, current_time - 45.0)
            and item.start <= current_time + 0.05
        ]
        compressed = []
        for item in recent:
            if compressed and item.chord == compressed[-1].chord and item.key == compressed[-1].key:
                continue
            compressed.append(item)
        compressed = compressed[-8:]
        self.heading.setText(f"FUNCTIONAL HARMONY · {current_key.upper()}")
        if not compressed:
            self.flow.setText("Melody only · no stable chord stack")
            self.roles.setText("Key tracking remains active without inventing chord roots")
            return
        labels = [
            f"{item.function} · {item.chord}" if item.function else item.chord
            for item in compressed
        ]
        self.flow.setText("  →  ".join(labels))
        roles = [_harmonic_role(item.function) for item in compressed]
        role_chain = [role for index, role in enumerate(roles) if not index or role != roles[index - 1]]
        cadence = _cadence_label([item.function for item in compressed])
        detail = "  →  ".join(role_chain)
        self.roles.setText(f"{detail}  ·  {cadence}" if cadence else detail)


def _harmonic_role(function: str) -> str:
    normalized = function.replace("7", "")
    if normalized in {"I", "i", "III", "iii", "VI", "vi"}:
        return "TONIC"
    if normalized in {"II", "ii", "ii°", "IV", "iv"}:
        return "PREDOMINANT"
    if normalized in {"V", "v", "VII", "vii°"}:
        return "DOMINANT"
    return "CHROMATIC"


def _cadence_label(functions: list[str]) -> str:
    normalized = [item.replace("7", "") for item in functions if item]
    if len(normalized) < 2:
        return ""
    pair = normalized[-2:]
    if pair[0] == "V" and pair[1] in {"I", "i"}:
        return "AUTHENTIC CADENCE"
    if pair[0] in {"IV", "iv"} and pair[1] in {"I", "i"}:
        return "PLAGAL CADENCE"
    if pair[0] in {"II", "ii", "ii°", "IV", "iv"} and pair[1] == "V":
        return "PREDOMINANT TO DOMINANT"
    return ""


def _current_bpm(snapshot: AnalysisSnapshot) -> float:
    visible = [point.bpm for point in snapshot.tempo if point.time <= snapshot.playhead + 0.05]
    return visible[-1] if visible else snapshot.average_bpm


def _live_state_text(snapshot: AnalysisSnapshot) -> str:
    if snapshot.cooldown_remaining > 0:
        return f"PAUSED · AUTO IN {snapshot.cooldown_remaining:.1f}S"
    if snapshot.recording_mode == "off":
        return "RECORDING OFF"
    if snapshot.auto_candidate:
        return "WAITING FOR AUDIO INPUT"
    return {
        MonitorState.CONNECTING: "RECONNECTING",
        MonitorState.STANDBY: "WAITING FOR AUDIO INPUT"
        if snapshot.recording_mode == "auto"
        else "RECORDING ON · WAITING",
        MonitorState.ANALYZING: "LIVE ANALYSIS",
        MonitorState.FINALIZING: "AUTO-ARCHIVING",
        MonitorState.DEGRADED: "DEGRADED · CONTINUING",
    }[snapshot.state]


def _latency_text(snapshot: AnalysisSnapshot, *, compact: bool = False) -> str:
    prefix = "LAG" if compact else "ANALYSIS LAG"
    if snapshot.state not in {MonitorState.ANALYZING, MonitorState.DEGRADED}:
        return f"{prefix} --"
    return f"{prefix} {snapshot.analysis_latency:.1f}S"


def _set_latency_label(
    label: QLabel, snapshot: AnalysisSnapshot, *, compact: bool = False
) -> None:
    label.setText(_latency_text(snapshot, compact=compact))
    alert = (
        snapshot.state in {MonitorState.ANALYZING, MonitorState.DEGRADED}
        and snapshot.analysis_latency > 5.0
    )
    if label.property("alert") != alert:
        label.setProperty("alert", alert)
        label.style().unpolish(label)
        label.style().polish(label)


class MiniNoteCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._snapshot = AnalysisSnapshot()
        self.setMinimumHeight(132)

    def set_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0c1211"))
        right_time = max(12.0, self._snapshot.playhead or self._snapshot.elapsed)
        left_time = max(0.0, right_time - 12.0)
        visible = [
            note
            for note in self._snapshot.notes
            if note.end >= left_time and note.start <= right_time
        ]
        for second in range(math.ceil(left_time), math.floor(right_time) + 1):
            x = (second - left_time) / 12.0 * self.width()
            painter.setPen(QPen(GRID, 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        for beat in self._snapshot.beats:
            if not left_time <= beat.time <= right_time:
                continue
            x = (beat.time - left_time) / 12.0 * self.width()
            color = QColor(255, 107, 98, 205) if beat.is_downbeat else QColor(118, 230, 165, 100)
            painter.setPen(QPen(color, 2.2 if beat.is_downbeat else 1.0))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            if beat.is_downbeat:
                painter.fillRect(QRectF(x - 2.5, 0, 5, 5), RED)
        if not visible:
            painter.setPen(MUTED)
            painter.setFont(QFont("Cascadia Mono", 8))
            painter.drawText(self.rect(), Qt.AlignCenter, "NO PITCH EVENTS IN THE LAST 12 SECONDS")
            return
        low = min(note.midi_note for note in visible) - 2
        high = max(note.midi_note for note in visible) + 2
        high = max(high, low + 12)
        lane = max(5.0, self.height() / max(12, high - low + 1) * 0.7)
        for note in visible:
            left = (max(left_time, note.start) - left_time) / 12.0 * self.width()
            right = (min(right_time, note.end) - left_time) / 12.0 * self.width()
            y = self.height() - (note.midi_note - low + 0.5) / (high - low + 1) * self.height()
            rect = QRectF(left, y - lane / 2, max(2.0, right - left), lane)
            painter.setBrush(QColor(79, 202, 151, 145))
            painter.setPen(QPen(GREEN, 1))
            painter.drawRoundedRect(rect, 2, 2)


class FloatingMonitor(QWidget):
    def __init__(
        self,
        on_pause: Callable[[], None],
        on_start: Callable[[], None],
        on_terminate: Callable[[], None],
        on_label_changed: Callable[[str], None],
    ) -> None:
        super().__init__(None, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setObjectName("floatingRoot")
        self.setWindowTitle("LyreHelper · Live BPM")
        self.resize(540, 278)
        self.setMinimumSize(520, 240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        status_bar = QHBoxLayout()
        status_bar.setSpacing(7)
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status = QLabel("CONNECTING")
        self.status.setObjectName("status")
        self.score = QLabel("HUMANIZED --")
        self.score.setObjectName("floatingScore")
        self.score.setToolTip("Humanized timing score")
        self.score.setAccessibleName("Humanized timing score")
        self.latency = QLabel("ANALYSIS LAG --")
        self.latency.setObjectName("latency")
        self.label_group = QButtonGroup(self)
        self.label_group.setExclusive(True)
        self.label_buttons: dict[str, QToolButton] = {}
        for tag, text, tooltip in (
            ("human", "H", "Mark this session as Human"),
            ("non_human", "N", "Mark this session as Non-human"),
            ("none", "", "Do not label this session"),
        ):
            button = QToolButton()
            button.setObjectName("tagButton")
            button.setProperty("tag", tag)
            button.setCheckable(True)
            button.setChecked(tag == "none")
            button.setFixedSize(24, 22)
            if text:
                button.setText(text)
            else:
                button.setIcon(_prohibited_icon(QColor("#f2f5f3")))
                button.setIconSize(QSize(14, 14))
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.clicked.connect(
                lambda checked, selected=tag: on_label_changed(selected) if checked else None
            )
            self.label_group.addButton(button)
            self.label_buttons[tag] = button
        status_bar.addWidget(self.status_dot)
        status_bar.addWidget(self.status)
        status_bar.addSpacing(3)
        for tag in ("human", "non_human", "none"):
            status_bar.addWidget(self.label_buttons[tag])
        status_bar.addStretch()
        status_bar.addWidget(self.score)
        status_bar.addWidget(self.latency)
        layout.addLayout(status_bar)
        metrics = QHBoxLayout()
        self.current = Metric("Current BPM")
        self.range = Metric("BPM range")
        self.variance = Metric("Variance")
        metrics.addWidget(self.current)
        metrics.addWidget(self.range)
        metrics.addWidget(self.variance)
        self.start_button = QToolButton()
        self.start_button.setObjectName("commandButton")
        self.start_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                QColor("#f2f5f3"),
            )
        )
        self.start_button.setToolTip("Start recording now")
        self.start_button.setAccessibleName("Start recording")
        self.start_button.clicked.connect(on_start)
        metrics.addWidget(self.start_button)
        self.pause_button = QToolButton()
        self.pause_button.setObjectName("commandButton")
        self.pause_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause),
                QColor("#f2f5f3"),
            )
        )
        self.pause_button.setToolTip("Pause, archive, wait three seconds, then resume in AUTO")
        self.pause_button.setAccessibleName("Pause and archive")
        self.pause_button.clicked.connect(on_pause)
        metrics.addWidget(self.pause_button)
        self.terminate_button = QToolButton()
        self.terminate_button.setObjectName("commandButton")
        self.terminate_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
                QColor("#f2f5f3"),
            )
        )
        self.terminate_button.setToolTip("Terminate recording and switch to OFF")
        self.terminate_button.setAccessibleName("Terminate recording")
        self.terminate_button.clicked.connect(on_terminate)
        metrics.addWidget(self.terminate_button)
        layout.addLayout(metrics)
        self.notes = MiniNoteCanvas()
        layout.addWidget(self.notes, 1)
        self.setStyleSheet(STYLESHEET)

    def set_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        self.status.setText(_live_state_text(snapshot))
        has_score = bool(snapshot.tempo and snapshot.notes)
        self.score.setText(
            f"HUMANIZED {snapshot.human_score:.0f}%" if has_score else "HUMANIZED --"
        )
        _set_latency_label(self.latency, snapshot)
        self.status_dot.setProperty("active", snapshot.state == MonitorState.ANALYZING)
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)
        current = _current_bpm(snapshot)
        self.current.value.setText(f"{current:.1f}" if current else "--")
        self.range.value.setText(
            f"{snapshot.min_bpm:.0f}–{snapshot.max_bpm:.0f}" if snapshot.min_bpm else "--"
        )
        self.variance.value.setText(
            f"{snapshot.bpm_std ** 2:.2f}" if snapshot.average_bpm else "--"
        )
        self.notes.set_snapshot(snapshot)
        selected_label = (
            snapshot.session_label if snapshot.session_label in self.label_buttons else "none"
        )
        self.label_buttons[selected_label].setChecked(True)
        paused = snapshot.cooldown_remaining > 0
        self.start_button.setEnabled(snapshot.recording_mode != "on" or paused)
        self.pause_button.setEnabled(not paused)
        self.terminate_button.setEnabled(snapshot.recording_mode != "off" or paused)

    def closeEvent(self, event: object) -> None:
        event.ignore()
        self.hide()


class HistoryDialog(QDialog):
    def __init__(self, output_directory: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analysis history")
        self.setMinimumSize(620, 430)
        self.entries = list_history(output_directory)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        heading = QLabel("ANALYSIS HISTORY")
        heading.setObjectName("dialogTitle")
        detail = QLabel("Archived audio, transcription MIDI, tempo map, chords and keys")
        detail.setObjectName("dialogDetail")
        layout.addWidget(heading)
        layout.addWidget(detail)
        self.list = QListWidget()
        self.list.setObjectName("historyList")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for index, entry in enumerate(self.entries):
            audio_status = "WAV + MIDI" if entry.audio_path else "MIDI ONLY"
            item = QListWidgetItem(f"{entry.session_id}    {audio_status}")
            item.setData(Qt.UserRole, index)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.list.addItem("No archived sessions")
            self.list.setEnabled(False)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self.list, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Open)
        buttons.button(QDialogButtonBox.Open).setEnabled(bool(self.entries))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_entry(self) -> HistoryEntry | None:
        item = self.list.currentItem()
        if item is None or not self.entries:
            return None
        index = item.data(Qt.UserRole)
        return self.entries[int(index)] if index is not None else None


class AudioSourceDialog(QDialog):
    def __init__(self, selected_device: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio input source")
        self.setModal(True)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(16)
        heading = QLabel("AUDIO INPUT SOURCE")
        heading.setObjectName("dialogTitle")
        detail = QLabel("Capture the selected Windows output through WASAPI loopback.")
        detail.setObjectName("dialogDetail")
        detail.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(detail)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setVerticalSpacing(10)
        self.devices = QComboBox()
        self.devices.setObjectName("deviceSelector")
        self.devices.addItem("Follow system default", "")
        available = list_output_devices()
        for name in available:
            self.devices.addItem(name, name)
        if selected_device and selected_device not in available:
            self.devices.addItem(f"{selected_device} (currently unavailable)", selected_device)
        selected_index = self.devices.findData(selected_device)
        self.devices.setCurrentIndex(max(0, selected_index))
        form.addRow("OUTPUT DEVICE", self.devices)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_device(self) -> str:
        return str(self.devices.currentData() or "")


class MainWindow(QMainWindow):
    def __init__(
        self,
        pipeline: AnalysisPipeline,
        config: AppConfig,
        on_audio_source_changed: Callable[[str], None] | None = None,
        on_recording_mode_changed: Callable[[str], None] | None = None,
        analysis_device: str = "CPU",
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.on_audio_source_changed = on_audio_source_changed
        self.on_recording_mode_changed = on_recording_mode_changed
        self.analysis_device = analysis_device
        self._allow_quit = False
        self._history_snapshot: AnalysisSnapshot | None = None
        self._history_entry: HistoryEntry | None = None
        self._persisted_recording_mode = config.recording_mode
        self._history_audio = QAudioOutput(self)
        self._history_audio.setVolume(0.8)
        self._history_player = QMediaPlayer(self)
        self._history_player.setAudioOutput(self._history_audio)
        self._history_player.playbackStateChanged.connect(self._update_history_play_icon)
        self.setWindowTitle("LyreHelper — Live Harmonic Monitor")
        self.resize(1360, 820)
        self.setMinimumSize(980, 650)
        self._build_ui()
        self.floating = FloatingMonitor(
            self._pause_recording,
            lambda: self._set_recording_mode("on"),
            lambda: self._set_recording_mode("off"),
            self.pipeline.set_session_label,
        )
        self._build_tray()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(80)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        shell = QVBoxLayout(root)
        shell.setContentsMargins(22, 18, 22, 20)
        shell.setSpacing(14)
        header = QHBoxLayout()
        brand = QLabel("LYRE / HELPER")
        brand.setObjectName("brand")
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status = QLabel("CONNECTING")
        self.status.setObjectName("status")
        self.device = QLabel("Searching for output device")
        self.device.setObjectName("device")
        self.latency = QLabel("LAG --")
        self.latency.setObjectName("latency")
        self.compute = QLabel(self.analysis_device)
        self.compute.setObjectName("computeDevice")
        self.compute.setProperty(
            "accelerated", not self.analysis_device.startswith("CPU")
        )
        self.pulse = SignalPulse()
        header.addWidget(brand)
        header.addSpacing(24)
        header.addWidget(self.status_dot)
        header.addWidget(self.status)
        header.addWidget(self.device)
        header.addWidget(self.latency)
        header.addWidget(self.compute)
        header.addStretch()
        self.record_group = QButtonGroup(self)
        self.record_group.setExclusive(True)
        self.record_buttons: dict[str, QToolButton] = {}
        for recording_mode, label in (("on", "ON"), ("off", "OFF"), ("auto", "AUTO")):
            button = QToolButton()
            button.setObjectName("segmentButton")
            button.setText(label)
            button.setCheckable(True)
            button.setChecked(self.config.recording_mode == recording_mode)
            button.clicked.connect(
                lambda checked, selected=recording_mode: self._set_recording_mode(selected)
                if checked
                else None
            )
            self.record_group.addButton(button)
            self.record_buttons[recording_mode] = button
            header.addWidget(button)
        self.pause_button = QToolButton()
        self.pause_button.setObjectName("commandButton")
        self.pause_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause),
                QColor("#f2f5f3"),
            )
        )
        self.pause_button.setToolTip("Pause, archive, wait three seconds, then resume in AUTO")
        self.pause_button.setAccessibleName("Pause and archive")
        self.pause_button.clicked.connect(self._pause_recording)
        header.addWidget(self.pause_button)
        self.float_button = QToolButton()
        self.float_button.setObjectName("commandButton")
        self.float_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton),
                QColor("#f2f5f3"),
            )
        )
        self.float_button.setToolTip("Show floating BPM monitor")
        self.float_button.clicked.connect(self._toggle_floating)
        header.addWidget(self.float_button)
        self.history_button = QToolButton()
        self.history_button.setObjectName("commandButton")
        self.history_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                QColor("#f2f5f3"),
            )
        )
        self.history_button.setToolTip("Open analysis history")
        self.history_button.clicked.connect(self._open_history)
        header.addWidget(self.history_button)
        self.history_play_button = QToolButton()
        self.history_play_button.setObjectName("commandButton")
        self.history_play_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                QColor("#f2f5f3"),
            )
        )
        self.history_play_button.setToolTip("Play or pause archived audio")
        self.history_play_button.clicked.connect(self._toggle_history_playback)
        self.history_play_button.hide()
        header.addWidget(self.history_play_button)
        self.live_button = QPushButton("LIVE")
        self.live_button.setObjectName("liveButton")
        self.live_button.clicked.connect(self._return_live)
        self.live_button.hide()
        header.addWidget(self.live_button)
        self.mode = QLabel("AUTO · FULL QUALITY")
        self.mode.setObjectName("mode")
        header.addWidget(self.mode)
        self.audio_source_button = QToolButton()
        self.audio_source_button.setObjectName("audioSourceButton")
        self.audio_source_button.setIcon(
            _tinted_icon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume),
                QColor("#f2f5f3"),
            )
        )
        self.audio_source_button.setIconSize(QSize(18, 18))
        self.audio_source_button.setToolTip("Select audio input source")
        self.audio_source_button.setAccessibleName("Select audio input source")
        self.audio_source_button.clicked.connect(self._choose_audio_source)
        header.addWidget(self.audio_source_button)
        header.addWidget(self.pulse)
        shell.addLayout(header)
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        self.timeline = TimelineWidget()
        self.harmony = HarmonyFlow()
        left.addWidget(self.timeline, 1)
        left.addWidget(self.harmony)
        body.addLayout(left, 1)
        self.summary = SummaryPanel()
        body.addWidget(self.summary)
        shell.addLayout(body, 1)
        self.setCentralWidget(root)
        self.setStyleSheet(STYLESHEET)

    def _build_tray(self) -> None:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#76e6a5"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(3, 3, 26, 26)
        painter.setPen(QPen(QColor("#0b1210"), 3))
        painter.drawLine(10, 21, 15, 10)
        painter.drawLine(15, 10, 21, 22)
        painter.end()
        self.setWindowIcon(QIcon(pixmap))
        self.tray = QSystemTrayIcon(QIcon(pixmap), self)
        self.tray.setToolTip("LyreHelper · automatic monitoring")
        menu = QMenu()
        open_action = menu.addAction("Open monitor")
        open_action.triggered.connect(self._show_from_tray)
        floating_action = menu.addAction("Floating monitor")
        floating_action.triggered.connect(self._toggle_floating)
        history_action = menu.addAction("Analysis history...")
        history_action.triggered.connect(self._open_history)
        cut_action = menu.addAction("Pause and archive")
        cut_action.triggered.connect(self._pause_recording)
        recording_menu = menu.addMenu("Recording mode")
        recording_actions = QActionGroup(self)
        recording_actions.setExclusive(True)
        self.recording_actions = {}
        for recording_mode, label in (("on", "On"), ("off", "Off"), ("auto", "Auto")):
            action = recording_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.config.recording_mode == recording_mode)
            action.triggered.connect(
                lambda checked, selected=recording_mode: self._set_recording_mode(selected)
                if checked
                else None
            )
            recording_actions.addAction(action)
            self.recording_actions[recording_mode] = action
        source_action = menu.addAction("Audio input source...")
        source_action.triggered.connect(self._choose_audio_source)
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(self.quit_application)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._show_from_tray() if reason == QSystemTrayIcon.DoubleClick else None
        )
        self.tray.show()

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()

    def _choose_audio_source(self) -> None:
        dialog = AudioSourceDialog(self.config.device_name, self)
        dialog.setStyleSheet(STYLESHEET)
        if dialog.exec() != QDialog.Accepted:
            return
        selected = dialog.selected_device
        if selected == self.config.device_name:
            return
        if self.on_audio_source_changed is not None:
            self.on_audio_source_changed(selected)

    def _set_recording_mode(self, mode: str) -> None:
        self.pipeline.set_recording_mode(mode)
        self.config.recording_mode = mode
        self._sync_recording_controls(mode)
        if self.on_recording_mode_changed is not None:
            self.on_recording_mode_changed(mode)
        self._persisted_recording_mode = mode

    def _sync_recording_controls(self, mode: str) -> None:
        for name, button in self.record_buttons.items():
            button.setChecked(name == mode)
        for name, action in self.recording_actions.items():
            action.setChecked(name == mode)

    def _pause_recording(self) -> None:
        self.pipeline.pause_recording()

    def _toggle_floating(self) -> None:
        if self.floating.isVisible():
            self.floating.hide()
        else:
            self.floating.show()
            self.floating.raise_()
            self.floating.activateWindow()

    def _open_history(self) -> None:
        dialog = HistoryDialog(self.config.output_path, self)
        dialog.setStyleSheet(STYLESHEET)
        if dialog.exec() != QDialog.Accepted:
            return
        entry = dialog.selected_entry
        if entry is None:
            return
        try:
            snapshot = load_history_snapshot(entry)
        except (OSError, ValueError, TypeError):
            return
        self._history_entry = entry
        self._history_snapshot = snapshot
        self.history_play_button.setVisible(entry.audio_path is not None)
        self.live_button.show()
        if entry.audio_path is not None:
            self._history_player.setSource(QUrl.fromLocalFile(str(entry.audio_path)))
        else:
            self._history_player.setSource(QUrl())

    def _return_live(self) -> None:
        self._history_player.stop()
        self._history_snapshot = None
        self._history_entry = None
        self.history_play_button.hide()
        self.live_button.hide()
        self.timeline.follow_playhead()

    def _toggle_history_playback(self) -> None:
        if self._history_entry is None or self._history_entry.audio_path is None:
            return
        if self._history_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._history_player.pause()
        else:
            self._history_player.play()

    def _update_history_play_icon(self, state: QMediaPlayer.PlaybackState) -> None:
        icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if state == QMediaPlayer.PlaybackState.PlayingState
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self.history_play_button.setIcon(
            _tinted_icon(self.style().standardIcon(icon), QColor("#f2f5f3"))
        )

    def _refresh(self) -> None:
        live_snapshot = self.pipeline.get_display_snapshot()
        self._sync_recording_controls(live_snapshot.recording_mode)
        if live_snapshot.recording_mode != self._persisted_recording_mode:
            self.config.recording_mode = live_snapshot.recording_mode
            if self.on_recording_mode_changed is not None:
                self.on_recording_mode_changed(live_snapshot.recording_mode)
            self._persisted_recording_mode = live_snapshot.recording_mode
        self.floating.set_snapshot(live_snapshot)
        if self._history_snapshot is not None and self._history_entry is not None:
            if self._history_entry.audio_path is not None:
                self._history_snapshot.playhead = self._history_player.position() / 1000.0
            snapshot = self._history_snapshot
            state_text = f"HISTORY · {self._history_entry.session_id} · LIVE RECORDING {live_snapshot.recording_mode.upper()}"
        else:
            snapshot = live_snapshot
            state_text = _live_state_text(snapshot)
        self.status.setText(state_text)
        _set_latency_label(self.latency, live_snapshot, compact=True)
        self.device.setText(live_snapshot.device_name)
        self.mode.setText(f"{snapshot.mode.upper()} · {snapshot.quality.upper()} QUALITY")
        self.status_dot.setProperty("active", live_snapshot.state == MonitorState.ANALYZING)
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)
        self.timeline.set_snapshot(snapshot)
        self.summary.set_snapshot(snapshot)
        self.harmony.set_snapshot(snapshot)
        self.pulse.update_pulse(snapshot)

    def closeEvent(self, event: object) -> None:
        if self.config.keep_running_in_tray and not self._allow_quit:
            event.ignore()
            self.hide()
            return
        event.accept()

    def quit_application(self) -> None:
        self._allow_quit = True
        self.pipeline.stop()
        QApplication.quit()


STYLESHEET = """
QWidget#root { background: #0b100f; color: #e7ebe9; font-family: 'Bahnschrift'; }
QWidget#floatingRoot { background: #0b100f; color: #e7ebe9; font-family: 'Bahnschrift'; }
QScrollArea#keyListArea { background: transparent; border: none; }
QScrollArea#keyListArea QWidget { background: transparent; }
QScrollBar:vertical { background: #111816; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #34413e; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QLabel#brand { color: #f2f5f3; font-family: 'Georgia'; font-size: 20px; font-weight: 700; }
QLabel#statusDot { color: #7e8b88; font-size: 12px; }
QLabel#statusDot[active="true"] { color: #76e6a5; }
QLabel#status { color: #d8dfdc; font-family: 'Cascadia Mono'; font-size: 10px; font-weight: 700; }
QLabel#device { color: #697572; font-size: 11px; }
QLabel#latency { color: #f1b95b; font-family: 'Cascadia Mono'; font-size: 9px; font-weight: 700; }
QLabel#latency[alert="true"] { color: #ff6b62; }
QLabel#computeDevice { color: #f1b95b; font-family: 'Cascadia Mono'; font-size: 9px; font-weight: 700; border: 1px solid #4a3c24; padding: 3px 6px; }
QLabel#computeDevice[accelerated="true"] { color: #76e6a5; border-color: #244c3c; }
QLabel#floatingScore { color: #76e6a5; font-family: 'Cascadia Mono'; font-size: 9px; font-weight: 700; padding-right: 8px; }
QToolButton#tagButton { background: transparent; color: #d8dfdc; border: 1px solid #34413e; padding: 1px; font-family: 'Cascadia Mono'; font-size: 9px; font-weight: 700; }
QToolButton#tagButton:hover { border-color: #8d9996; }
QToolButton#tagButton[tag="human"]:checked { background: #244c3c; color: #ffffff; border-color: #76e6a5; }
QToolButton#tagButton[tag="non_human"]:checked { background: #4a2626; color: #ffffff; border-color: #ff6b62; }
QToolButton#tagButton[tag="none"]:checked { background: #26312f; border-color: #8d9996; }
QLabel#mode { color: #8d9996; font-family: 'Cascadia Mono'; font-size: 9px; border: 1px solid #26312f; padding: 5px 8px; }
QFrame#summaryPanel, QFrame#harmonyFlow { background: #111816; border: 1px solid #26312f; }
QLabel#sectionTitle { color: #7e8b88; font-family: 'Cascadia Mono'; font-size: 9px; font-weight: 700; }
QLabel#metricValue { color: #f1f4f2; font-family: 'Georgia'; font-size: 23px; }
QLabel#metricCaption { color: #687471; font-family: 'Cascadia Mono'; font-size: 8px; }
QLabel#score { color: #76e6a5; font-family: 'Georgia'; font-size: 42px; }
QLabel#muted { color: #899592; font-size: 11px; }
QLabel#keyList { color: #cbd2cf; font-family: 'Cascadia Mono'; font-size: 10px; line-height: 1.6; }
QLabel#archive { color: #63706d; font-family: 'Cascadia Mono'; font-size: 8px; border-top: 1px solid #26312f; padding-top: 12px; }
QLabel#flowText { color: #dfe5e2; font-family: 'Georgia'; font-size: 17px; }
QLabel#flowDetail { color: #8d9996; font-family: 'Cascadia Mono'; font-size: 9px; }
QFrame#divider { color: #26312f; }
QProgressBar { background: #202a28; border: none; }
QProgressBar::chunk { background: #76e6a5; }
QMenu { background: #151d1b; color: #e7ebe9; border: 1px solid #33403d; padding: 6px; }
QMenu::item { padding: 7px 24px; }
QMenu::item:selected { background: #244c3c; }
QToolButton#audioSourceButton { background: transparent; border: 1px solid #26312f; padding: 6px; }
QToolButton#audioSourceButton:hover { background: #182421; border-color: #4a5a56; }
QToolButton#commandButton { background: transparent; border: 1px solid #26312f; padding: 6px; min-width: 20px; min-height: 20px; }
QToolButton#commandButton:hover { background: #182421; border-color: #76e6a5; }
QToolButton#segmentButton { background: #101715; color: #7e8b88; border: 1px solid #26312f; padding: 6px 9px; font-family: 'Cascadia Mono'; font-size: 8px; }
QToolButton#segmentButton:checked { background: #244c3c; color: #e7ebe9; border-color: #76e6a5; }
QPushButton#liveButton { background: #182421; color: #76e6a5; border: 1px solid #76e6a5; padding: 6px 10px; font-family: 'Cascadia Mono'; font-size: 8px; }
QDialog { background: #101715; color: #e7ebe9; font-family: 'Bahnschrift'; }
QLabel#dialogTitle { color: #f2f5f3; font-family: 'Georgia'; font-size: 19px; font-weight: 700; }
QLabel#dialogDetail { color: #7e8b88; font-size: 11px; }
QComboBox#deviceSelector { background: #0b100f; color: #e7ebe9; border: 1px solid #34413e; padding: 8px 10px; min-height: 20px; }
QComboBox#deviceSelector:hover, QComboBox#deviceSelector:focus { border-color: #76e6a5; }
QComboBox QAbstractItemView { background: #151d1b; color: #e7ebe9; border: 1px solid #34413e; selection-background-color: #244c3c; }
QListWidget#historyList { background: #0b100f; color: #dce3e0; border: 1px solid #34413e; font-family: 'Cascadia Mono'; font-size: 10px; padding: 4px; }
QListWidget#historyList::item { min-height: 34px; padding: 4px 8px; border-bottom: 1px solid #202a28; }
QListWidget#historyList::item:selected { background: #244c3c; color: #ffffff; }
QDialogButtonBox QPushButton { background: #182421; color: #e7ebe9; border: 1px solid #34413e; padding: 7px 16px; min-width: 72px; }
QDialogButtonBox QPushButton:hover { border-color: #76e6a5; }
"""
