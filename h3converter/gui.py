"""Native PySide6 desktop interface.

A native GUI, not a browser one, because the source is ~66 GB: the file picker
hands us a path and we read it in place. A browser file input would copy or
upload it.

Four screens, and exactly one decision for the user to make: A or B.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QtMsgType, QThread, QUrl, Qt, Signal, qInstallMessageHandler
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from h3converter import constants as C
from h3converter.h3_detect import Detection
from h3converter.hardware import system_info
from h3converter.logging_setup import active_log_path, get_logger
from h3converter.pipeline import (
    ConversionOutcome,
    ConversionRequest,
    SourceAnalysis,
    analyze_source,
    convert,
)
from h3converter.progress import ProgressEvent
from h3converter.quant import capability

log = get_logger("h3converter.gui")

SCREEN_SELECT, SCREEN_VALIDATE, SCREEN_CONVERT, SCREEN_DONE = range(4)


def human_bytes(size: float | int | None) -> str:
    if size is None:
        return "-"
    size = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.2f} {unit}"
        size /= 1000
    return f"{size:.2f} TB"


def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class CapabilityWorker(QThread):
    """Runs the dependency self-tests off the UI thread."""

    finished_probe = Signal(object)

    def run(self) -> None:  # noqa: D102
        try:
            self.finished_probe.emit(capability.probe())
        except Exception as exc:  # noqa: BLE001
            log.exception("Capability probe failed")
            report = capability.CapabilityReport()
            report.add("runtime.probe", False, f"{type(exc).__name__}: {exc}")
            self.finished_probe.emit(report)


class AnalysisWorker(QThread):
    """Header-only source inspection, off the UI thread."""

    finished_analysis = Signal(object)

    def __init__(self, source: Path):
        super().__init__()
        self._source = source

    def run(self) -> None:  # noqa: D102
        try:
            self.finished_analysis.emit(analyze_source(self._source))
        except Exception as exc:  # noqa: BLE001
            log.exception("Source analysis failed")
            analysis = SourceAnalysis(path=self._source, size_bytes=0, detection=Detection())
            analysis.errors.append(str(exc))
            self.finished_analysis.emit(analysis)


class ConversionWorker(QThread):
    """Runs one conversion, forwarding progress to the UI thread."""

    progressed = Signal(object)
    finished_conversion = Signal(object)

    def __init__(self, request: ConversionRequest):
        super().__init__()
        self._request = request
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        outcome = convert(
            self._request,
            progress_callback=self.progressed.emit,
            should_cancel=self._cancel.is_set,
        )
        self.finished_conversion.emit(outcome)


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def heading(text: str, size: int = 18) -> QLabel:
    label = QLabel(text)
    font = QFont()
    font.setPointSize(size)
    font.setBold(True)
    label.setFont(font)
    return label


def muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


class FactTable(QWidget):
    """A two-column label/value grid used on the validation and done screens."""

    def __init__(self) -> None:
        super().__init__()
        self._grid = QGridLayout(self)
        self._grid.setColumnStretch(1, 1)
        self._grid.setVerticalSpacing(6)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._rows: dict[str, QLabel] = {}

    def set(self, key: str, value: str, tone: str = "") -> None:
        if key not in self._rows:
            row = self._grid.rowCount()
            name = QLabel(key)
            name.setObjectName("factKey")
            name.setAlignment(Qt.AlignRight | Qt.AlignTop)
            value_label = QLabel()
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._grid.addWidget(name, row, 0)
            self._grid.addWidget(value_label, row, 1)
            self._rows[key] = value_label
        label = self._rows[key]
        label.setText(value)
        label.setObjectName({"ok": "factOk", "bad": "factBad", "warn": "factWarn"}.get(tone, ""))
        label.style().unpolish(label)
        label.style().polish(label)

    def clear(self) -> None:
        for label in self._rows.values():
            label.setText("")


class FormatButton(QPushButton):
    """One of the two conversion choices."""

    def __init__(self, key: str, title: str, subtitle: str):
        super().__init__()
        self.key = key
        self.setObjectName("formatButton")
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        self._title = QLabel(title)
        self._title.setObjectName("formatTitle")
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("formatSubtitle")
        self._subtitle.setWordWrap(True)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{C.APP_NAME} {C.APP_VERSION}")
        self.resize(940, 720)

        self._analysis: SourceAnalysis | None = None
        self._capability: capability.CapabilityReport | None = None
        self._system = system_info()
        self._conversion: ConversionWorker | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._outcome: ConversionOutcome | None = None

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._stack.addWidget(self._build_select())
        self._stack.addWidget(self._build_validate())
        self._stack.addWidget(self._build_convert())
        self._stack.addWidget(self._build_done())
        self.setStyleSheet(_STYLESHEET)

        self._capability_worker = CapabilityWorker()
        self._capability_worker.finished_probe.connect(self._on_capability)
        self._capability_worker.start()

    # -- screen 1: select --------------------------------------------------

    def _build_select(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(16)

        layout.addWidget(heading("Select a checkpoint to shrink", 22))
        layout.addWidget(muted(
            "Choose the full, unpruned BF16 .safetensors file. It is read in place and never "
            "modified; the converted checkpoint is written next to it."
        ))
        layout.addSpacing(16)

        row = QHBoxLayout()
        self._source_label = QLabel("No file selected")
        self._source_label.setObjectName("pathBox")
        self._source_label.setWordWrap(True)
        self._source_label.setMinimumHeight(56)
        browse = QPushButton("Browse...")
        browse.setObjectName("primary")
        browse.setMinimumHeight(56)
        browse.setMinimumWidth(150)
        browse.clicked.connect(self._on_browse)
        row.addWidget(self._source_label, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        self._select_status = muted("")
        layout.addWidget(self._select_status)
        layout.addStretch(1)
        layout.addWidget(muted(f"Log folder: {active_log_path().parent if active_log_path() else 'logs'}"))
        return page

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "", "Safetensors checkpoints (*.safetensors)"
        )
        if not path:
            return
        source = Path(path)
        self._source_label.setText(str(source))
        self._select_status.setText("Reading checkpoint header...")
        QApplication.processEvents()

        self._analysis_worker = AnalysisWorker(source)
        self._analysis_worker.finished_analysis.connect(self._on_analysis)
        self._analysis_worker.start()

    def _on_analysis(self, analysis: SourceAnalysis) -> None:
        self._analysis = analysis
        self._select_status.setText("")
        self._populate_validation()
        self._stack.setCurrentIndex(SCREEN_VALIDATE)

    # -- screen 2: validation and choice -----------------------------------

    def _build_validate(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 40, 48, 40)
        layout.setSpacing(14)

        layout.addWidget(heading("Checkpoint and system", 20))
        self._facts = FactTable()
        layout.addWidget(self._facts)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("rule")
        layout.addWidget(line)

        self._choice_heading = heading("Choose an output format", 16)
        layout.addWidget(self._choice_heading)

        self._button_a = FormatButton(C.FORMAT_W4A8, "A  -  " + C.FORMAT_LABELS[C.FORMAT_W4A8], "")
        self._button_b = FormatButton(C.FORMAT_NVFP4, "B  -  " + C.FORMAT_LABELS[C.FORMAT_NVFP4], "")
        self._button_krea = FormatButton(C.FORMAT_KREA2_FP8, C.FORMAT_LABELS[C.FORMAT_KREA2_FP8], "")
        for button in (self._button_a, self._button_b, self._button_krea):
            button.clicked.connect(lambda _=False, b=button: self._start_conversion(b.key))
            layout.addWidget(button)

        self._validate_status = muted("")
        layout.addWidget(self._validate_status)
        layout.addStretch(1)

        back = QPushButton("Choose a different file")
        back.clicked.connect(lambda: self._stack.setCurrentIndex(SCREEN_SELECT))
        row = QHBoxLayout()
        row.addWidget(back)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _on_capability(self, report: capability.CapabilityReport) -> None:
        self._capability = report
        failed = report.failures()
        log.info("Capability probe finished; %d failed checks", len(failed))
        if self._analysis is not None:
            self._populate_validation()

    def _populate_validation(self) -> None:
        analysis = self._analysis
        if analysis is None:
            return
        detection = analysis.detection
        facts = self._facts

        facts.set("Source file", analysis.path.name)
        facts.set("Source size", human_bytes(analysis.size_bytes))

        if detection is None or (not detection.is_h3 and not getattr(detection, "is_krea2", False)):
            reason = "; ".join((detection.errors if detection else []) + analysis.errors)
            facts.set("Detected model", "Unsupported / unknown checkpoint", "bad")
            facts.set("Compatibility", reason or "unrecognised checkpoint", "bad")
        elif detection.is_h3:
            geometry = detection.geometry
            facts.set("Detected model", "MiniMax H3", "ok")
            facts.set("Source precision", detection.float_dtype or "unknown")
            if geometry:
                facts.set("Blocks", f"{geometry.num_layers} transformer blocks "
                                    f"(+{geometry.token_refiner_num_layers} token refiner)")
                facts.set("Dimensions", f"hidden {geometry.hidden_size}, "
                                        f"{geometry.num_attention_heads}x{geometry.attention_head_dim} heads, "
                                        f"ffn {geometry.ffn_hidden_size}, time embed {geometry.time_embed_dim}")
            facts.set("Pruning state", "curve-pruned already" if detection.already_curve_pruned
                      else "full time embedder (convertible)")
            facts.set("Quantization state", "already quantized" if detection.already_quantized else "none")
            facts.set(
                "Compatibility",
                "Ready to convert" if analysis.ok else "; ".join(detection.errors + analysis.errors),
                "ok" if analysis.ok else "bad",
            )
        else:
            geometry = detection.geometry
            facts.set("Detected model", "Krea 2", "ok")
            facts.set("Source precision", detection.float_dtype or "unknown")
            if geometry:
                facts.set("Blocks", "28 transformer blocks (+12 text fusion layers)")
                facts.set("Dimensions", "features 6144, channels 64, 48 heads / 12 KV heads, head dim 128")
            facts.set("Quantization state", "already quantized" if detection.already_quantized else "none")
            facts.set("Compatibility", "Ready to convert" if analysis.ok else "; ".join(detection.errors + analysis.errors),
                      "ok" if analysis.ok else "bad")

        system = self._system
        gpu = system.gpu
        facts.set("GPU", f"{gpu.name} ({gpu.capability_str})" if gpu.available
                  else f"none detected - {gpu.detail}", "ok" if gpu.available else "warn")
        facts.set("VRAM", f"{human_bytes(gpu.total_vram_bytes)} total, "
                          f"{human_bytes(gpu.free_vram_bytes)} free" if gpu.available else "-")
        facts.set("System RAM", f"{human_bytes(system.total_ram_bytes)} total, "
                                f"{human_bytes(system.available_ram_bytes)} available")

        is_krea = bool(getattr(detection, "is_krea2", False))
        self._button_a.setVisible(not is_krea)
        self._button_b.setVisible(not is_krea)
        self._button_krea.setVisible(is_krea)
        for key, button in ((C.FORMAT_W4A8, self._button_a), (C.FORMAT_NVFP4, self._button_b),
                            (C.FORMAT_KREA2_FP8, self._button_krea)):
            self._configure_format_button(key, button)

        if self._capability is None:
            self._validate_status.setText("Running dependency self-tests...")
        else:
            self._validate_status.setText("")

    def _configure_format_button(self, output_format: str, button: FormatButton) -> None:
        analysis = self._analysis
        plan = analysis.plan_preview.get(output_format) if analysis else None
        disk = analysis.disk.get(output_format) if analysis else None

        reasons: list[str] = []
        if analysis is None or not analysis.ok:
            reasons.append("source is not convertible")
        if self._capability is None:
            reasons.append("self-tests still running")
        elif not self._capability.ok_for(output_format):
            reasons.append(self._capability.blocking_reason(output_format) or "self-test failed")
        if disk is not None and not disk.ok:
            reasons.append(disk.message)

        if reasons:
            button.setEnabled(False)
            button.set_subtitle("Unavailable: " + "; ".join(reasons))
            return

        button.setEnabled(True)
        estimate = human_bytes(plan.total_bytes) if plan else "-"
        ratio = (analysis.size_bytes / plan.total_bytes) if plan and plan.total_bytes else 0
        detail = f"Estimated output {estimate} ({ratio:.1f}x smaller), " \
                 f"{plan.quantized_layer_count if plan else 0} quantized layers"
        if disk is not None and disk.message:
            detail += f"  -  {disk.message}"
        button.set_subtitle(detail)

    # -- screen 3: conversion ---------------------------------------------

    def _build_convert(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 48, 48, 40)
        layout.setSpacing(14)

        self._convert_heading = heading("Converting", 20)
        layout.addWidget(self._convert_heading)
        self._phase_label = QLabel("Starting...")
        self._phase_label.setObjectName("phase")
        layout.addWidget(self._phase_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(True)
        self._progress.setMinimumHeight(28)
        layout.addWidget(self._progress)

        self._detail_label = muted("")
        layout.addWidget(self._detail_label)

        self._convert_facts = FactTable()
        layout.addWidget(self._convert_facts)
        layout.addStretch(1)

        row = QHBoxLayout()
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._on_cancel)
        row.addStretch(1)
        row.addWidget(self._cancel_button)
        layout.addLayout(row)
        return page

    def _start_conversion(self, output_format: str) -> None:
        analysis = self._analysis
        if analysis is None or not analysis.ok:
            return

        self._convert_heading.setText(f"Converting to {C.FORMAT_LABELS[output_format]}")
        self._progress.setValue(0)
        self._phase_label.setText("Starting...")
        self._detail_label.setText("")
        self._convert_facts.set("Source", analysis.path.name)
        self._convert_facts.set("Log file", str(active_log_path() or "-"))
        self._cancel_button.setEnabled(True)
        self._cancel_button.setText("Cancel")
        self._stack.setCurrentIndex(SCREEN_CONVERT)

        request = ConversionRequest(source=analysis.path, output_format=output_format)
        self._conversion = ConversionWorker(request)
        self._conversion.progressed.connect(self._on_progress)
        self._conversion.finished_conversion.connect(self._on_finished)
        self._conversion.start()

    def _on_progress(self, event: ProgressEvent) -> None:
        self._progress.setValue(int(event.overall * 1000))
        self._progress.setFormat(f"{event.overall * 100:.1f}%")
        self._phase_label.setText(event.phase_label)
        self._detail_label.setText(event.detail)
        self._convert_facts.set("Elapsed", human_duration(event.elapsed_seconds))
        self._convert_facts.set("Output written", human_bytes(event.bytes_written))
        if event.peak_ram_bytes:
            self._convert_facts.set("Peak RAM", human_bytes(event.peak_ram_bytes))
        if event.peak_vram_bytes:
            self._convert_facts.set("Peak VRAM", human_bytes(event.peak_vram_bytes))

    def _on_cancel(self) -> None:
        if self._conversion is None:
            return
        self._conversion.cancel()
        self._cancel_button.setEnabled(False)
        self._cancel_button.setText("Cancelling...")
        self._phase_label.setText("Finishing the current step, then stopping...")

    # -- screen 4: complete -----------------------------------------------

    def _build_done(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 48, 48, 40)
        layout.setSpacing(14)

        self._done_heading = heading("Conversion complete", 22)
        layout.addWidget(self._done_heading)
        self._done_message = muted("")
        layout.addWidget(self._done_message)
        self._done_facts = FactTable()
        layout.addWidget(self._done_facts)
        layout.addStretch(1)

        row = QHBoxLayout()
        self._open_folder = QPushButton("Open Folder")
        self._open_folder.clicked.connect(self._on_open_folder)
        again = QPushButton("Convert Another")
        again.setObjectName("primary")
        again.clicked.connect(self._on_convert_another)
        row.addWidget(self._open_folder)
        row.addStretch(1)
        row.addWidget(again)
        layout.addLayout(row)
        return page

    def _on_finished(self, outcome: ConversionOutcome) -> None:
        self._outcome = outcome
        self._conversion = None
        facts = self._done_facts
        facts.clear()

        analysis = self._analysis
        facts.set("Source file", analysis.path.name if analysis else "-")
        facts.set("Source size", human_bytes(analysis.size_bytes if analysis else None))

        if outcome.ok:
            report = outcome.report
            self._done_heading.setText("Conversion complete")
            self._done_message.setText(
                "The converted checkpoint has been validated and written next to the source. "
                "The source file was not modified."
            )
            facts.set("Output file", str(report.output.get("path")), "ok")
            facts.set("Output size", human_bytes(report.output.get("size_bytes")))
            facts.set("Compression", f"{report.output.get('compression_ratio', 0):.2f}x smaller")
            facts.set("Format", str(report.output.get("format_label")))
            facts.set("Quantized layers", str(report.quantization.get("quantized_layers")))
            error = report.quantization.get("layer_error") or {}
            if error.get("measured_layers"):
                facts.set("Weight error", f"{error['rel_l2_median']:.4f} median relative L2 "
                                          f"({error['measured_layers']} layers sampled)")
            worst = report.pruning.get("worst_projection") or {}
            if worst:
                facts.set("AdaLN error", f"{worst.get('rel_l2', 0):.2e} relative (worst projection)")
            facts.set("Validation", str(report.validation.get("summary")), "ok")
            facts.set("Elapsed", human_duration(report.resources.get("elapsed_seconds", 0)))
            facts.set("Report", str(outcome.report_path))
            self._open_folder.setEnabled(True)
        else:
            self._done_heading.setText("Conversion cancelled" if outcome.cancelled else "Conversion failed")
            self._done_message.setText(
                (outcome.error or "Unknown error")
                + "\n\nThe source checkpoint was not modified and no output file was created."
            )
            facts.set("Log file", str(active_log_path() or "-"), "warn")
            self._open_folder.setEnabled(False)

        self._stack.setCurrentIndex(SCREEN_DONE)

    def _on_open_folder(self) -> None:
        outcome = self._outcome
        if outcome is None or outcome.output_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(outcome.output_path.parent)))

    def _on_convert_another(self) -> None:
        self._outcome = None
        self._stack.setCurrentIndex(SCREEN_SELECT)

    # -- lifecycle ---------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._conversion is not None and self._conversion.isRunning():
            answer = QMessageBox.question(
                self,
                "Conversion in progress",
                "A conversion is still running. Stop it and quit?\n\n"
                "The source file will not be affected and no output will be created.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._conversion.cancel()
            self._conversion.wait(30000)

        # Let the background probes finish before their QThreads are collected.
        for worker in (self._capability_worker, self._analysis_worker):
            if worker is not None and worker.isRunning():
                worker.wait(5000)
        event.accept()


_STYLESHEET = """
QWidget { background: #14161a; color: #e6e8ec; font-size: 13px; }
QLabel#muted { color: #98a0ad; }
QLabel#factKey { color: #98a0ad; padding-right: 10px; }
QLabel#factOk { color: #6ee7a8; }
QLabel#factBad { color: #ff8080; }
QLabel#factWarn { color: #ffc46b; }
QLabel#phase { font-size: 15px; color: #cfd4dd; }
QLabel#pathBox {
    background: #1c1f26; border: 1px solid #2b303a; border-radius: 6px; padding: 12px;
    color: #cfd4dd;
}
QFrame#rule { color: #2b303a; }
QPushButton {
    background: #232833; border: 1px solid #333a47; border-radius: 6px;
    padding: 10px 18px; color: #e6e8ec;
}
QPushButton:hover { background: #2b3140; }
QPushButton:disabled { background: #1a1d23; color: #5d6470; border-color: #262b34; }
QPushButton#primary { background: #3563e9; border-color: #3563e9; font-weight: 600; }
QPushButton#primary:hover { background: #4a74ee; }
QPushButton#formatButton { text-align: left; padding: 0; }
QLabel#formatTitle { font-size: 15px; font-weight: 600; color: #e6e8ec; }
QLabel#formatSubtitle { color: #98a0ad; }
QPushButton#formatButton:disabled QLabel#formatTitle { color: #5d6470; }
QProgressBar {
    background: #1c1f26; border: 1px solid #2b303a; border-radius: 6px; text-align: center;
}
QProgressBar::chunk { background: #3563e9; border-radius: 5px; }
"""


_QT_LEVELS = {
    QtMsgType.QtDebugMsg: log.debug,
    QtMsgType.QtInfoMsg: log.info,
    QtMsgType.QtWarningMsg: log.warning,
    QtMsgType.QtCriticalMsg: log.error,
    QtMsgType.QtFatalMsg: log.critical,
}


def _qt_message_handler(mode, context, message) -> None:
    """Send Qt's own warnings and fatals to the run log."""
    _QT_LEVELS.get(mode, log.info)("Qt: %s", message)


def run_gui(argv: list[str] | None = None) -> int:
    qInstallMessageHandler(_qt_message_handler)
    app = QApplication(argv or [])
    app.setApplicationName(C.APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()
