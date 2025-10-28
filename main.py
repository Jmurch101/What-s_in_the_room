import sys
import os
from typing import Dict, List, Tuple

# Qt compatibility: prefer PySide6, fallback to PyQt5
try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
    QT_BINDING = "PySide6"
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    QT_BINDING = "PyQt5"

# Aliases to keep rest of code unchanged
Qt = QtCore.Qt
QThread = QtCore.QThread
try:
    from PySide6.QtCore import Signal as Signal  # type: ignore
except Exception:
    from PyQt5.QtCore import pyqtSignal as Signal  # type: ignore

QPixmap = QtGui.QPixmap
QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QWidget = QtWidgets.QWidget
QPushButton = QtWidgets.QPushButton
QLabel = QtWidgets.QLabel
QTextEdit = QtWidgets.QTextEdit
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QFileDialog = QtWidgets.QFileDialog
QHBoxLayout = QtWidgets.QHBoxLayout
QVBoxLayout = QtWidgets.QVBoxLayout
QScrollArea = QtWidgets.QScrollArea
QMessageBox = QtWidgets.QMessageBox
QProgressBar = QtWidgets.QProgressBar

# Qt5/Qt6 enum compatibility
try:
    ALIGN_CENTER = Qt.AlignCenter
except AttributeError:
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter  # Qt6

try:
    KEEP_ASPECT_RATIO = Qt.KeepAspectRatio
except AttributeError:
    KEEP_ASPECT_RATIO = Qt.AspectRatioMode.KeepAspectRatio  # Qt6

try:
    SMOOTH_TRANSFORMATION = Qt.SmoothTransformation
except AttributeError:
    SMOOTH_TRANSFORMATION = Qt.TransformationMode.SmoothTransformation  # Qt6


def _ensure_qt_platform_plugin():
    """On macOS venvs, help Qt find the 'cocoa' platform plugin."""
    try:
        plugins_dir = None
        platforms_dir = None
        if QT_BINDING == "PySide6":
            import PySide6
            qt_dir = os.path.join(os.path.dirname(PySide6.__file__), "Qt")
            plugins_dir = os.path.join(qt_dir, "plugins")
            platforms_dir = os.path.join(plugins_dir, "platforms")
        else:
            import PyQt5
            qt_dir = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5")
            plugins_dir = os.path.join(qt_dir, "plugins")
            platforms_dir = os.path.join(plugins_dir, "platforms")
        if os.path.isdir(platforms_dir):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", platforms_dir)
        if os.path.isdir(plugins_dir):
            os.environ.setdefault("QT_PLUGIN_PATH", plugins_dir)
        os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
    except Exception:
        # Best-effort only; if this fails Qt will use defaults
        pass


class ModelManager:
    """Lazy-load models and provide inference helpers."""

    _caption_pipeline = None
    _yolo_model = None
    _device = "cpu"

    @classmethod
    def _detect_device(cls) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @classmethod
    def get_caption_pipeline(cls):
        if cls._caption_pipeline is None:
            cls._device = cls._detect_device()
            from transformers import pipeline

            # Lightweight, CPU-friendly captioner
            cls._caption_pipeline = pipeline(
                "image-to-text",
                model="nlpconnect/vit-gpt2-image-captioning",
                device=-1 if cls._device == "cpu" else 0,
            )
        return cls._caption_pipeline

    @classmethod
    def get_yolo_model(cls):
        if cls._yolo_model is None:
            cls._device = cls._detect_device()
            from ultralytics import YOLO

            # Small model for faster startup; downloads on first run
            cls._yolo_model = YOLO("yolov8n.pt")
        return cls._yolo_model

    @classmethod
    def caption_image(cls, image_path: str) -> str:
        pipe = cls.get_caption_pipeline()
        try:
            outputs = pipe(image_path, max_new_tokens=32)
            if isinstance(outputs, list) and outputs:
                text = outputs[0].get("generated_text") or outputs[0].get("caption")
                return text.strip() if text else ""
            return ""
        except Exception as exc:
            return f"Captioning error: {exc}"

    @classmethod
    def detect_objects(cls, image_path: str, conf: float = 0.3) -> List[Tuple[str, float]]:
        model = cls.get_yolo_model()
        try:
            results = model.predict(image_path, conf=conf, verbose=False)
            items: List[Tuple[str, float]] = []
            for res in results:
                names = res.names  # id->name mapping
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue
                cls_ids = boxes.cls.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy().tolist()
                for cid, score in zip(cls_ids, confs):
                    label = names.get(int(cid), str(int(cid)))
                    items.append((label, float(score)))
            return items
        except Exception as exc:
            return [(f"Detection error: {exc}", 0.0)]


class AnalyzerWorker(QThread):
    finished = Signal(str, list)
    status = Signal(str)
    progress = Signal(int)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def run(self):
        self.status.emit("Loading models (first run may download weights)...")
        self.progress.emit(5)
        # Warm up (lazy load)
        _ = ModelManager.get_caption_pipeline()
        self.progress.emit(15)
        _ = ModelManager.get_yolo_model()

        self.status.emit("Generating caption...")
        self.progress.emit(35)
        caption = ModelManager.caption_image(self.image_path)

        self.status.emit("Detecting items in the image...")
        self.progress.emit(65)
        detections = ModelManager.detect_objects(self.image_path)

        self.progress.emit(100)
        self.status.emit("Done")
        self.finished.emit(caption, detections)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("What's In The Room - Image Captioning & Items")
        self.resize(1200, 800)

        self.current_image_path: str = ""
        self.worker: AnalyzerWorker = None
        self.original_pixmap: QPixmap = None

        # Controls
        self.open_button = QPushButton("Open Image…")
        self.analyze_button = QPushButton("Analyze Image")
        self.analyze_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Image view
        self.image_label = QLabel("Open an image to begin")
        self.image_label.setAlignment(ALIGN_CENTER)
        self.image_label.setMinimumSize(400, 300)

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)

        # Outputs
        self.caption_edit = QTextEdit()
        self.caption_edit.setReadOnly(True)
        self.caption_edit.setPlaceholderText("Caption will appear here…")

        self.items_list = QListWidget()

        # Layouts
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.open_button)
        top_bar.addWidget(self.analyze_button)
        top_bar.addWidget(self.progress_bar)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Caption"))
        right_col.addWidget(self.caption_edit, 1)
        right_col.addWidget(QLabel("Items Detected"))
        right_col.addWidget(self.items_list, 2)

        main_row = QHBoxLayout()
        main_row.addWidget(self.image_scroll, 3)
        right_container = QWidget()
        right_container.setLayout(right_col)
        main_row.addWidget(right_container, 2)

        root = QVBoxLayout()
        root.addLayout(top_bar)
        root.addLayout(main_row, 1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # Signals
        self.open_button.clicked.connect(self.choose_image)
        self.analyze_button.clicked.connect(self.start_analysis)

        self.statusBar().showMessage("Ready")

    def choose_image(self):
        dlg = QFileDialog(self, "Choose an image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilters([
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif *.heic *.heif)",
            "All files (*)",
        ])
        if dlg.exec_():
            paths = dlg.selectedFiles()
            if paths:
                self.load_image(paths[0])

    def load_image(self, path: str):
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", f"File not found:\n{path}")
            return
        pix = QPixmap(path)
        if pix.isNull():
            # Fallback to Pillow to load formats not supported by Qt plugins (e.g., HEIC)
            try:
                from PIL import Image
                try:
                    import pillow_heif  # Optional HEIC/HEIF support
                    pillow_heif.register_heif_opener()
                except Exception:
                    pass
                pil_img = Image.open(path)
                if pil_img.mode not in ("RGB", "RGBA"):
                    pil_img = pil_img.convert("RGBA")
                from PIL.ImageQt import ImageQt
                qimage = ImageQt(pil_img)
                pix = QPixmap.fromImage(qimage)
                if pix.isNull():
                    raise RuntimeError("Conversion to QPixmap failed.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Unsupported image or failed to load.\n\n{e}")
                return
        self.current_image_path = path
        self.setWindowTitle(f"What's In The Room - {os.path.basename(path)}")
        self.original_pixmap = pix
        self.display_pixmap(pix)
        self.caption_edit.clear()
        self.items_list.clear()
        self.progress_bar.setValue(0)
        self.analyze_button.setEnabled(True)
        self.statusBar().showMessage("Image loaded. Click Analyze to begin.")

    def display_pixmap(self, pix: QPixmap):
        # Fit image to scroll area while preserving aspect ratio
        container = self.image_scroll.viewport().size()
        scaled = pix.scaled(container, KEEP_ASPECT_RATIO, SMOOTH_TRANSFORMATION)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-fit current image on resize
        if self.original_pixmap is not None:
            self.display_pixmap(self.original_pixmap)

    def start_analysis(self):
        if not self.current_image_path:
            return
        self.analyze_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Analyzing…")

        self.worker = AnalyzerWorker(self.current_image_path)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self, caption: str, detections: List[Tuple[str, float]]):
        self.analyze_button.setEnabled(True)
        self.caption_edit.setText(caption or "(No caption)")

        # Aggregate detections by label
        counts: Dict[str, int] = {}
        confs: Dict[str, float] = {}
        for label, score in detections:
            counts[label] = counts.get(label, 0) + 1
            confs[label] = max(confs.get(label, 0.0), float(score))

        self.items_list.clear()
        for label in sorted(counts.keys()):
            count = counts[label]
            score = confs[label]
            text = f"{label}  (x{count}, max conf {score:.2f})"
            self.items_list.addItem(QListWidgetItem(text))
        self.statusBar().showMessage("Analysis complete")


def main():
    _ensure_qt_platform_plugin()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    try:
        ret = app.exec()
    except AttributeError:
        ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
    main()


