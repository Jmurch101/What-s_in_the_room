# What's In The Room

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/HF-Transformers-blue)](https://huggingface.co/docs/transformers)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)](https://docs.ultralytics.com)
[![Qt](https://img.shields.io/badge/Qt-PySide6%2FPyQt5-41cd52)](https://www.qt.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)

A PyQt app that captions an image and lists items detected in the scene.

## Features
- Open an image and view it in the UI
- Generate a natural-language caption (Transformers image-to-text)
- Detect and list items in the image (Ultralytics YOLO)
- Runs inference in a background thread to keep the UI responsive

## Description (Models & Process)
- Captioning model: `nlpconnect/vit-gpt2-image-captioning` (Vision Transformer encoder + GPT‑2 decoder) for concise scene descriptions.
- Object detection: Ultralytics YOLOv8n for fast, general-purpose item detection with class labels and confidences.
- Device selection: Uses CUDA if available, then Apple Metal (MPS) on Apple Silicon, else CPU.
- Robust image load: Loads via Qt; falls back to Pillow with optional HEIC/HEIF support when needed.

## How it works
1) You open an image in the GUI.
2) The app runs two parallel steps in a worker thread:
   - Captioning pipeline generates a short natural-language description of the scene.
   - YOLOv8 detects objects; results are aggregated into a de-duplicated list with counts and max confidence.
3) The UI updates with the caption and an item list without blocking the main thread.

## Setup

1) Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- First run will download model weights (captioning + YOLO). This may take a few minutes.
- On Apple Silicon, PyTorch will use MPS if available; otherwise CPU.

## Run
```bash
python main.py
```

## Troubleshooting
- If PyTorch or Ultralytics installation fails, try updating pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
- If you prefer PyQt6, you can install `PyQt6` and adapt imports in `main.py`.
- If downloads are blocked, manually download models or run once with internet access.

## License
MIT
