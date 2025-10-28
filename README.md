# What's In The Room

A PyQt app that captions an image and lists items detected in the scene.

## Features
- Open an image and view it in the UI
- Generate a natural-language caption (Transformers image-to-text)
- Detect and list items in the image (Ultralytics YOLO)
- Runs inference in a background thread to keep the UI responsive

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
