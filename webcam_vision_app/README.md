# VisionGround: Modular PyQt6 + OpenCV + Gemini 2.0 Flash Spatial Grounding

A production-grade Python desktop application that requests webcam permissions, streams live video at 30 FPS, detects objects on demand or continuously, and overlays colored bounding boxes with exact object labels directly onto the video feed using Gemini 2.0 Flash spatial grounding coordinates.

---

## 1. Architecture & Modular Structure

```
/webcam_vision_app
│── main.py            # Main GUI window, event loop, and aspect-ratio frame rendering pipeline
│── camera_manager.py  # Threaded OpenCV camera stream with permission verification and disconnect guardrails
│── vision_engine.py   # Gemini 2.0 Flash client, 2D spatial coordinate parser, & QRunnable worker
│── config.py          # Centralized configuration (FPS, coordinate scaling, palette, troubleshooting)
│── requirements.txt   # Python package dependencies
└── .env.example       # Environment variable template
```

---

## 2. Core Technical Highlights & Edge-Case Guardrails

1. **Non-Blocking Vision Inference (Edge Case 1)**
   - API latency from Gemini never locks the GUI or camera frame pipeline.
   - Inference runs on dedicated `QRunnable` workers managed by `QThreadPool`.
   - The OpenCV camera thread streams at a consistent 30 FPS while AI inference runs asynchronously in the background.

2. **Letterbox & Aspect Ratio Coordinate Normalization (Edge Case 2)**
   - Gemini spatial grounding returns coordinates normalized between `0` and `1000` in `[ymin, xmin, ymax, xmax]`.
   - The custom `VideoCanvasWidget` calculates the exact rendered dimension and letterbox/pillarbox offsets (`offset_x`, `offset_y`).
   - Normalization math converts 0-1000 coordinates to match the exact displayed video area without distortion when resizing the window.

3. **Defensive Model Parsing & Schema Guardrails (Edge Case 3)**
   - The vision parser cleanly strips markdown wrappers (` ```json ... ``` `) and uses regex fallback if model outputs extra commentary.
   - Coordinates are clamped within `[0, 1000]`, inverted coordinates are flipped safely, and malformed items are ignored without crashing.

4. **Webcam Disconnections & Hardware Unavailability (Edge Case 4)**
   - OpenCV `cap.read()` failures are tracked.
   - If the webcam is unplugged or hijacked by Zoom/Teams mid-stream, the stream stops gracefully, displays a "Camera Disconnected" overlay, and allows device re-scanning.

---

## 3. Installation & Quick Start

### Prerequisites
- Python 3.10+
- Webcam / video capture device
- Google Gemini API Key

### Steps

```bash
# 1. Navigate to the application folder
cd webcam_vision_app

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
cp .env.example .env
# Edit .env and enter your GEMINI_API_KEY

# 5. Run the Desktop GUI Application
python main.py

# 6. Or Run the Local YOLO11 + Gemini Diagnostic FastAPI Server (Port 8000)
python server_yolo.py
```

The FastAPI server provides:
- `POST http://localhost:8000/detect` (30+ FPS YOLO11 inference with normalized [0..1] coordinates)
- `POST http://localhost:8000/diagnose-error` (Gemini error diagnosis with 10s cooldown guard)

