"""
Local YOLO11 & Gemini AI Diagnostic Server
-------------------------------------------
High-throughput FastAPI microservice powering real-time 30+ FPS local object detection
and Gemini-backed automated root cause diagnostic reasoning for vision pipelines.

Endpoints:
- POST /detect: Ingests base64 frame, runs YOLO11 inference, returns normalized [ymin, xmin, ymax, xmax] boxes.
- POST /diagnose-error: Ingests client/server error details, uses Gemini 2.0 Flash to synthesize root-cause fixes.

Run locally:
    python server_yolo.py
    # or: uvicorn server_yolo:app --host 0.0.0.0 --port 8000
"""

import os
import io
import time
import base64
import logging
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO11Server")

app = FastAPI(title="Local YOLO11 Vision & Diagnostic Server")

# Enable CORS for browser frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global YOLO model holder
yolo_model = None
try:
    from ultralytics import YOLO
    logger.info("Loading YOLO11 model (yolo11n.pt)...")
    yolo_model = YOLO("yolo11n.pt")
    logger.info("YOLO11 model initialized successfully.")
except Exception as e:
    logger.warning("ultralytics YOLO11 could not be loaded automatically (%s). Falling back to visual tracking.", e)

# Optional Gemini SDK for diagnostics
gemini_client = None
try:
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini Diagnostic Client initialized.")
except Exception as e:
    logger.warning("Gemini Client initialization skipped: %s", e)


class DetectRequest(BaseModel):
    image_base64: str


class DetectionBoxOutput(BaseModel):
    box_2d: List[float]  # [ymin, xmin, ymax, xmax] normalized [0..1]
    label: str
    confidence: float


class DetectResponse(BaseModel):
    boxes: List[DetectionBoxOutput]
    model: str
    inference_time_ms: float


class DiagnoseRequest(BaseModel):
    error_message: str
    stack_trace: Optional[str] = None


class DiagnoseResponse(BaseModel):
    reasoning: str


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "YOLO11 Local Server",
        "yolo_loaded": yolo_model is not None,
        "gemini_diagnostic_ready": gemini_client is not None,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """
    Accepts raw base64 frame, executes YOLO11 inference,
    and returns normalized [ymin, xmin, ymax, xmax] ratios (0.0 to 1.0).
    """
    start_t = time.perf_counter()
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        image_data = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        h, w = img.shape[:2]
        output_boxes: List[DetectionBoxOutput] = []

        if yolo_model is not None:
            results = yolo_model(img, verbose=False)
            for res in results:
                boxes = res.boxes
                if boxes is not None:
                    for b in boxes:
                        coords = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        conf = float(b.conf[0])
                        cls_idx = int(b.cls[0])
                        label = res.names.get(cls_idx, f"class_{cls_idx}")

                        # Convert to normalized [ymin, xmin, ymax, xmax] ratios (0..1)
                        xmin = max(0.0, min(1.0, coords[0] / w))
                        ymin = max(0.0, min(1.0, coords[1] / h))
                        xmax = max(0.0, min(1.0, coords[2] / w))
                        ymax = max(0.0, min(1.0, coords[3] / h))

                        output_boxes.append(
                            DetectionBoxOutput(
                                box_2d=[ymin, xmin, ymax, xmax],
                                label=label,
                                confidence=round(conf, 3),
                            )
                        )
            model_name = "YOLO11n (PyTorch)"
        else:
            # High-speed fallback motion/object contour detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area > 1200:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    ymin = y / h
                    xmin = x / w
                    ymax = (y + ch) / h
                    xmax = (x + cw) / w
                    output_boxes.append(
                        DetectionBoxOutput(
                            box_2d=[ymin, xmin, ymax, xmax],
                            label="detected_object",
                            confidence=0.88,
                        )
                    )
            model_name = "Edge Detector Fallback"

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 1)
        return DetectResponse(
            boxes=output_boxes,
            model=model_name,
            inference_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing /detect frame: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose-error", response_model=DiagnoseResponse)
async def diagnose_error(req: DiagnoseRequest):
    """
    Analyzes local runtime errors using Gemini 2.0 Flash reasoning.
    Called with a 10-second cooldown from the frontend to preserve quota limits.
    """
    logger.info("Received /diagnose-error request: %s", req.error_message)

    if gemini_client is not None:
        try:
            prompt = f"""You are a senior computer vision and systems diagnostics engineer.
Analyze the following error encountered by a local YOLO11 webcam client and suggest concise, actionable solutions:

Error: {req.error_message}
Stack: {req.stack_trace or 'N/A'}

Provide a 2-3 sentence root-cause diagnosis and recommended resolution."""
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            reasoning = response.text.strip() if response and response.text else "Diagnostic complete."
            return DiagnoseResponse(reasoning=reasoning)
        except Exception as err:
            logger.error("Gemini diagnosis failed: %s", err)

    # Heuristic fallback diagnosis when offline or API key missing
    err_low = req.error_message.lower()
    if "failed to fetch" in err_low or "econnrefused" in err_low or "connection" in err_low:
        reasoning = "Connection refused at http://localhost:8000. Verify the local YOLO11 Python server is running ('python server_yolo.py') and CORS is enabled."
    elif "404" in err_low:
        reasoning = "Endpoint not found on local backend. Ensure /detect and /diagnose-error routes are mounted on FastAPI."
    elif "429" in err_low:
        reasoning = "Rate limit reached. Ensure request frequency does not exceed local processing capacity."
    else:
        reasoning = f"Local error encountered: '{req.error_message}'. Check webcam device permissions, image encoding, and server logs."

    return DiagnoseResponse(reasoning=reasoning)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Local YOLO11 Server on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
