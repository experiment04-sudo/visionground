"""
Vision Engine Module
--------------------
Asynchronous Gemini 2.0 Flash spatial grounding client with 2D coordinate parsing,
thread-safe QRunnable worker execution, and defensive parsing guardrails.
"""

import io
import json
import re
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from config import CONFIG

logger = logging.getLogger("VisionEngine")

# Optional import guard for google-genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False
    logger.warning("google-genai SDK is not installed. Install with 'pip install google-genai'.")


@dataclass
class DetectionBox:
    """
    Standardized spatial object detection box.
    Coordinates are normalized between 0.0 and 1000.0: [ymin, xmin, ymax, xmax].
    """
    label: str
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    confidence: Optional[float] = None
    color_rgb: Tuple[int, int, int] = (0, 217, 255)

    def to_pixel_box(self, rendered_width: int, rendered_height: int, offset_x: int = 0, offset_y: int = 0) -> Tuple[int, int, int, int]:
        """
        Converts normalized 0-1000 coordinates into pixel coordinates
        matching the actual rendered video rectangle inside the display widget.
        Returns: (left, top, width, height) in pixels.
        """
        norm_scale = CONFIG.COORD_SCALE  # 1000.0
        
        # Clamp coordinates defensively
        c_ymin = max(0.0, min(norm_scale, self.ymin))
        c_xmin = max(0.0, min(norm_scale, self.xmin))
        c_ymax = max(0.0, min(norm_scale, self.ymax))
        c_xmax = max(0.0, min(norm_scale, self.xmax))

        # Scale to rendered dimensions
        left = int(offset_x + (c_xmin / norm_scale) * rendered_width)
        top = int(offset_y + (c_ymin / norm_scale) * rendered_height)
        width = int(((c_xmax - c_xmin) / norm_scale) * rendered_width)
        height = int(((c_ymax - c_ymin) / norm_scale) * rendered_height)

        # Minimum dimension guard to prevent 0-sized boxes
        width = max(2, width)
        height = max(2, height)

        return left, top, width, height


class VisionSignals(QObject):
    """Signals for thread-safe asynchronous communication between VisionWorker and GUI."""
    started = pyqtSignal()
    finished = pyqtSignal(list, float)  # (List[DetectionBox], latency_seconds)
    error = pyqtSignal(str)              # Error message


class VisionInferenceWorker(QRunnable):
    """
    Dedicated QRunnable worker for Edge Case 1: GUI Freezing During API Calls.
    Executes the Gemini 2.0 Flash spatial grounding API call on a background thread pool,
    leaving the 30 FPS video thread and GUI event loop completely uninterrupted.
    """

    SPATIAL_GROUNDING_PROMPT = (
        "Locate and detect the primary objects in this image. "
        "For each detected object, return a JSON array of objects with the exact format:\n"
        "[\n"
        "  {\"label\": \"object name\", \"box_2d\": [ymin, xmin, ymax, xmax]},\n"
        "  ...\n"
        "]\n"
        "Important rules:\n"
        "1. Coordinates must be integers normalized between 0 and 1000 [ymin, xmin, ymax, xmax].\n"
        "2. Labels must be concise, descriptive object names (e.g. 'coffee mug', 'smartphone', 'keyboard', 'person').\n"
        "3. Output strictly valid JSON without preamble or markdown formatting."
    )

    def __init__(self, frame_bgr: np.ndarray, api_key: str, model_name: str = CONFIG.MODEL_NAME):
        super().__init__()
        # Store a deep copy of the frame to prevent concurrency race conditions with the camera thread
        self.frame_bgr = frame_bgr.copy()
        self.api_key = api_key
        self.model_name = model_name
        self.signals = VisionSignals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self):
        """Worker execution on background QThreadPool thread."""
        start_time = time.perf_counter()
        self.signals.started.emit()

        if not self.api_key:
            self.signals.error.emit(
                "Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable."
            )
            return

        if not GENAI_AVAILABLE:
            self.signals.error.emit(
                "Google GenAI SDK is not installed. Please run: pip install google-genai"
            )
            return

        try:
            # 1. Encode OpenCV BGR frame to JPEG bytes
            success, buffer = cv2.imencode(".jpg", self.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Failed to encode frame to JPEG for vision inference.")
            jpeg_bytes = buffer.tobytes()

            # 2. Initialize GenAI client
            client = genai.Client(api_key=self.api_key)

            # 3. Call Gemini 2.0 Flash with automatic retry and model fallback
            config_params = {}
            if types is not None:
                config_params = {
                    "temperature": 0.2,
                    "response_mime_type": "application/json"
                }

            candidate_models = [self.model_name]
            if self.model_name != "gemini-1.5-flash":
                candidate_models.append("gemini-1.5-flash")

            response = None
            last_err = None

            for model in candidate_models:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=model,
                            contents=[
                                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                                self.SPATIAL_GROUNDING_PROMPT
                            ],
                            config=config_params if config_params else None
                        )
                        break
                    except Exception as err:
                        last_err = err
                        err_str = str(err)
                        is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str or "Resource has been exhausted" in err_str
                        if is_503 and attempt < max_retries - 1:
                            delay = 1.0 if attempt == 0 else 2.0
                            logger.warning(f"[Gemini Vision] 503/UNAVAILABLE on {model} (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                        logger.warning(f"[Gemini Vision] Attempt on {model} failed: {err_str}")
                        break
                if response is not None:
                    break

            if response is None and last_err is not None:
                raise last_err

            latency = time.perf_counter() - start_time
            raw_text = response.text or ""
            logger.info(f"Gemini spatial response received in {latency:.2f}s: {raw_text[:120]}...")

            # 4. Defensive coordinate parsing (Edge Case 3)
            detections = self._parse_spatial_output(raw_text)
            self.signals.finished.emit(detections, latency)

        except Exception as exc:
            logger.exception("Error during Gemini spatial grounding inference")
            self.signals.error.emit(f"Inference error: {str(exc)}")

    def _parse_spatial_output(self, raw_text: str) -> List[DetectionBox]:
        """
        Defensive Parser for Edge Case 3: Malformed Model Outputs.
        Handles:
        - Markdown wrapped json blocks (```json ... ```)
        - Malformed array envelopes or string keys
        - Clamping values to [0, 1000]
        - Ensuring ymin < ymax and xmin < xmax
        """
        cleaned_text = raw_text.strip()
        
        # Regex to strip ```json ... ``` code fences
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
        if markdown_match:
            cleaned_text = markdown_match.group(1).strip()

        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Fallback: search for first '[' and last ']'
            start_bracket = cleaned_text.find("[")
            end_bracket = cleaned_text.rfind("]")
            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                try:
                    data = json.loads(cleaned_text[start_bracket:end_bracket + 1])
                except json.JSONDecodeError as err:
                    logger.error(f"Failed fallback JSON decode: {err}")
                    return []
            else:
                logger.error("Response contains no valid JSON array.")
                return []

        # Data might be a dict with key 'objects' or 'detections', or directly a list
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for possible_key in ["objects", "detections", "items", "boxes"]:
                if possible_key in data and isinstance(data[possible_key], list):
                    items = data[possible_key]
                    break

        detections: List[DetectionBox] = []
        palette = CONFIG.PALETTE

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            label = str(item.get("label", item.get("name", "object"))).strip()
            # Try getting box_2d or [ymin, xmin, ymax, xmax]
            box = item.get("box_2d") or item.get("box") or item.get("bbox")
            if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            try:
                ymin, xmin, ymax, xmax = [float(v) for v in box]

                # Defensive normalization sanity: ensure min < max
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                if xmin > xmax:
                    xmin, xmax = xmax, xmin

                # Discard zero-area detections
                if (ymax - ymin) < 5 or (xmax - xmin) < 5:
                    continue

                color = palette[idx % len(palette)]
                conf = float(item["confidence"]) if "confidence" in item else None

                detections.append(DetectionBox(
                    label=label,
                    ymin=ymin,
                    xmin=xmin,
                    ymax=ymax,
                    xmax=xmax,
                    confidence=conf,
                    color_rgb=color
                ))
            except (ValueError, TypeError) as conv_err:
                logger.debug(f"Skipping malformed detection item {item}: {conv_err}")
                continue

        logger.info(f"Successfully parsed {len(detections)} spatial detection boxes.")
        return detections
