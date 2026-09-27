# =========================================================
# THREAD TUNING  — MUST be before importing torch / ultralytics
# =========================================================
import os

N_THREADS = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# =========================================================
# IMPORTS (torch-heavy)
# =========================================================
import io, base64, re, time, threading
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from starlette.concurrency import run_in_threadpool

# --- runtime thread limits (belt + suspenders) ---
import torch
torch.set_num_threads(N_THREADS)
cv2.setNumThreads(1)

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH     = os.getenv("MODEL_PATH", "yolo11n.pt")
OV_MODEL_DIR   = os.getenv("OV_MODEL_DIR", "yolo11n_openvino_model")
IMGSZ          = int(os.getenv("IMGSZ", "640"))

CONF_DEFAULT   = float(os.getenv("CONF_DEFAULT", "0.25"))
# NOTE: YOLO11n's COCO classes do NOT include a "hand" class (verified:
# 80 classes, no "hand"). Hands are handled by MediaPipe Stage 2, not YOLO.
CONF_BY_CLASS  = {"person": 0.45}

# COCO classes that produce false positives on clothing / small textures.
BLOCKED_CLASSES = {
    "tie", "sports ball", "frisbee", "kite", "skateboard",
    "surfboard", "snowboard", "baseball bat", "baseball glove",
    "tennis racket", "toothbrush", "hair drier",
}

# Classes we TRUST from YOLO and never let CLIP re-label.
# CLIP sees a black hoodie + zipper on a person crop and matches "deodorant
# bottle" or "thermos flask" — this guard blocks that class of bug entirely.
SKIP_CLIP_CLASSES = {"person", "hand"}

USE_HAND_ROI   = os.getenv("USE_HAND_ROI", "1") == "1"
USE_CLIP       = os.getenv("USE_CLIP",     "1") == "1"

# CLIP confidence bands
CLIP_OVERRIDE_STRONG  = float(os.getenv("CLIP_OVERRIDE_STRONG",  "0.50"))
CLIP_OVERRIDE_WEAK    = float(os.getenv("CLIP_OVERRIDE_WEAK",    "0.35"))
YOLO_WEAK             = float(os.getenv("YOLO_WEAK",             "0.40"))

# CLIP has no reject option by construction (softmax sums to 1), so we add:
#   1) a minimum top-1 vs top-2 gap (a real "margin")
#   2) a requirement that a label repeats before we lock it in
CLIP_MARGIN_MIN       = float(os.getenv("CLIP_MARGIN_MIN", "0.15"))
CLIP_CONFIRM_HITS     = int(os.getenv("CLIP_CONFIRM_HITS", "2"))

# Stage 2 (hand ROI) settings
HAND_ROI_PAD          = float(os.getenv("HAND_ROI_PAD", "1.4"))
HAND_ROI_IMGSZ        = int(os.getenv("HAND_ROI_IMGSZ", "960"))
HAND_ROI_CONF         = float(os.getenv("HAND_ROI_CONF", "0.15"))
HAND_ROI_UPSCALE      = float(os.getenv("HAND_ROI_UPSCALE", "2.0"))

# Per-session state TTL
SESSION_TTL_SECONDS   = int(os.getenv("SESSION_TTL_SECONDS", "300"))

# =========================================================
# APP
# =========================================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# =========================================================
# LOAD YOLO  (OpenVINO if folder exists, else PyTorch)
# =========================================================
def _load_yolo():
    if os.path.isdir(OV_MODEL_DIR):
        print(f"[Engine] Loading OpenVINO model from {OV_MODEL_DIR}")
        return YOLO(OV_MODEL_DIR)
    print(f"[Engine] Loading PyTorch model from {MODEL_PATH}")
    return YOLO(MODEL_PATH)

yolo_cpu = _load_yolo()

# YOLO inference isn't guaranteed thread-safe on one shared model instance;
# run_in_threadpool can dispatch concurrent requests to different threads.
_yolo_lock = threading.Lock()

def _yolo_predict(img_or_array, imgsz, conf):
    with _yolo_lock:
        return yolo_cpu(img_or_array, imgsz=imgsz, conf=conf,
                        agnostic_nms=False, verbose=False)[0]

# =========================================================
# MEDIAPIPE HANDS  (current Tasks API — legacy solutions API is removed)
# =========================================================
mp = None
hands = None

def _ensure_hand_model(model_path: str) -> str:
    if not os.path.exists(model_path):
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print(f"[Engine] Downloading hand landmark model to {model_path} ...")
        urllib.request.urlretrieve(url, model_path)
    return model_path

if USE_HAND_ROI:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_hand_model(os.getenv("HAND_MODEL_PATH", "hand_landmarker.task"))
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
        hands = mp_vision.HandLandmarker.create_from_options(options)
        print("[Engine] MediaPipe HandLandmarker loaded (Tasks API)")
    except Exception as e:
        print(f"[Engine] MediaPipe unavailable, hand-ROI disabled: {e}")
        hands = None

# =========================================================
# CLIP  (optional)
# =========================================================
clip_model = clip_proc = clip_text = clip_labels = None
CLIP_REJECT_LABELS = set()

if USE_CLIP:
    try:
        from transformers import CLIPModel, CLIPProcessor

        clip_labels = [
            # hand-held / desk objects
            "a photo of a calculator",
            "a photo of a tv remote control",
            "a photo of a smartphone",
            "a photo of a computer mouse",
            "a photo of a keyboard",
            "a photo of a pen",
            "a photo of a pencil",
            "a photo of a notebook",
            "a photo of a wallet",
            "a photo of a pair of glasses",
            "a photo of headphones",
            "a photo of a spoon",
            "a photo of a fork",
            "a photo of a key",
            "a photo of a coin",
            "a photo of scissors",
            # bottles & containers
            "a photo of a spray bottle",
            "a photo of a deodorant bottle",
            "a photo of a cosmetic bottle",
            "a photo of a water bottle",
            "a photo of a thermos flask",
            "a photo of a coffee mug",
            "a photo of a cup",
            "a photo of a can",
            "a photo of a pump bottle",
            "a photo of a grooming product bottle",
            "a photo of a shampoo bottle",
            "a photo of a body spray can",
            # scene
            "a photo of a person's face",
            "a photo of a hand",
            "a photo of a wall",
            "a photo of a piece of furniture",
            "a photo of a door",
            "a photo of a window",
            # generic "none of the above" anchors so softmax has somewhere
            # to send crops that don't match any real category
            "a photo of an unidentified small object",
            "a photo of food or a food wrapper",
            "a photo of a cable or charger",
        ]
        CLIP_REJECT_LABELS = {
            "unidentified small object",
            "food or a food wrapper",
            "cable or charger",
        }

        print("[Engine] Loading CLIP (first run may download ~600MB)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_text  = clip_proc(text=clip_labels, return_tensors="pt", padding=True)
        print("[Engine] CLIP loaded")
    except Exception as e:
        print(f"[Engine] CLIP unavailable, class re-naming disabled: {e}")
        clip_model = None
        CLIP_REJECT_LABELS = set()

# =========================================================
# HELPERS
# =========================================================
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------- Simple IoU tracker (for CLIP caching) ----------
class SimpleTracker:
    def __init__(self, iou_thresh=0.4, ttl=20):
        self.tracks = {}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.ttl = ttl

    def update(self, detections):
        for t in self.tracks.values():
            t["age"] += 1
        used = set()
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                # Identity is purely spatial (IoU). We do NOT gate on label
                # equality — YOLO flips labels ("cell phone" <-> "remote")
                # for the same object between frames; matching on the label
                # would spawn a new track and waste CLIP re-classification.
                if tid in used:
                    continue
                v = iou(det["box"], t["box"])
                if v > best_iou:
                    best_iou, best_id = v, tid
            if best_id is not None and best_iou >= self.iou_thresh:
                t = self.tracks[best_id]
                t["box"]   = det["box"]
                t["label"] = det["label"]
                t["age"]   = 0
                det["track_id"] = best_id
                used.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box": det["box"], "label": det["label"], "age": 0,
                    "clip_label": None, "clip_conf": 0.0,
                    "pending_label": None, "pending_count": 0,
                }
                det["track_id"] = tid
                used.add(tid)
        for tid in [t for t, v in self.tracks.items() if v["age"] > self.ttl]:
            del self.tracks[tid]
        return detections

    def get_clip(self, tid):
        t = self.tracks.get(tid)
        return (t["clip_label"], t["clip_conf"]) if t else (None, 0.0)

    def set_clip(self, tid, label, conf):
        if tid in self.tracks:
            self.tracks[tid]["clip_label"] = label
            self.tracks[tid]["clip_conf"]  = conf

    def bump_pending(self, tid, label):
        """Track a candidate CLIP label across frames; confirm once it repeats."""
        t = self.tracks.get(tid)
        if t is None:
            return 0
        if t["pending_label"] == label:
            t["pending_count"] += 1
        else:
            t["pending_label"] = label
            t["pending_count"] = 1
        return t["pending_count"]


# ---------- per-session state ----------
_sessions = {}
_sessions_lock = threading.Lock()

def get_session_state(session_id: str):
    now = time.time()
    with _sessions_lock:
        stale = [sid for sid, s in _sessions.items()
                 if now - s["last_seen"] > SESSION_TTL_SECONDS]
        for sid in stale:
            del _sessions[sid]
        if session_id not in _sessions:
            _sessions[session_id] = {"tracker": SimpleTracker(), "last_seen": now}
        _sessions[session_id]["last_seen"] = now
        return _sessions[session_id]["tracker"]


# ---------- MediaPipe → hand ROIs (pixel coords) ----------
def get_hand_rois(bgr, pad=HAND_ROI_PAD):
    if hands is None:
        return []
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb))
    result = hands.detect(mp_image)
    rois = []
    for landmarks in result.hand_landmarks:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw, bh = x2 - x1, y2 - y1
        X1 = max(0, int((x1 - bw*pad) * w))
        X2 = min(w, int((x2 + bw*pad) * w))
        Y1 = max(0, int((y1 - bh*pad) * h))
        Y2 = min(h, int((y2 + bh*pad) * h))
        if (X2 - X1) > 20 and (Y2 - Y1) > 20:
            rois.append((X1, Y1, X2, Y2))
    return rois


# ---------- CLIP classifier on a small crop ----------
def classify_crop(bgr_crop):
    """Returns (label, top1_conf, margin) where margin = top1 - top2 prob."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = clip_proc(images=pil, return_tensors="pt")
    with torch.inference_mode():
        out = clip_model(**{**clip_text, **inputs})
        probs = out.logits_per_image[0].softmax(dim=0)
    top2 = torch.topk(probs, 2)
    idx, idx2  = int(top2.indices[0]), int(top2.indices[1])
    conf, conf2 = float(top2.values[0]), float(top2.values[1])
    clean = re.sub(r"^a photo of (a |an )?", "", clip_labels[idx]).strip()
    return clean, conf, conf - conf2


# =========================================================
# INFERENCE PIPELINE
# =========================================================
def run_inference(image: Image.Image, tracker: "SimpleTracker"):
    rgb = np.array(image)
    bgr = rgb[:, :, ::-1].copy()
    h, w = bgr.shape[:2]

    detections = []

    # ---------- STAGE 1: full-frame YOLO ----------
    r1 = _yolo_predict(image, IMGSZ, CONF_DEFAULT)
    for b in r1.boxes:
        c     = float(b.conf[0])
        label = r1.names[int(b.cls[0])]
        if label in BLOCKED_CLASSES:
            continue
        if c < CONF_BY_CLASS.get(label, CONF_DEFAULT):
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        detections.append({"box": (x1, y1, x2, y2), "label": label, "conf": c})

    # ---------- STAGE 2: hand ROI re-detect ----------
    if hands is not None:
        for (rx1, ry1, rx2, ry2) in get_hand_rois(bgr, pad=HAND_ROI_PAD):
            crop = bgr[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            up = cv2.resize(crop, None,
                            fx=HAND_ROI_UPSCALE, fy=HAND_ROI_UPSCALE,
                            interpolation=cv2.INTER_CUBIC)
            r2 = _yolo_predict(up, HAND_ROI_IMGSZ, HAND_ROI_CONF)
            for b in r2.boxes:
                c     = float(b.conf[0])
                label = r2.names[int(b.cls[0])]
                if label in ("person", "hand"):
                    continue
                if label in BLOCKED_CLASSES:
                    continue
                ux1, uy1, ux2, uy2 = b.xyxy[0].tolist()
                x1 = ux1 / HAND_ROI_UPSCALE + rx1
                y1 = uy1 / HAND_ROI_UPSCALE + ry1
                x2 = ux2 / HAND_ROI_UPSCALE + rx1
                y2 = uy2 / HAND_ROI_UPSCALE + ry1
                detections.append({"box": (x1, y1, x2, y2),
                                   "label": label, "conf": c,
                                   "from_roi": True})

    # ---------- DEDUPE (ROI detections get a small bonus) ----------
    def _dedupe_key(d):
        bonus = 0.15 if d.get("from_roi") else 0.0
        return -(d["conf"] + bonus)

    merged = []
    for d in sorted(detections, key=_dedupe_key):
        if any(iou(m["box"], d["box"]) > 0.55 for m in merged):
            continue
        merged.append(d)
    detections = merged

    # ---------- STAGE 3: tracking + CLIP naming ----------
    detections = tracker.update(detections)

    if clip_model is not None:
        for d in detections:
            tid = d["track_id"]

            # --- HARD GUARD: never let CLIP re-label trusted classes ---
            # A person crop (black hoodie + silver zipper) matches
            # "deodorant bottle"/"thermos flask" in CLIP's softmax. Skip
            # CLIP entirely for these.
            if d["label"] in SKIP_CLIP_CLASSES:
                d["display_label"] = d["label"]
                d["display_conf"]  = d["conf"]
                continue

            cached_label, cached_conf = tracker.get_clip(tid)
            if cached_label is not None:
                d["display_label"] = cached_label
                d["display_conf"]  = cached_conf
                continue

            x1, y1, x2, y2 = (int(v) for v in d["box"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                d["display_label"] = d["label"]
                d["display_conf"]  = d["conf"]
                continue

            try:
                label, conf, margin = classify_crop(bgr[y1:y2, x1:x2])
                confident = (conf >= CLIP_OVERRIDE_STRONG
                             and margin >= CLIP_MARGIN_MIN)
                weakly_confident = (conf >= CLIP_OVERRIDE_WEAK
                                    and margin >= CLIP_MARGIN_MIN
                                    and d["conf"] < YOLO_WEAK)

                if (confident or weakly_confident) and label not in CLIP_REJECT_LABELS:
                    hits = tracker.bump_pending(tid, label)
                    if hits >= CLIP_CONFIRM_HITS:
                        d["display_label"] = label
                        d["display_conf"]  = conf
                        tracker.set_clip(tid, label, conf)
                    else:
                        # seen once so far — not confirmed yet
                        d["display_label"] = d["label"]
                        d["display_conf"]  = d["conf"]
                else:
                    d["display_label"] = d["label"]
                    d["display_conf"]  = d["conf"]
            except Exception:
                d["display_label"] = d["label"]
                d["display_conf"]  = d["conf"]
    else:
        for d in detections:
            d["display_label"] = d["label"]
            d["display_conf"]  = d["conf"]

    return detections, w, h


# =========================================================
# PYDANTIC MODELS
# =========================================================
class FramePayload(BaseModel):
    image_base64: str
    session_id: Optional[str] = "_default"

class ErrorPayload(BaseModel):
    error_message: str
    stack_trace: Optional[str] = None


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
async def health():
    return {
        "yolo":             MODEL_PATH if not os.path.isdir(OV_MODEL_DIR) else OV_MODEL_DIR,
        "hand_roi":         hands is not None,
        "hand_roi_pad":     HAND_ROI_PAD,
        "hand_roi_imgsz":   HAND_ROI_IMGSZ,
        "clip":             clip_model is not None,
        "imgsz":            IMGSZ,
        "threads":          N_THREADS,
        "conf_default":     CONF_DEFAULT,
        "blocked_classes":  sorted(BLOCKED_CLASSES),
        "skip_clip":        sorted(SKIP_CLIP_CLASSES),
        "active_sessions":  len(_sessions),
    }


@app.post("/detect")
async def detect(payload: FramePayload):
    try:
        base64_str = payload.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size

        tracker = get_session_state(payload.session_id or "_default")
        detections, _, _ = await run_in_threadpool(run_inference, image, tracker)

        formatted_boxes = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            formatted_boxes.append({
                # NOTE: normalized 0-1, [ymin, xmin, ymax, xmax].
                # The field name mirrors Gemini's box_2d, but Gemini's
                # convention is 0-1000, not 0-1. Confirm your frontend
                # multiplies by image dimensions (not divides by 1000).
                "box_2d": [
                    round(y1 / img_h, 4),
                    round(x1 / img_w, 4),
                    round(y2 / img_h, 4),
                    round(x2 / img_w, 4),
                ],
                "label":      d["display_label"],
                "confidence": round(float(d["display_conf"]), 2),
            })
        return {"boxes": formatted_boxes}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": "detection_failed",
                                     "message": str(e)})


@app.post("/diagnose-error")
async def diagnose_error(payload: ErrorPayload):
    try:
        # Lazy + guarded import: a broken/missing google-genai install
        # should not crash the whole server at startup.
        try:
            from google import genai
        except ImportError:
            return JSONResponse(status_code=500,
                                content={"error": "diagnosis_failed",
                                         "message": "google-genai package not installed"})

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500,
                                content={"error": "diagnosis_failed",
                                         "message": "GEMINI_API_KEY not set"})
        ai_client = genai.Client(api_key=api_key)
        prompt = f"""
        System fault detected in live vision app:
        Error: {payload.error_message}
        Stack Trace: {payload.stack_trace or 'N/A'}
        Provide a concise 2-sentence cause and 2 bullet points to resolve it.
        """
        response = await run_in_threadpool(
            ai_client.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"reasoning": response.text}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": "diagnosis_failed",
                                     "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)