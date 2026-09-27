"""
VisionGround Application Configuration
--------------------------------------
Centralized configuration for camera settings, Gemini 2.0 Flash spatial grounding,
PyQt6 frame rendering pipelines, and high-contrast bounding box styling.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    # --------------------------------------------------------------------------
    # Google GenAI / Gemini API Settings
    # --------------------------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # Default to Gemini 3.6 Flash as required for spatial grounding
    MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    
    # Grounding coordinate system: Gemini spatial grounding outputs coordinates
    # normalized on a 0 to 1000 integer scale: [ymin, xmin, ymax, xmax]
    COORD_SCALE: float = 1000.0
    
    # Analysis timings
    CONTINUOUS_INFERENCE_INTERVAL_MS: int = 1500  # 1.5 seconds per live inference
    INFERENCE_TIMEOUT_SEC: float = 8.0
    
    # --------------------------------------------------------------------------
    # Video & Camera Settings
    # --------------------------------------------------------------------------
    TARGET_FPS: int = 30
    FRAME_INTERVAL_MS: int = int(1000 / TARGET_FPS)  # ~33 ms per frame
    DEFAULT_CAMERA_INDEX: int = 0
    REQUESTED_WIDTH: int = 1280
    REQUESTED_HEIGHT: int = 720
    
    # Max consecutive read failures before declaring camera disconnected
    MAX_CONSECUTIVE_READ_FAILURES: int = 5
    
    # --------------------------------------------------------------------------
    # Bounding Box & Label Overlay Styling
    # --------------------------------------------------------------------------
    BOX_BORDER_WIDTH: int = 2
    LABEL_FONT_FAMILY: str = "Segoe UI"
    LABEL_FONT_SIZE: int = 11
    LABEL_FONT_BOLD: bool = True
    LABEL_PADDING_H: int = 7
    LABEL_PADDING_V: int = 4
    LABEL_CORNER_RADIUS: int = 4
    BOX_FILL_ALPHA: int = 35  # 0-255 opacity for semi-transparent box fill
    
    # Distinct, high-contrast color palette (R, G, B) for varied object classes
    PALETTE: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 217, 255),    # Vibrant Cyan
        (16, 185, 129),   # Emerald Green
        (245, 158, 11),   # Warm Amber
        (239, 68, 68),    # Coral Crimson
        (168, 85, 247),   # Vivid Purple
        (236, 72, 153),   # Hot Pink
        (59, 130, 246),   # Electric Blue
        (132, 204, 22),   # Bright Lime
        (20, 184, 166),   # Rich Teal
        (249, 115, 22),   # Solar Orange
    ])

    # --------------------------------------------------------------------------
    # Troubleshooting & Permission Help Texts
    # --------------------------------------------------------------------------
    PERMISSION_TROUBLESHOOTING_GUIDE: str = (
        "Camera Access Denied or Unavailable!\n\n"
        "Please check the following:\n"
        "1. Application Permissions:\n"
        "   - macOS: System Settings -> Privacy & Security -> Camera -> Allow Terminal / Python.\n"
        "   - Windows: Settings -> Privacy & Security -> Camera -> Allow desktop apps.\n"
        "   - Linux: Ensure your user is in the 'video' group (e.g., 'sudo usermod -a -G video $USER').\n\n"
        "2. Device Occupation:\n"
        "   - Ensure Zoom, Teams, Google Meet, or OBS is not locking the webcam exclusively.\n\n"
        "3. Hardware Connection:\n"
        "   - Verify that the USB cable or built-in webcam is properly connected and detected."
    )


# Singleton configuration instance
CONFIG = AppConfig()
