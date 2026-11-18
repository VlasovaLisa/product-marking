import cv2
import numpy as np

OVERLAY_WIDTH = 125
OVERLAY_HEIGHT = 55
REFERENCE_WIDTH = 1280


def mask_overlay(image: np.ndarray) -> np.ndarray:
    """Blank the camera stamp (name and time) in the top-left corner so OCR does not read it as marking."""
    scale = image.shape[1] / REFERENCE_WIDTH
    masked = image.copy()
    masked[: int(OVERLAY_HEIGHT * scale), : int(OVERLAY_WIDTH * scale)] = 0
    return masked


def apply_clahe(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


DOT_JOINING_MODES = {
    "dilate3": lambda gray: cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
    "close5": lambda gray: cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
}


def connect_dots(image: np.ndarray, mode: str) -> np.ndarray:
    """Join the bright dots of stamped (dotted) marking into strokes so OCR sees characters, not scattered points."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(DOT_JOINING_MODES[mode](gray), cv2.COLOR_GRAY2BGR)
