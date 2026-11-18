import cv2
import numpy as np

ROTATION_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    return cv2.rotate(image, ROTATION_CODES[angle])
