from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Region = Tuple[int, int, int, int]


def _box_bounds(horizontal: Sequence[Sequence[float]], free: Sequence[Sequence[Sequence[float]]]) -> list:
    bounds = [(x_min, y_min, x_max, y_max) for x_min, x_max, y_min, y_max in horizontal]
    for polygon in free:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        bounds.append((min(xs), min(ys), max(xs), max(ys)))
    return bounds


def text_region(
    horizontal: Sequence[Sequence[float]],
    free: Sequence[Sequence[Sequence[float]]],
    image_shape: Tuple[int, ...],
    padding_ratio: float = 0.15,
) -> Optional[Region]:
    """Bounding box (x0, y0, x1, y1) around all detected text boxes, padded and clipped to the image."""
    bounds = _box_bounds(horizontal, free)
    if not bounds:
        return None
    height, width = image_shape[:2]
    x0 = min(b[0] for b in bounds)
    y0 = min(b[1] for b in bounds)
    x1 = max(b[2] for b in bounds)
    y1 = max(b[3] for b in bounds)
    pad_x = (x1 - x0) * padding_ratio
    pad_y = (y1 - y0) * padding_ratio
    return (
        max(0, int(x0 - pad_x)),
        max(0, int(y0 - pad_y)),
        min(width, int(x1 + pad_x)),
        min(height, int(y1 + pad_y)),
    )


def crop_and_upscale(image: np.ndarray, region: Region, target_max_side: int, max_scale: float = 4.0) -> np.ndarray:
    x0, y0, x1, y1 = region
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return image
    scale = min(target_max_side / max(crop.shape[:2]), max_scale)
    if scale <= 1.0:
        return crop
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
