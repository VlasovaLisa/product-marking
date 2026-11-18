import time
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Tuple

import cv2
import numpy as np

from src.localization import crop_and_upscale, text_region
from src.postprocessing import postprocess_lines
from src.preprocessing import apply_clahe, connect_dots, mask_overlay
from src.rotation import rotate_image


class OcrEngine(Protocol):
    def readtext(self, image: np.ndarray, **kwargs) -> List[Tuple[list, str, float]]: ...

    def detect(self, image: np.ndarray) -> Tuple[list, list]: ...


@dataclass(frozen=True)
class PipelineConfig:
    mask_overlay: bool = True
    clahe: bool = False
    dots: str = "dilate3"
    rotations: Tuple[int, ...] = (0, 90, 180, 270)
    min_confidence: float = 0.0
    crop_to_text: bool = False
    postprocess: bool = True
    refine_side: int = 0
    refine_min_confidence: float = 0.6
    crop_target_side: int = 1600
    readtext_options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    lines: Tuple[str, ...]
    confidence: float
    angle: int
    elapsed_seconds: float


def _orientation_score(detections: Sequence[Tuple[list, str, float]]) -> float:
    return sum(confidence * sum(char.isalnum() for char in text) for _, text, confidence in detections)


class Pipeline:
    def __init__(self, engine: OcrEngine, config: PipelineConfig):
        self.engine = engine
        self.config = config

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        started = time.perf_counter()
        prepared = self._preprocess(image)
        best_angle, best_detections, best_score, best_image = 0, [], -1.0, prepared
        for angle in self.config.rotations:
            rotated = prepared if angle == 0 else rotate_image(prepared, angle)
            if self.config.crop_to_text:
                rotated = self._crop_to_text(rotated)
            detections = [
                d
                for d in self.engine.readtext(rotated, **self.config.readtext_options)
                if d[2] >= self.config.min_confidence
            ]
            score = _orientation_score(detections)
            if score > best_score:
                best_angle, best_detections, best_score, best_image = angle, detections, score, rotated
        if self.config.refine_side > 0 and best_detections:
            refined = self._read_enlarged_text(best_image, best_detections)
            refined_score = _orientation_score(refined)
            if refined_score > best_score:
                best_detections = refined
        lines = tuple(text for _, text, _ in best_detections)
        if self.config.postprocess:
            lines = postprocess_lines(lines)
        confidence = (
            sum(c for _, _, c in best_detections) / len(best_detections) if best_detections else 0.0
        )
        return RecognitionResult(
            text="\n".join(lines),
            lines=lines,
            confidence=confidence,
            angle=best_angle,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _read_enlarged_text(self, image: np.ndarray, detections: Sequence[Tuple[list, str, float]]):
        """Second pass: crop around the text found in the first pass and read it enlarged (cheap, unlike enlarging the frame)."""
        confident = [d for d in detections if d[2] >= self.config.refine_min_confidence] or list(detections)
        region = text_region([], [polygon for polygon, _, _ in confident], image.shape)
        if region is None:
            return []
        crop = crop_and_upscale(image, region, self.config.refine_side)
        return [
            d
            for d in self.engine.readtext(crop, **self.config.readtext_options)
            if d[2] >= self.config.min_confidence
        ]

    def _crop_to_text(self, image: np.ndarray) -> np.ndarray:
        horizontal, free = self.engine.detect(image)
        region = text_region(horizontal[0], free[0], image.shape)
        if region is None:
            return image
        return crop_and_upscale(image, region, self.config.crop_target_side)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.config.mask_overlay:
            image = mask_overlay(image)
        if self.config.clahe:
            image = apply_clahe(image)
        if self.config.dots:
            image = connect_dots(image, self.config.dots)
        return image
