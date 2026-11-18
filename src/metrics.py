from dataclasses import dataclass
from typing import Sequence

from src.text import normalize_field

MIN_FRAGMENT_LENGTH = 3


def edit_distance_to_best_fragment(reference: str, text: str) -> int:
    """Levenshtein distance from `reference` to the closest fragment of `text` (text outside the fragment is free)."""
    previous = list(range(len(reference) + 1))
    best = previous[-1]
    for char in text:
        current = [0]
        for i, ref_char in enumerate(reference, start=1):
            cost = 0 if ref_char == char else 1
            current.append(min(previous[i - 1] + cost, previous[i] + 1, current[i - 1] + 1))
        previous = current
        best = min(best, previous[-1])
    return best


def is_measurable(reference: str) -> bool:
    """A 1-2 character fragment can be found in almost any text, so such a field cannot be scored honestly."""
    return len(reference) >= MIN_FRAGMENT_LENGTH


@dataclass(frozen=True)
class ImageScore:
    heat_edits: int
    heat_length: int
    sheet_edits: int
    sheet_length: int

    @property
    def edits(self) -> int:
        return self.heat_edits + self.sheet_edits

    @property
    def length(self) -> int:
        return self.heat_length + self.sheet_length


def score_image(ref_heat: str, ref_sheet: str, ocr_lines: Sequence[str]) -> ImageScore:
    text = normalize_field(" ".join(ocr_lines))
    sheet_measurable = is_measurable(ref_sheet)
    return ImageScore(
        heat_edits=edit_distance_to_best_fragment(ref_heat, text),
        heat_length=len(ref_heat),
        sheet_edits=edit_distance_to_best_fragment(ref_sheet, text) if sheet_measurable else 0,
        sheet_length=len(ref_sheet) if sheet_measurable else 0,
    )


def cer(edits: int, length: int) -> float:
    return edits / length if length else 0.0


HIT_MAX_EDITS = 2
NOISE_SEEDS = range(5)


def hit_rate(heat_edits: Sequence[int], max_edits: int = HIT_MAX_EDITS) -> float:
    return sum(edits <= max_edits for edits in heat_edits) / len(heat_edits) if heat_edits else 0.0


def noise_floor_heat_cer(references: Sequence[str], ocr_lines: Sequence[Sequence[str]]) -> float:
    """Heat CER when each OCR output is scored against another image's reference. A real reading must beat this,
    otherwise the metric is rewarding chance matches (extra OCR text costs nothing in fragment matching)."""
    import random

    edits = length = 0
    for seed in NOISE_SEEDS:
        shuffled = list(references)
        random.Random(seed).shuffle(shuffled)
        for own, wrong, lines in zip(references, shuffled, ocr_lines):
            if wrong == own:
                continue
            edits += edit_distance_to_best_fragment(wrong, normalize_field(" ".join(lines)))
            length += len(wrong)
    return cer(edits, length)
