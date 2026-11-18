from typing import List, Tuple

import numpy as np


class PaddleEngine:
    """Adapts PaddleOCR 3.x to the readtext() shape the pipeline expects: [(polygon, text, score)]."""

    def __init__(self, predictor):
        self.predictor = predictor

    def readtext(self, image: np.ndarray, **kwargs) -> List[Tuple[list, str, float]]:
        results = self.predictor.predict(image)
        if not results:
            return []
        page = results[0]
        return [
            (np.asarray(polygon).tolist(), text, float(score))
            for polygon, text, score in zip(page["rec_polys"], page["rec_texts"], page["rec_scores"])
        ]


def build_engine(name: str):
    """Create the OCR engine once; model files are downloaded on the first run."""
    if name == "paddle":
        from paddleocr import PaddleOCR

        return PaddleEngine(
            PaddleOCR(
                lang="ru",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        )
    import easyocr

    return easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
