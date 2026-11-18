import numpy as np

from src.engines import PaddleEngine


class FakePredictor:
    def __init__(self, pages):
        self.pages = pages

    def predict(self, image):
        return self.pages


def test_converts_paddle_page_to_readtext_shape():
    page = {
        "rec_polys": [np.array([[0, 0], [10, 0], [10, 5], [0, 5]])],
        "rec_texts": ["2023"],
        "rec_scores": [np.float32(0.97)],
    }
    detections = PaddleEngine(FakePredictor([page])).readtext(np.zeros((10, 10, 3), np.uint8))
    assert detections == [([[0, 0], [10, 0], [10, 5], [0, 5]], "2023", float(np.float32(0.97)))]


def test_no_pages_means_no_detections():
    assert PaddleEngine(FakePredictor([])).readtext(np.zeros((10, 10, 3), np.uint8)) == []
