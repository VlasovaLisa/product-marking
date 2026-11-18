import numpy as np

from src.pipeline import Pipeline, PipelineConfig


class FakeEngine:
    """Returns text only when the image is rotated the 'right' way, i.e. the marker pixel is in the top-left."""

    def readtext(self, image, **kwargs):
        if image[0, 0].any():
            return [([], "2023-B1", 0.9), ([], "x", 0.1)]
        return [([], "noise", 0.2)]


def make_image_with_marker_at_bottom_right() -> np.ndarray:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[-1, -1] = 255
    return image


def test_picks_orientation_with_best_score():
    pipeline = Pipeline(FakeEngine(), PipelineConfig(mask_overlay=False, rotations=(0, 90, 180, 270)))
    result = pipeline.recognize(make_image_with_marker_at_bottom_right())
    assert result.angle == 180
    assert result.lines == ("2023-B1", "x")


def test_empty_detections_give_empty_text():
    class Silent:
        def readtext(self, image, **kwargs):
            return []

    result = Pipeline(Silent(), PipelineConfig(mask_overlay=False)).recognize(np.zeros((10, 10, 3), np.uint8))
    assert result.text == ""
    assert result.confidence == 0.0


def test_min_confidence_filters_detections():
    pipeline = Pipeline(FakeEngine(), PipelineConfig(mask_overlay=False, min_confidence=0.5))
    image = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert pipeline.recognize(image).lines == ("2023-B1",)


def test_overlay_masking_blanks_top_left():
    seen = {}

    class Spy:
        def readtext(self, image, **kwargs):
            seen["corner"] = image[0, 0].copy()
            return []

    Pipeline(Spy(), PipelineConfig(mask_overlay=True, rotations=(0,))).recognize(np.full((720, 1280, 3), 200, np.uint8))
    assert not seen["corner"].any()


def test_postprocess_flag_fixes_heat_letter():
    class Reader:
        def readtext(self, image, **kwargs):
            return [([], "2023-8208567", 0.9)]

    image = np.zeros((10, 10, 3), np.uint8)
    assert Pipeline(Reader(), PipelineConfig(mask_overlay=False, postprocess=False)).recognize(image).lines == ("2023-8208567",)
    fixed = Pipeline(Reader(), PipelineConfig(mask_overlay=False)).recognize(image)
    assert fixed.lines == ("2023-В208567",)


def test_dot_joining_bridges_gap_between_dots():
    from src.preprocessing import connect_dots

    image = np.zeros((20, 24, 3), dtype=np.uint8)
    image[9:12, 8:11] = 255
    image[9:12, 13:16] = 255
    for mode in ("dilate3", "close5"):
        joined = connect_dots(image, mode)[:, :, 0]
        assert joined[10, 11] > 0 and joined[10, 12] > 0


SMALL_BOX = [[0, 0], [4, 0], [4, 4], [0, 4]]


def test_refine_rereads_enlarged_text_region_and_keeps_better_result():
    class SizeAware:
        def readtext(self, image, **kwargs):
            if image.shape[1] >= 12:
                return [(SMALL_BOX, "2023-В1234567", 0.9)]
            return [(SMALL_BOX, "20", 0.9)]

    image = np.zeros((10, 10, 3), np.uint8)
    plain = Pipeline(SizeAware(), PipelineConfig(mask_overlay=False, dots="", rotations=(0,))).recognize(image)
    refined = Pipeline(
        SizeAware(), PipelineConfig(mask_overlay=False, dots="", rotations=(0,), refine_side=100)
    ).recognize(image)
    assert plain.lines == ("20",)
    assert refined.lines == ("2023-В1234567",)


def test_refine_keeps_first_reading_when_second_pass_is_worse():
    class Degrades:
        def readtext(self, image, **kwargs):
            return [(SMALL_BOX, "2023-В1", 0.9)] if image.shape[1] == 10 else [(SMALL_BOX, "x", 0.1)]

    image = np.zeros((10, 10, 3), np.uint8)
    result = Pipeline(
        Degrades(), PipelineConfig(mask_overlay=False, dots="", rotations=(0,), refine_side=100)
    ).recognize(image)
    assert result.lines == ("2023-В1",)
