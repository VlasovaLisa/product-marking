import numpy as np

from src.localization import crop_and_upscale, text_region


def test_region_covers_all_boxes_with_padding():
    horizontal = [[100, 200, 50, 80]]
    free = [[[300, 120], [400, 120], [400, 200], [300, 200]]]
    x0, y0, x1, y1 = text_region(horizontal, free, (720, 1280, 3), padding_ratio=0.1)
    assert x0 < 100 and y0 < 50 and x1 > 400 and y1 > 200


def test_region_is_clipped_to_image():
    region = text_region([[0, 1300, 0, 800]], [], (720, 1280, 3), padding_ratio=0.5)
    assert region == (0, 0, 1280, 720)


def test_no_boxes_gives_no_region():
    assert text_region([], [], (720, 1280, 3)) is None


def test_crop_is_upscaled_but_not_beyond_max_scale():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = crop_and_upscale(image, (0, 0, 100, 50), target_max_side=1600, max_scale=4.0)
    assert out.shape[:2] == (200, 400)


def test_large_crop_is_left_alone():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert crop_and_upscale(image, (0, 0, 1280, 720), target_max_side=1000).shape[:2] == (720, 1280)
