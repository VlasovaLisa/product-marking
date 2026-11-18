import numpy as np

from src.rotation import rotate_image


def test_90_swaps_dimensions():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert rotate_image(image, 90).shape[:2] == (1280, 720)


def test_180_twice_restores_original():
    image = np.random.default_rng(0).integers(0, 255, (10, 20, 3), dtype=np.uint8)
    assert np.array_equal(rotate_image(rotate_image(image, 180), 180), image)


def test_270_is_inverse_of_90():
    image = np.random.default_rng(0).integers(0, 255, (10, 20, 3), dtype=np.uint8)
    assert np.array_equal(rotate_image(rotate_image(image, 90), 270), image)
