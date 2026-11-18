import io

import cv2
import numpy as np
import pytest

from src.app import (
    MESSAGE_BAD_FORMAT,
    MESSAGE_CORRUPTED,
    MESSAGE_NO_FILE,
    MESSAGE_NOT_RECOGNIZED,
    classify,
    create_app,
)
from src.pipeline import RecognitionResult


def make_result(text: str, confidence: float = 0.9) -> RecognitionResult:
    return RecognitionResult(text=text, lines=tuple(text.splitlines()), confidence=confidence, angle=90, elapsed_seconds=1.234)


def png_bytes() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    assert ok
    return encoded.tobytes()


def post_file(client, name: str, content: bytes):
    return client.post("/recognize", data={"image": (io.BytesIO(content), name)}, content_type="multipart/form-data")


@pytest.fixture
def client():
    return create_app(lambda image: make_result("2023-В208567\n19120-1002")).test_client()


def test_index_page_has_upload_controls(client):
    page = client.get("/").get_data(as_text=True)
    assert 'type="file"' in page and "Распознать" in page and "<textarea" in page


def test_recognizes_png_and_returns_text(client):
    response = post_file(client, "sheet.png", png_bytes())
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ok" and body["text"].startswith("2023-В208567")
    assert body["angle"] == 90 and body["elapsed_seconds"] == 1.2


@pytest.mark.parametrize("name", ["sheet.JPG", "sheet.jpeg", "sheet.PNG"])
def test_accepts_allowed_extensions_case_insensitively(client, name):
    assert post_file(client, name, png_bytes()).status_code == 200


def test_rejects_unsupported_format(client):
    response = post_file(client, "sheet.gif", b"GIF89a")
    assert response.status_code == 400 and response.get_json()["error"] == MESSAGE_BAD_FORMAT


def test_rejects_corrupted_image(client):
    response = post_file(client, "sheet.jpg", b"not an image at all")
    assert response.status_code == 400 and response.get_json()["error"] == MESSAGE_CORRUPTED


def test_rejects_missing_file(client):
    response = client.post("/recognize", data={}, content_type="multipart/form-data")
    assert response.status_code == 400 and response.get_json()["error"] == MESSAGE_NO_FILE


def test_empty_recognition_reports_not_recognized():
    client = create_app(lambda image: make_result("", 0.0)).test_client()
    body = post_file(client, "sheet.png", png_bytes()).get_json()
    assert body["status"] == "not_recognized"
    assert body["message"] == MESSAGE_NOT_RECOGNIZED == "Маркировка не распознана"
    assert body["text"] == ""


def test_low_confidence_is_partial_but_text_is_still_returned():
    client = create_app(lambda image: make_result("2023-В2", 0.4)).test_client()
    body = post_file(client, "sheet.png", png_bytes()).get_json()
    assert body["status"] == "partial" and body["text"] == "2023-В2" and body["message"]


def test_punctuation_only_is_not_recognized():
    assert classify(make_result("- . *", 0.9)) == "not_recognized"
