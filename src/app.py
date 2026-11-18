import threading
from typing import Callable

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.engines import build_engine
from src.pipeline import Pipeline, PipelineConfig, RecognitionResult

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOW_CONFIDENCE = 0.7

MESSAGE_NOT_RECOGNIZED = "Маркировка не распознана"
MESSAGE_PARTIAL = "Распознано не полностью: проверьте текст и исправьте его вручную."
MESSAGE_NO_FILE = "Файл не выбран."
MESSAGE_BAD_FORMAT = "Неподдерживаемый формат файла. Допустимы JPG, JPEG и PNG."
MESSAGE_CORRUPTED = "Файл повреждён или не является изображением."
MESSAGE_TOO_LARGE = "Файл слишком большой (максимум 10 МБ)."

Recognizer = Callable[[np.ndarray], RecognitionResult]


def classify(result: RecognitionResult) -> str:
    if not any(char.isalnum() for char in result.text):
        return "not_recognized"
    return "partial" if result.confidence < LOW_CONFIDENCE else "ok"


def _error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def create_app(recognize: Recognizer) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    recognition_lock = threading.Lock()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/recognize")
    def recognize_endpoint():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return _error(MESSAGE_NO_FILE, 400)
        extension = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            return _error(MESSAGE_BAD_FORMAT, 400)
        image = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return _error(MESSAGE_CORRUPTED, 400)

        with recognition_lock:
            result = recognize(image)
        status = classify(result)
        message = {"ok": "", "partial": MESSAGE_PARTIAL, "not_recognized": MESSAGE_NOT_RECOGNIZED}[status]
        return jsonify(
            {
                "status": status,
                "message": message,
                "text": "" if status == "not_recognized" else result.text,
                "angle": result.angle,
                "elapsed_seconds": round(result.elapsed_seconds, 1),
            }
        )

    @app.errorhandler(413)
    def too_large(_):
        return _error(MESSAGE_TOO_LARGE, 413)

    return app


def main() -> None:
    pipeline = Pipeline(build_engine("paddle"), PipelineConfig())
    create_app(pipeline.recognize).run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
