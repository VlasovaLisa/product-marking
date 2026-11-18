import sys
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, Response, abort, redirect, render_template_string, request, url_for

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, IMAGES_DIR
from src.labeling import READABILITY_VALUES, LabelStore, build_queue

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Разметка {{ position + 1 }}/{{ total }}</title>
<style>
body{font-family:system-ui;margin:12px;background:#222;color:#eee}
.viewer{overflow:auto;max-height:78vh;border:1px solid #555}
textarea{width:520px;height:110px;font:18px monospace}
button,a.btn{font-size:16px;padding:6px 12px;margin-right:6px}
.meta{color:#aaa} .error{color:#f66} .row{margin:8px 0}
</style></head><body>
<div class="meta">{{ position + 1 }} / {{ total }} · размечено {{ done }} · {{ split }} · {{ manufacturer }} ({{ marking_format }}) · {{ filename }}</div>
<div class="row">Поворот картинки:
{% for angle in rotations %}<a class="btn" href="{{ url_for('item', position=position, rot=angle) }}">{{ angle }}°</a>{% endfor %}
(только для просмотра, на эталон не влияет)</div>
<div class="viewer"><img src="{{ url_for('image', position=position, rot=rot) }}"></div>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
<form method="post" action="{{ url_for('item', position=position, rot=rot) }}">
<div class="row">Строки маркировки сверху вниз (при правильной ориентации):<br>
<textarea name="reference_text" autofocus>{{ text }}</textarea></div>
<div class="row">{% for value in readability_values %}
<label><input type="radio" name="readability" value="{{ value }}" {{ 'checked' if value == readability }}> {{ value }}</label>&nbsp;
{% endfor %}</div>
<div class="row">Комментарий: <input name="comment" size="60" value="{{ comment }}"></div>
<div class="row"><button type="submit">Сохранить и дальше</button>
{% if position > 0 %}<a class="btn" href="{{ url_for('item', position=position - 1) }}">← назад</a>{% endif %}
{% if position + 1 < total %}<a class="btn" href="{{ url_for('item', position=position + 1) }}">пропустить →</a>{% endif %}
</div></form></body></html>"""


def create_app(images_dir: Path, data_dir: Path, seed: int = 42) -> Flask:
    app = Flask(__name__)
    split = pd.read_csv(data_dir / "split.csv")
    metadata = pd.read_csv(data_dir / "metadata.csv").set_index("filename")
    queue = build_queue(split, seed)
    split_by_file = split.set_index("filename").split
    store = LabelStore(data_dir / "labels.csv")

    def first_unlabeled(start: int) -> int:
        for position in range(start, len(queue)):
            if store.get(queue[position]) is None:
                return position
        return len(queue) - 1

    @app.get("/")
    def index():
        return redirect(url_for("item", position=first_unlabeled(0)))

    @app.route("/item/<int:position>", methods=["GET", "POST"])
    def item(position: int):
        if not 0 <= position < len(queue):
            abort(404)
        filename = queue[position]
        rot = request.args.get("rot", 0, type=int)
        error = None
        if request.method == "POST":
            try:
                store.save(
                    filename,
                    request.form["reference_text"],
                    request.form.get("readability", ""),
                    request.form.get("comment", ""),
                )
                return redirect(url_for("item", position=first_unlabeled(position + 1)))
            except ValueError as exc:
                error = str(exc)
        saved = store.get(filename) or {}
        row = metadata.loc[filename]
        return render_template_string(
            PAGE,
            position=position,
            total=len(queue),
            done=len(store.labels),
            split=split_by_file[filename],
            manufacturer=row.manufacturer,
            marking_format=row.marking_format,
            filename=filename,
            rotations=list(ROTATIONS),
            rot=rot,
            error=error,
            text=request.form.get("reference_text") if error else saved.get("reference_text", row.heat),
            readability=request.form.get("readability") if error else saved.get("readability", "readable"),
            comment=request.form.get("comment", "") if error else saved.get("comment", ""),
            readability_values=READABILITY_VALUES,
        )

    @app.get("/image/<int:position>")
    def image(position: int):
        if not 0 <= position < len(queue):
            abort(404)
        picture = cv2.imread(str(images_dir / queue[position]))
        code = ROTATIONS.get(request.args.get("rot", 0, type=int))
        if code is not None:
            picture = cv2.rotate(picture, code)
        _, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(encoded.tobytes(), mimetype="image/jpeg")

    return app


if __name__ == "__main__":
    create_app(IMAGES_DIR, DATA_DIR).run(host="127.0.0.1", port=5001)
