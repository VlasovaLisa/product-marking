import sys
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, Response, abort, redirect, render_template_string, request, url_for

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, IMAGES_DIR
from src.readability import ReadabilityStore, build_test_queue
from src.rotation import ROTATION_CODES

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>{{ position + 1 }}/{{ total }}</title>
<style>
body{font-family:system-ui;margin:10px;background:#222;color:#eee}
.viewer{overflow:auto;max-height:76vh;border:1px solid #555}
button,a.btn{font-size:18px;padding:8px 16px;margin-right:8px}
.meta{color:#aaa;margin-bottom:6px} .big{font-size:22px;color:#fff}
.hint{color:#999;margin-top:8px}
</style></head><body>
<div class="meta">{{ position + 1 }} / {{ total }} · размечено {{ done }} · сейчас: {{ current or '—' }}</div>
<div class="big">Ищем на листе: плавка <b>{{ heat }}</b>, лист <b>{{ sheet }}</b></div>
<div class="viewer"><img src="{{ url_for('image', position=position, rot=rot) }}"></div>
<form method="post" action="{{ url_for('item', position=position, rot=rot) }}" style="margin-top:8px">
{% for value, label, key in options %}
<button name="readability" value="{{ value }}" id="b{{ key }}">{{ key }} · {{ label }}</button>
{% endfor %}
{% for angle in rotations %}<a class="btn" href="{{ url_for('item', position=position, rot=angle) }}">{{ angle }}°</a>{% endfor %}
{% if position > 0 %}<a class="btn" href="{{ url_for('item', position=position - 1) }}">← назад</a>{% endif %}
</form>
<div class="hint">Клавиши 1 / 2 / 3. Поворот картинки только для просмотра.</div>
<script>
document.addEventListener('keydown', e => { const b = document.getElementById('b' + e.key); if (b) b.click(); });
</script></body></html>"""

OPTIONS = (
    ("readable", "читается: номер плавки виден целиком", "1"),
    ("partial", "частично: видна только часть", "2"),
    ("unreadable", "не читается: номера плавки не видно", "3"),
)


def create_app(images_dir: Path, data_dir: Path, seed: int = 42) -> Flask:
    app = Flask(__name__)
    split = pd.read_csv(data_dir / "split.csv")
    metadata = pd.read_csv(data_dir / "metadata.csv", dtype=str).set_index("filename")
    queue = build_test_queue(split, seed)
    store = ReadabilityStore(data_dir / "readability.csv")

    def first_unmarked(start: int) -> int:
        for position in range(start, len(queue)):
            if store.get(queue[position]) is None:
                return position
        return len(queue) - 1

    @app.get("/")
    def index():
        return redirect(url_for("item", position=first_unmarked(0)))

    @app.route("/item/<int:position>", methods=["GET", "POST"])
    def item(position: int):
        if not 0 <= position < len(queue):
            abort(404)
        filename = queue[position]
        rot = request.args.get("rot", 0, type=int)
        if request.method == "POST":
            try:
                store.save(filename, request.form.get("readability", ""))
            except ValueError:
                abort(400)
            return redirect(url_for("item", position=first_unmarked(position + 1)))
        row = metadata.loc[filename]
        return render_template_string(
            PAGE, position=position, total=len(queue), done=len(store.values), current=store.get(filename),
            heat=row.heat, sheet=row.sheet, rot=rot, options=OPTIONS, rotations=[0, *ROTATION_CODES],
        )

    @app.get("/image/<int:position>")
    def image(position: int):
        if not 0 <= position < len(queue):
            abort(404)
        picture = cv2.imread(str(images_dir / queue[position]))
        code = ROTATION_CODES.get(request.args.get("rot", 0, type=int))
        if code is not None:
            picture = cv2.rotate(picture, code)
        _, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(encoded.tobytes(), mimetype="image/jpeg")

    return app


if __name__ == "__main__":
    create_app(IMAGES_DIR, DATA_DIR).run(host="127.0.0.1", port=5002)
