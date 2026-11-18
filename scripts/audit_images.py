import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, IMAGES_DIR

AUDIT_PATH = DATA_DIR / "audit.csv"


def audit_image(path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return {"filename": path.name, "is_corrupted": True, "width": 0, "height": 0}
    height, width = image.shape[:2]
    return {"filename": path.name, "is_corrupted": False, "width": width, "height": height}


def main() -> None:
    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    rows = [audit_image(path) for path in paths]
    with AUDIT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "is_corrupted", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)
    corrupted = [row["filename"] for row in rows if row["is_corrupted"]]
    sizes = sorted({(row["width"], row["height"]) for row in rows if not row["is_corrupted"]})
    print(f"files: {len(rows)}, corrupted: {len(corrupted)}, sizes: {sizes}")
    for name in corrupted:
        print(f"CORRUPTED: {name}")


if __name__ == "__main__":
    main()
