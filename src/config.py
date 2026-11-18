import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", PROJECT_ROOT / "images"))
DATA_DIR = PROJECT_ROOT / "data"
