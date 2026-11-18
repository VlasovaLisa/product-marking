import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, IMAGES_DIR
from src.rotation import ROTATION_CODES, rotate_image

PER_FORMAT = 30
SEED = 42
ROTATED_DIR = DATA_DIR / "rotated"
MANIFEST_PATH = DATA_DIR / "rotated_manifest.csv"


def main() -> None:
    split = pd.read_csv(DATA_DIR / "split.csv")
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")[["filename", "marking_format"]]
    candidates = split[split.in_label_set].merge(metadata, on="filename")

    rows = []
    for split_name in ("test", "tune"):
        in_split = candidates[candidates.split == split_name]
        for _, group in in_split.groupby("marking_format"):
            for source in group.sample(n=min(PER_FORMAT, len(group)), random_state=SEED).filename:
                image = cv2.imread(str(IMAGES_DIR / source))
                for angle in ROTATION_CODES:
                    target_dir = ROTATED_DIR / split_name / str(angle)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target_dir / source), rotate_image(image, angle))
                    rows.append(
                        {
                            "rotated_path": f"{split_name}/{angle}/{source}",
                            "source_filename": source,
                            "angle": angle,
                            "split": split_name,
                        }
                    )
    pd.DataFrame(rows).to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    manifest = pd.DataFrame(rows)
    print(manifest.groupby(["split", "angle"]).size())


if __name__ == "__main__":
    main()
