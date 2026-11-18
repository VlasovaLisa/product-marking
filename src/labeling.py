import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LABEL_COLUMNS = ["filename", "reference_text", "readability", "comment"]
READABILITY_VALUES = ("readable", "partial", "unreadable")


class LabelStore:
    def __init__(self, path: Path):
        self.path = path
        self.labels: Dict[str, dict] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    self.labels[row["filename"]] = row

    def get(self, filename: str) -> Optional[dict]:
        return self.labels.get(filename)

    def save(self, filename: str, reference_text: str, readability: str, comment: str) -> None:
        if readability not in READABILITY_VALUES:
            raise ValueError(f"unknown readability: {readability}")
        if readability != "unreadable" and not reference_text.strip():
            raise ValueError("reference text is required unless the image is unreadable")
        text = "" if readability == "unreadable" else "\n".join(
            line.strip() for line in reference_text.splitlines() if line.strip()
        )
        self.labels[filename] = {
            "filename": filename,
            "reference_text": text,
            "readability": readability,
            "comment": comment.strip(),
        }
        self._write()

    def _write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(self.labels.values())
        os.replace(temporary, self.path)


def build_queue(split: pd.DataFrame, seed: int) -> List[str]:
    """Test images first (they are the priority), each part shuffled so formats are mixed."""
    queue: List[str] = []
    for split_name in ("test", "tune"):
        part = split[(split.split == split_name) & split.in_label_set]
        queue.extend(part.sample(frac=1, random_state=seed).filename)
    return queue
