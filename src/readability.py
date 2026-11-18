import csv
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

READABILITY_COLUMNS = ["filename", "readability"]
READABILITY_VALUES = ("readable", "partial", "unreadable")


class ReadabilityStore:
    def __init__(self, path: Path):
        self.path = path
        self.values: Dict[str, str] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as file:
                self.values = {row["filename"]: row["readability"] for row in csv.DictReader(file)}

    def get(self, filename: str):
        return self.values.get(filename)

    def save(self, filename: str, readability: str) -> None:
        if readability not in READABILITY_VALUES:
            raise ValueError(f"unknown readability: {readability}")
        self.values[filename] = readability
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(READABILITY_COLUMNS)
            writer.writerows(self.values.items())
        os.replace(temporary, self.path)

    def unreadable(self) -> set:
        return {name for name, value in self.values.items() if value == "unreadable"}


def build_test_queue(split: pd.DataFrame, seed: int) -> List[str]:
    return list(split[split.split == "test"].sample(frac=1, random_state=seed).filename)


def without_unreadable(frame: pd.DataFrame, unreadable_names: set) -> pd.DataFrame:
    return frame[~frame.filename.isin(unreadable_names)].reset_index(drop=True)
