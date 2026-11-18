import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR
from src.references import reference_heat
from src.text import normalize_field


def main() -> None:
    metadata = pd.read_csv(DATA_DIR / "metadata.csv", dtype=str)
    split = pd.read_csv(DATA_DIR / "split.csv")[["filename", "split"]]
    references = metadata[["filename", "marking_format", "heat", "sheet"]].copy()
    references["ref_heat"] = [
        reference_heat(heat, marking_format)
        for heat, marking_format in zip(references.heat, references.marking_format)
    ]
    references["ref_sheet"] = references.sheet.map(normalize_field)
    references = references.merge(split, on="filename")
    references[["filename", "split", "marking_format", "ref_heat", "ref_sheet"]].to_csv(
        DATA_DIR / "references.csv", index=False, encoding="utf-8"
    )
    print(references.groupby(["split", "marking_format"]).agg(
        n=("filename", "size"),
        heat_len=("ref_heat", lambda s: s.str.len().median()),
        sheet_len=("ref_sheet", lambda s: s.str.len().median()),
    ))


if __name__ == "__main__":
    main()
