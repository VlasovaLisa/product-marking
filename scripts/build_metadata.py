import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, IMAGES_DIR
from src.metadata import parse_filename

METADATA_PATH = DATA_DIR / "metadata.csv"
AUDIT_PATH = DATA_DIR / "audit.csv"


def main() -> None:
    filenames = sorted(path.name for path in IMAGES_DIR.glob("*.jpg"))
    parsed = [parse_filename(name) for name in filenames]
    unparsed = [name for name, meta in zip(filenames, parsed) if meta is None]
    for name in unparsed:
        print(f"UNPARSED: {name}")

    metadata = pd.DataFrame([meta.__dict__ for meta in parsed if meta is not None])
    audit = pd.read_csv(AUDIT_PATH)[["filename", "is_corrupted"]]
    metadata = metadata.merge(audit, on="filename", how="left")
    metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8")

    print(f"rows: {len(metadata)}, unparsed: {len(unparsed)}")
    print(metadata.groupby(["manufacturer", "marking_format"]).size())
    duplicates = metadata[metadata.duplicated("group_key", keep=False)].sort_values("group_key")
    print(f"group_key duplicates (rows): {len(duplicates)}")
    print(duplicates[["filename", "group_key"]].to_string(index=False))


if __name__ == "__main__":
    main()
