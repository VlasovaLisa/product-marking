import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR
from src.split import assign_split, select_tune_label_set

SEED = 42
TEST_FRACTION = 0.2
TUNE_LABEL_PER_FORMAT = {"В": 150, "V": 75, "Z": 75}
SPLIT_PATH = DATA_DIR / "split.csv"


def main() -> None:
    if SPLIT_PATH.exists():
        sys.exit(f"{SPLIT_PATH} already exists: the split is frozen and must not be regenerated")
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    metadata = metadata[~metadata.is_corrupted].reset_index(drop=True)

    metadata["split"] = assign_split(metadata, TEST_FRACTION, SEED)
    selected = select_tune_label_set(metadata, metadata["split"], TUNE_LABEL_PER_FORMAT, SEED)
    metadata["in_label_set"] = selected | (metadata["split"] == "test")
    metadata[["filename", "split", "in_label_set"]].to_csv(SPLIT_PATH, index=False, encoding="utf-8")

    print(pd.crosstab(metadata.marking_format, metadata.split, margins=True))
    print(f"label set: {int(metadata.in_label_set.sum())} images")
    shared = metadata.groupby("group_key").split.nunique()
    print(f"groups in both splits: {int((shared > 1).sum())}")


if __name__ == "__main__":
    main()
