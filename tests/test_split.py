import pandas as pd

from src.split import assign_split, select_tune_label_set


def make_metadata() -> pd.DataFrame:
    rows = []
    for marking_format, sheets in (("В", 60), ("V", 30)):
        for sheet in range(sheets):
            copies = 2 if sheet % 10 == 0 else 1
            for copy in range(copies):
                rows.append(
                    {
                        "filename": f"{marking_format}{sheet}_{copy}.jpg",
                        "marking_format": marking_format,
                        "group_key": f"{marking_format}|{sheet}",
                    }
                )
    return pd.DataFrame(rows)


def test_no_group_in_both_splits():
    metadata = make_metadata()
    split = assign_split(metadata, 0.2, seed=1)
    per_group = metadata.assign(split=split).groupby("group_key").split.nunique()
    assert (per_group == 1).all()


def test_every_format_present_in_both_splits_with_about_20_percent():
    metadata = make_metadata()
    split = assign_split(metadata, 0.2, seed=1)
    for _, stratum in metadata.assign(split=split).groupby("marking_format"):
        test_share = (stratum.split == "test").mean()
        assert 0.15 <= test_share <= 0.27


def test_split_is_reproducible():
    metadata = make_metadata()
    assert assign_split(metadata, 0.2, seed=5).equals(assign_split(metadata, 0.2, seed=5))


def test_label_set_comes_only_from_tune():
    metadata = make_metadata()
    split = assign_split(metadata, 0.2, seed=1)
    selected = select_tune_label_set(metadata, split, {"В": 10, "V": 5}, seed=1)
    assert selected.sum() == 15
    assert (split[selected] == "tune").all()
