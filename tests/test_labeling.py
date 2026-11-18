import pandas as pd
import pytest

from src.labeling import LabelStore, build_queue


def test_save_and_reload(tmp_path):
    path = tmp_path / "labels.csv"
    LabelStore(path).save("a.jpg", "ММК 10Г2ФБЮ\n 2023-В303529 ", "readable", "")
    saved = LabelStore(path).get("a.jpg")
    assert saved["reference_text"] == "ММК 10Г2ФБЮ\n2023-В303529"
    assert saved["readability"] == "readable"


def test_unreadable_clears_text(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.save("a.jpg", "junk", "unreadable", "worn out")
    assert store.get("a.jpg")["reference_text"] == ""


def test_text_required_unless_unreadable(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    with pytest.raises(ValueError):
        store.save("a.jpg", "  ", "partial", "")


def test_unknown_readability_rejected(tmp_path):
    with pytest.raises(ValueError):
        LabelStore(tmp_path / "labels.csv").save("a.jpg", "x", "maybe", "")


def test_resave_overwrites(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.save("a.jpg", "one", "readable", "")
    store.save("a.jpg", "two", "readable", "")
    assert len(LabelStore(tmp_path / "labels.csv").labels) == 1
    assert store.get("a.jpg")["reference_text"] == "two"


def test_queue_puts_test_first_and_skips_unselected():
    split = pd.DataFrame(
        {
            "filename": ["t1", "t2", "u1", "u2", "u3"],
            "split": ["test", "test", "tune", "tune", "tune"],
            "in_label_set": [True, True, True, False, True],
        }
    )
    queue = build_queue(split, seed=1)
    assert set(queue[:2]) == {"t1", "t2"}
    assert set(queue[2:]) == {"u1", "u3"}
