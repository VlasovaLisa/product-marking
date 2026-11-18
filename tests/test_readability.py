import pandas as pd
import pytest

from src.readability import ReadabilityStore, build_test_queue


def test_save_and_reload(tmp_path):
    path = tmp_path / "readability.csv"
    store = ReadabilityStore(path)
    store.save("a.jpg", "unreadable")
    store.save("b.jpg", "readable")
    reloaded = ReadabilityStore(path)
    assert reloaded.get("a.jpg") == "unreadable"
    assert reloaded.unreadable() == {"a.jpg"}


def test_changing_a_decision_overwrites_it(tmp_path):
    store = ReadabilityStore(tmp_path / "readability.csv")
    store.save("a.jpg", "unreadable")
    store.save("a.jpg", "partial")
    assert store.unreadable() == set()


def test_unknown_value_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ReadabilityStore(tmp_path / "readability.csv").save("a.jpg", "maybe")


def test_queue_contains_only_test_images():
    split = pd.DataFrame({"filename": ["t1", "t2", "u1"], "split": ["test", "test", "tune"]})
    assert sorted(build_test_queue(split, seed=1)) == ["t1", "t2"]


def test_without_unreadable_drops_flagged_rows():
    from src.readability import without_unreadable

    frame = pd.DataFrame({"filename": ["a", "b", "c"], "x": [1, 2, 3]})
    assert list(without_unreadable(frame, {"b"}).filename) == ["a", "c"]
