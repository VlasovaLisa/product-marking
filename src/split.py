import random
from typing import Dict

import pandas as pd


def assign_split(metadata: pd.DataFrame, test_fraction: float, seed: int) -> pd.Series:
    """Group-level split stratified by marking_format: all photos of one sheet land in the same split."""
    split = pd.Series("tune", index=metadata.index)
    for _, stratum in metadata.groupby("marking_format"):
        group_sizes = stratum.groupby("group_key").size().sort_index()
        groups = list(group_sizes.index)
        random.Random(seed).shuffle(groups)
        target = round(len(stratum) * test_fraction)
        test_groups, taken = set(), 0
        for group in groups:
            if taken >= target:
                break
            test_groups.add(group)
            taken += group_sizes[group]
        split[stratum.index[stratum.group_key.isin(test_groups)]] = "test"
    return split


def select_tune_label_set(
    metadata: pd.DataFrame, split: pd.Series, per_format: Dict[str, int], seed: int
) -> pd.Series:
    selected = pd.Series(False, index=metadata.index)
    tune = metadata[split == "tune"]
    for marking_format, count in per_format.items():
        candidates = tune[tune.marking_format == marking_format]
        selected[candidates.sample(n=count, random_state=seed).index] = True
    return selected
