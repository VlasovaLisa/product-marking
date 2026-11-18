"""Collects every number used in the project report into report/numbers.json.

Run with the project environment: .venv/bin/python report/collect_numbers.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import cer, hit_rate, noise_floor_heat_cer, score_image  # noqa: E402
from src.postprocessing import postprocess_lines  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUTPUT = Path(__file__).resolve().parent / "numbers.json"

EXPERIMENTS = [
    "00-baseline-raw",
    "02-mask-only",
    "01-mask-rot4",
    "03-mask-mag2",
    "04-mask-crop",
    "05-crop-rot4-allow",
    "06-paddle-mask",
    "07-paddle-rot4",
    "08-vz-clahe",
    "09-vz-dilate3",
    "10-vz-close5",
    "11-mmk-dilate3",
    "13-z-refine-crop",
]


def load_report(name: str, references: pd.DataFrame, postprocess: bool = False) -> pd.DataFrame:
    report = pd.read_csv(REPORTS / f"{name}.csv", dtype=str).fillna("")
    rows = []
    for item in report.itertuples():
        lines = item.ocr_text.split(" | ") if item.ocr_text else []
        if postprocess:
            lines = list(postprocess_lines(lines))
        reference = references.loc[item.filename]
        score = score_image(reference.ref_heat, reference.ref_sheet, lines)
        rows.append(
            {
                "filename": item.filename,
                "fmt": item.marking_format,
                "lines": lines,
                "ref_heat": reference.ref_heat,
                "he": score.heat_edits,
                "hl": score.heat_length,
                "se": score.sheet_edits,
                "sl": score.sheet_length,
                "e": score.edits,
                "l": score.length,
                "t": float(item.elapsed),
                "rot": getattr(item, "applied_rotation", "0"),
            }
        )
    return pd.DataFrame(rows)


def stats(frame: pd.DataFrame, noise: bool = False) -> dict:
    result = {
        "n": int(len(frame)),
        "cer": cer(frame.e.sum(), frame.l.sum()),
        "cer_heat": cer(frame.he.sum(), frame.hl.sum()),
        "cer_sheet": cer(frame.se.sum(), frame.sl.sum()) if frame.sl.sum() else None,
        "hit2": hit_rate(frame.he.tolist()),
        "empty": float((frame.lines.map(len) == 0).mean()),
        "t_mean": float(frame.t.mean()),
        "t_max": float(frame.t.max()),
    }
    if noise:
        result["noise"] = noise_floor_heat_cer(frame.ref_heat.tolist(), frame.lines.tolist())
    return result


def by_format(frame: pd.DataFrame, noise: bool = False) -> dict:
    result = {fmt: stats(group, noise) for fmt, group in frame.groupby("fmt")}
    ural = frame[frame.fmt != "В"]
    if len(ural) and ural.fmt.nunique() > 1:
        result["URAL"] = stats(ural, noise)
    result["ALL"] = stats(frame, noise)
    return result


def main() -> None:
    metadata = pd.read_csv(DATA / "metadata.csv", dtype=str)
    split = pd.read_csv(DATA / "split.csv")
    references = pd.read_csv(DATA / "references.csv", dtype=str).set_index("filename")
    readability = pd.read_csv(DATA / "readability.csv")
    merged = metadata.merge(split, on="filename")
    test = merged[merged.split == "test"]
    tune = merged[merged.split == "tune"]
    shared_heats = set(test.heat) & set(tune.heat)

    numbers = {
        "dataset": {
            **{
                fmt: {"photos": len(g), "heats": g.heat.nunique(), "sheets": g.group_key.nunique()}
                for fmt, g in merged.groupby("marking_format")
            },
            "ALL": {"photos": len(merged), "heats": merged.heat.nunique(), "sheets": merged.group_key.nunique()},
        },
        "split": {fmt: {s: int((g.split == s).sum()) for s in ("tune", "test")} for fmt, g in merged.groupby("marking_format")},
        "test_heats": {
            "MMK": test[test.marking_format == "В"].heat.nunique(),
            "URAL": test[test.marking_format != "В"].heat.nunique(),
            "ALL": test.heat.nunique(),
        },
        "heat_overlap": {
            "test_heats_also_in_tune": len(shared_heats),
            "test_frames_with_shared_heat": int(test.heat.isin(shared_heats).sum()),
        },
        "readability": {
            fmt: g.readability.value_counts().to_dict()
            for fmt, g in test.merge(readability, on="filename").groupby("marking_format")
        },
    }

    experiments = {name: by_format(load_report(name, references), noise=True) for name in EXPERIMENTS}
    experiments["07-paddle-rot4+pp"] = by_format(load_report("07-paddle-rot4", references, postprocess=True), noise=True)
    final_tune = pd.concat([load_report("11-mmk-dilate3", references), load_report("09-vz-dilate3", references)])
    experiments["final-tune"] = by_format(final_tune, noise=True)
    numbers["experiments"] = experiments

    final = load_report("final-test", references).merge(readability, on="filename")
    numbers["test"] = by_format(final, noise=True)
    numbers["test_no_unreadable"] = by_format(final[final.readability != "unreadable"], noise=True)
    numbers["test_readable"] = by_format(final[final.readability == "readable"], noise=True)
    numbers["test_by_readability"] = {k: stats(g, noise=True) for k, g in final.groupby("readability")}
    per_image = final.e / final.l
    numbers["test_frames_cer_le_30"] = {"count": int((per_image <= 0.30).sum()), "of": len(final)}
    numbers["test_time"] = {
        "mean": float(final.t.mean()),
        "median": float(final.t.median()),
        "p90": float(final.t.quantile(0.9)),
        "p95": float(final.t.quantile(0.95)),
        "p99": float(final.t.quantile(0.99)),
        "max": float(final.t.max()),
        "over60": int((final.t > 60).sum()),
    }

    rotated = load_report("final-rotated", references).merge(readability, on="filename")
    numbers["rotated"] = {str(angle): stats(g) for angle, g in rotated.groupby("rot")}
    numbers["rotated_no_unreadable"] = {
        str(angle): stats(g) for angle, g in rotated[rotated.readability != "unreadable"].groupby("rot")
    }
    numbers["rotated_all"] = stats(rotated)
    numbers["rotated_base"] = stats(final[final.filename.isin(rotated.filename)])
    numbers["rotated_sources_readability"] = rotated.drop_duplicates("filename").readability.value_counts().to_dict()

    estimate = pd.read_csv(DATA / "readability_estimate_tune.csv")
    numbers["estimate_tune"] = estimate.groupby(["fmt", "label"]).size().unstack(fill_value=0).to_dict(orient="index")

    OUTPUT.write_text(json.dumps(numbers, ensure_ascii=False, indent=1, default=lambda v: v.item()), encoding="utf-8")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
