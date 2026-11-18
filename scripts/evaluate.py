import argparse
import ast
import statistics
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, IMAGES_DIR, PROJECT_ROOT
from src.engines import build_engine
from src.metrics import cer, hit_rate, noise_floor_heat_cer, score_image
from src.pipeline import Pipeline, PipelineConfig
from src.postprocessing import postprocess_lines
from src.readability import ReadabilityStore, without_unreadable

REPORTS_DIR = PROJECT_ROOT / "reports"
SEED = 42
ROTATED_DIR = DATA_DIR / "rotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the OCR pipeline on filename-derived references")
    parser.add_argument("--rescore", help="only print the summary of an existing reports/<name>.csv")
    parser.add_argument("--name", default="unnamed", help="experiment name, used for the output file")
    parser.add_argument("--split", choices=("tune", "test"), default="tune")
    parser.add_argument("--final", action="store_true", help="required to touch the test split (stage 5 only)")
    parser.add_argument("--per-format", type=int, default=40, help="images per marking format (same sample every run)")
    parser.add_argument("--rotations", default="0,90,180,270", help="comma-separated angles to try")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--no-mask", action="store_true", help="do not blank the camera overlay")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false", help="skip text post-processing")
    parser.add_argument("--dots", choices=("none", "dilate3", "close5"), default="dilate3", help="join dots of stamped marking")
    parser.add_argument("--formats", default="В,V,Z", help="comma-separated marking formats to evaluate")
    parser.add_argument("--refine-side", type=int, default=0, help="second pass: re-read the found text region enlarged to this many pixels")
    parser.add_argument("--exclude-unreadable", action="store_true", help="drop images marked unreadable in data/readability.csv")
    parser.add_argument("--rotated", action="store_true", help="evaluate the rotated copies from data/rotated_manifest.csv")
    parser.add_argument("--engine", choices=("easyocr", "paddle"), default="paddle")
    parser.add_argument("--crop", action="store_true", help="crop to detected text and upscale before recognition")
    parser.add_argument("--crop-side", type=int, default=1600)
    parser.add_argument("--readtext", nargs="*", default=[], help="extra EasyOCR options as key=value, e.g. mag_ratio=2")
    parser.add_argument("--allowlist", default=None, help="restrict EasyOCR to these characters")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    options = {"allowlist": args.allowlist} if args.allowlist else {}
    for item in args.readtext:
        key, value = item.split("=", 1)
        options[key] = ast.literal_eval(value)
    return PipelineConfig(
        mask_overlay=not args.no_mask,
        clahe=args.clahe,
        dots="" if args.dots == "none" else args.dots,
        rotations=tuple(int(angle) for angle in args.rotations.split(",")),
        min_confidence=args.min_confidence,
        crop_to_text=args.crop,
        refine_side=args.refine_side,
        postprocess=args.postprocess,
        crop_target_side=args.crop_side,
        readtext_options=options,
    )


def load_sample(split: str, per_format: int, formats: list) -> pd.DataFrame:
    references = pd.read_csv(DATA_DIR / "references.csv", dtype=str)
    pool = references[(references.split == split) & references.marking_format.isin(formats)]
    parts = [
        group.sample(n=min(per_format, len(group)), random_state=SEED)
        for _, group in pool.groupby("marking_format")
    ]
    return pd.concat(parts).reset_index(drop=True)


def load_rotated_sample(split: str, per_format: int, formats: list) -> pd.DataFrame:
    manifest = pd.read_csv(DATA_DIR / "rotated_manifest.csv", dtype=str)
    references = pd.read_csv(DATA_DIR / "references.csv", dtype=str)
    sources = references[references.filename.isin(manifest.source_filename) & references.marking_format.isin(formats)]
    sources = sources[sources.split == split]
    chosen = pd.concat(
        [group.sample(n=min(per_format, len(group)), random_state=SEED) for _, group in sources.groupby("marking_format")]
    )
    rows = manifest[manifest.source_filename.isin(chosen.filename)].merge(
        chosen, left_on="source_filename", right_on="filename"
    )
    rows["image_path"] = [str(ROTATED_DIR / path) for path in rows.rotated_path]
    rows["applied_rotation"] = rows.angle
    return rows.sort_values(["marking_format", "filename", "applied_rotation"]).reset_index(drop=True)


def apply_readability_filter(results: pd.DataFrame, exclude: bool) -> pd.DataFrame:
    if not exclude:
        return results
    unreadable = ReadabilityStore(DATA_DIR / "readability.csv").unreadable()
    kept = without_unreadable(results, unreadable)
    print(f"excluded as unreadable: {len(results) - len(kept)} of {len(results)}")
    return kept


def summarize(results: pd.DataFrame) -> None:
    def row(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(group),
                "CER_all": cer(group.edits.sum(), group.length.sum()),
                "CER_heat": cer(group.heat_edits.sum(), group.heat_length.sum()),
                "CER_sheet": cer(group.sheet_edits.sum(), group.sheet_length.sum()),
                "heat_hit2": hit_rate(group.heat_edits.tolist()),
                "heat_noise": noise_floor_heat_cer(
                    group.ref_heat.tolist(), [text.split(" | ") if text else [] for text in group.ocr_text]
                ),
                "empty": (group.ocr_text == "").mean(),
                "t_mean": group.elapsed.mean(),
                "t_max": group.elapsed.max(),
            }
        )

    by_format = pd.DataFrame({name: row(group) for name, group in results.groupby("marking_format")}).T
    overall = row(results).to_frame("ALL").T
    print(pd.concat([by_format, overall]).round(3).to_string())
    print(f"median time: {statistics.median(results.elapsed):.2f}s")
    if "applied_rotation" in results and results.applied_rotation.nunique() > 1:
        print("by applied rotation:")
        print(
            results.groupby("applied_rotation")
            .apply(
                lambda group: pd.Series(
                    {
                        "n": len(group),
                        "CER_heat": cer(group.heat_edits.sum(), group.heat_length.sum()),
                        "heat_hit2": hit_rate(group.heat_edits.tolist()),
                    }
                ),
                include_groups=False,
            )
            .round(3)
            .to_string()
        )


def rescore(name: str, postprocess: bool = False, exclude_unreadable: bool = False) -> None:
    """Recompute scores from the saved OCR text against the current references.csv."""
    report = pd.read_csv(REPORTS_DIR / f"{name}.csv", dtype=str).fillna("")
    references = pd.read_csv(DATA_DIR / "references.csv", dtype=str)[["filename", "ref_heat", "ref_sheet"]]
    report = report[["filename", "marking_format", "ocr_text", "angle", "elapsed"]].merge(references, on="filename")
    rows = []
    for item in report.itertuples():
        lines = item.ocr_text.split(" | ") if item.ocr_text else []
        score = score_image(item.ref_heat, item.ref_sheet, postprocess_lines(lines) if postprocess else lines)
        rows.append(
            {
                "filename": item.filename, "marking_format": item.marking_format, "ref_heat": item.ref_heat,
                "ocr_text": item.ocr_text, "elapsed": float(item.elapsed),
                "heat_edits": score.heat_edits, "heat_length": score.heat_length,
                "sheet_edits": score.sheet_edits, "sheet_length": score.sheet_length,
                "edits": score.edits, "length": score.length,
            }
        )
    summarize(apply_readability_filter(pd.DataFrame(rows), exclude_unreadable))


def main() -> None:
    args = parse_args()
    if args.rescore:
        rescore(args.rescore, args.postprocess, args.exclude_unreadable)
        return
    if args.split == "test" and not args.final:
        sys.exit("test split is closed until stage 5: pass --final only for the final evaluation")

    config = build_config(args)
    pipeline = Pipeline(build_engine(args.engine), config)
    formats = args.formats.split(",")
    sample = (
        load_rotated_sample(args.split, args.per_format, formats)
        if args.rotated
        else load_sample(args.split, args.per_format, formats)
    )

    rows = []
    for index, item in sample.iterrows():
        path = item.image_path if args.rotated else str(IMAGES_DIR / item.filename)
        image = cv2.imread(path)
        result = pipeline.recognize(image)
        score = score_image(item.ref_heat, item.ref_sheet, result.lines)
        rows.append(
            {
                "filename": item.filename,
                "applied_rotation": item.applied_rotation if args.rotated else "0",
                "marking_format": item.marking_format,
                "ref_heat": item.ref_heat,
                "ref_sheet": item.ref_sheet,
                "ocr_text": " | ".join(result.lines),
                "angle": result.angle,
                "elapsed": result.elapsed_seconds,
                "heat_edits": score.heat_edits,
                "heat_length": score.heat_length,
                "sheet_edits": score.sheet_edits,
                "sheet_length": score.sheet_length,
                "edits": score.edits,
                "length": score.length,
            }
        )
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{len(sample)}", flush=True)

    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / f"{args.name}.csv", index=False, encoding="utf-8")
    print(f"config: {config}")
    summarize(apply_readability_filter(results, args.exclude_unreadable))


if __name__ == "__main__":
    main()
