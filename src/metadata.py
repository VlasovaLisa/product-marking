import re
from dataclasses import dataclass
from typing import Optional

FILENAME_PATTERN = re.compile(
    r"^Труба №(?P<image_id>\d+) Плавка №(?P<year>\d{4})-(?P<heat_code>\S+) Лист №(?P<sheet>\S+)\.jpg$"
)
HEAT_PREFIX_PATTERN = re.compile(r"^(?P<prefix>\D+)")

MANUFACTURER_BY_PREFIX = {
    "В": "ММК",
    "V": "Уральская Сталь",
    "Z": "Уральская Сталь",
}


@dataclass(frozen=True)
class FileMetadata:
    image_id: str
    filename: str
    heat: str
    sheet: str
    heat_prefix: str
    manufacturer: str
    marking_format: str
    group_key: str


def parse_filename(filename: str) -> Optional[FileMetadata]:
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    heat = f"{match['year']}-{match['heat_code']}"
    prefix_match = HEAT_PREFIX_PATTERN.match(match["heat_code"])
    prefix = prefix_match["prefix"] if prefix_match else ""
    return FileMetadata(
        image_id=match["image_id"],
        filename=filename,
        heat=heat,
        sheet=match["sheet"],
        heat_prefix=prefix,
        manufacturer=MANUFACTURER_BY_PREFIX.get(prefix, "unknown"),
        marking_format=prefix,
        group_key=f"{heat}|{match['sheet']}",
    )
