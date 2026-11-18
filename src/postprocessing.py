import re
from typing import Sequence, Tuple

# A heat number is year + letter + digits. OCR often reads the Cyrillic letter В right after the year as 8 or Latin B.
_HEAT_LETTER = re.compile(r"(20\d{2})([-.\s]?)[8B](?=\d{5,6}(?!\d))")


def fix_heat_letter(line: str) -> str:
    return _HEAT_LETTER.sub(r"\1\2В", line)


def postprocess_lines(lines: Sequence[str]) -> Tuple[str, ...]:
    return tuple(fix_heat_letter(line) for line in lines)
