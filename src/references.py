from src.text import normalize_field

# Checked against OCR output on the tuning split: the plate of V and Z sheets carries the heat code without the year.
YEAR_PRINTED_ON_PLATE = {"В": True, "V": False, "Z": False}


def reference_heat(heat: str, marking_format: str) -> str:
    year, code = heat.split("-", 1)
    return normalize_field(heat if YEAR_PRINTED_ON_PLATE.get(marking_format, True) else code)
