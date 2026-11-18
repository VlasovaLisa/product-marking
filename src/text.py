import re

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Single comparison form for reference and OCR text: upper case, one space between tokens, lines joined."""
    return _WHITESPACE.sub(" ", text.upper()).strip()


# Latin letters drawn identically to Cyrillic ones: a photo cannot tell them apart, so they must not count as errors.
_LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")


def normalize_field(text: str) -> str:
    """Form for filename-derived reference fields: upper case, letters and digits only (separators on the plate vary),
    look-alike Latin letters folded to Cyrillic."""
    return "".join(char for char in text.upper() if char.isalnum()).translate(_LATIN_TO_CYRILLIC)
