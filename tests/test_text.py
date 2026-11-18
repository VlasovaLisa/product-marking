from src.text import normalize


def test_upper_case_and_trim():
    assert normalize("  ммк 10г2фбю ") == "ММК 10Г2ФБЮ"


def test_collapses_whitespace_and_joins_lines():
    assert normalize("2023-В303529   100\n12.00*3134*12040") == "2023-В303529 100 12.00*3134*12040"


def test_keeps_cyrillic_and_latin_as_written():
    assert normalize("В") != normalize("B")


def test_empty_string():
    assert normalize("") == ""


def test_normalize_field_drops_separators():
    from src.text import normalize_field

    assert normalize_field("2023-В105741") == "2023В105741"
    assert normalize_field("19080-3000") == "190803000"
    assert normalize_field("12.00*3134") == "1200" + "3134"


def test_normalize_field_folds_latin_lookalikes_to_cyrillic():
    from src.text import normalize_field

    assert normalize_field("2023-B105741") == normalize_field("2023-В105741")
    assert normalize_field("V30336") == "V30336"
