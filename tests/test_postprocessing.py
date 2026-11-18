from src.postprocessing import fix_heat_letter, postprocess_lines


def test_eight_after_year_becomes_cyrillic_ve():
    assert fix_heat_letter("2023-8208567") == "2023-В208567"


def test_latin_b_after_year_becomes_cyrillic_ve():
    assert fix_heat_letter("2023-B208567") == "2023-В208567"


def test_correct_heat_is_untouched():
    assert fix_heat_letter("2023-В208567") == "2023-В208567"


def test_other_numbers_are_untouched():
    for line in ("V30336", "12.00*3134*12040", "2023", "19050-3000", "2023-8"):
        assert fix_heat_letter(line) == line


def test_applies_to_every_line():
    assert postprocess_lines(["x", "2022 8105741"]) == ("x", "2022 В105741")
