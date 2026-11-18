from src.references import reference_heat


def test_mmk_heat_keeps_the_year():
    assert reference_heat("2023-В303529", "В") == "2023В303529"


def test_ural_steel_heat_drops_the_year():
    assert reference_heat("2023-V30336", "V") == "V30336"
    assert reference_heat("2022-Z26849", "Z") == "Z26849"


def test_unknown_format_keeps_the_year():
    assert reference_heat("2023-Q1", "Q") == "2023Q1"
