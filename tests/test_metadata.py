from src.metadata import parse_filename


def test_parses_mmk_filename():
    meta = parse_filename("Труба №18985 Плавка №2023-В303529 Лист №191201002.jpg")
    assert meta.image_id == "18985"
    assert meta.heat == "2023-В303529"
    assert meta.sheet == "191201002"
    assert meta.heat_prefix == "В"
    assert meta.manufacturer == "ММК"
    assert meta.group_key == "2023-В303529|191201002"


def test_latin_prefixes_map_to_ural_steel():
    for prefix in ("V", "Z"):
        meta = parse_filename(f"Труба №20221 Плавка №2023-{prefix}30587 Лист №2.jpg")
        assert meta.manufacturer == "Уральская Сталь"
        assert meta.marking_format == prefix


def test_cyrillic_and_latin_prefixes_are_distinct():
    cyrillic = parse_filename("Труба №1 Плавка №2023-В1 Лист №1.jpg")
    latin = parse_filename("Труба №2 Плавка №2023-V1 Лист №1.jpg")
    assert cyrillic.marking_format != latin.marking_format


def test_unparseable_filename_returns_none():
    assert parse_filename("random.jpg") is None


def test_unknown_prefix_is_marked_unknown():
    meta = parse_filename("Труба №1 Плавка №2023-Q1 Лист №1.jpg")
    assert meta.manufacturer == "unknown"
