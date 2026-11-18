from src.metrics import cer, edit_distance_to_best_fragment, score_image


def test_exact_fragment_costs_nothing():
    assert edit_distance_to_best_fragment("ABC123", "XXABC123YY") == 0


def test_substitution_deletion_insertion_each_cost_one():
    assert edit_distance_to_best_fragment("ABC123", "ABX123") == 1
    assert edit_distance_to_best_fragment("ABC123", "ABC12") == 1
    assert edit_distance_to_best_fragment("ABC123", "ABC1X23") == 1


def test_empty_text_costs_full_reference_length():
    assert edit_distance_to_best_fragment("ABC123", "") == 6


def test_zero_vs_eight_confusion_is_an_error():
    assert edit_distance_to_best_fragment("190803000", "190803800") == 1


def test_short_sheet_is_excluded_from_the_score():
    score = score_image("V30336", "13", ["V30336", "13"])
    assert score.sheet_length == 0
    assert score.sheet_edits == 0
    assert score.length == 6


def test_score_image_combines_both_fields():
    score = score_image("2023В303529", "191201002", ["MMK 17Г1С", "2023-В303529 100", "19120-1002"])
    assert score.heat_edits == 0
    assert score.sheet_edits == 0
    assert score.length == 20


def test_cer_handles_empty_reference():
    assert cer(0, 0) == 0.0
    assert cer(3, 12) == 0.25


def test_noise_floor_is_high_even_when_ocr_reads_perfectly():
    from src.metrics import noise_floor_heat_cer

    references = ["2023В303529", "2023V31959", "2023Z32168", "2022V25859"]
    ocr_lines = [[ref] for ref in references]
    assert noise_floor_heat_cer(references, ocr_lines) > 0.3


def test_hit_rate_counts_images_within_two_edits():
    from src.metrics import hit_rate

    assert hit_rate([0, 2, 3, 9]) == 0.5
    assert hit_rate([]) == 0.0
