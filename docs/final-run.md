# Итоговый прогон на тестовой выборке

Запуск разрешён и начат 2026-11-15 00:54. До этого момента тестовая выборка (581 кадр) нигде не использовалась: ни при подборе модели, ни при настройке конвейера.

Условия, зафиксированные перед запуском:

- Конфигурация: `PipelineConfig()` по умолчанию (PaddleOCR, маска оверлея, расширение точек 3×3, углы 0/90/180/270, постобработка, без второго прохода), описана в [ocr-model-choice.md](ocr-model-choice.md).
- Решение о читаемости: `data/readability.csv`, SHA-256 `3994cb9dbd6796ca62afb2114f320a0ed728a5a373d9b2f9ce059920090b22ce` (совпадает с [readability-decision.md](readability-decision.md)).
- Команды:
  - `python scripts/evaluate.py --name final-test --split test --final --per-format 100000`
  - `python scripts/evaluate.py --name final-rotated --split test --rotated --final --per-format 10`
- Результаты — `reports/final-test.csv`, `reports/final-rotated.csv`; сводка и выводы — в `docs/evaluation-report.md`.
- После прогона конвейер не меняется. Если потребуется доработка, она делается по выборке настройки, тест прогоняется заново, и в отчёте указывается, что это повторный прогон.
