# 109: Тулинг результатов + LLM-judge выводов (track: distributed)

## Зависит от
000-conventions, 020-llm-judge, 101, 108

## Что сделать
Два артефакта, завершающих трек.

### A. Скрипт агрегации результатов
`scripts/make_results_table.py` (в torchcode-labs; пользователь скопирует
или вызовет с путём к лабе):

```
python scripts/make_results_table.py --results-dir <path>/results \
    --out-md table.md --out-plots plots/
```
Поведение:
1. Читает все `results/*.json` формата лабы (поля: mode, model, params_m,
   world_size, device, bf16, ms_per_step, tokens_per_s,
   peak_vram_gb_per_rank: list, final_loss; допускать отсутствующие
   поля → NaN, warning).
2. Группирует по (mode, world_size); по каждой группе — медианы
   ms_per_step, tokens_per_s, final_loss; peak VRAM = медиана
   max(per_rank).
3. Считает scaling vs baseline: tokens_per_s(mode) /
   tokens_per_s(mode="single") — если single есть; иначе колонка «—».
4. Генерирует markdown-таблицу с фиксированным порядком строк:
   single, ddp, fsdp FULL_SHARD, fsdp SHARD_GRAD_OP, deepspeed ZeRO-2,
   deepspeed ZeRO-3 (отсутствующие режимы пропускаются).
5. Два PNG (matplotlib, без seaborn): bar tokens/s по режимам;
   bar peak VRAM/rank по режимам. Заголовки включают model и device.

Тесты `tests/test_make_results_table.py` на фикстурных JSON (положить в
`tests/fixtures/results_demo/`, 3 прогона × 3 режима, значения с
известными медианами): правильные медианы (включая чётное число прогонов),
правильный порядок строк, scaling посчитан верно, битый JSON → warning и
пропуск (не падение), PNG-файлы созданы.

### B. Рубрика LLM-judge для Conclusions
`rubrics/distributed_conclusions.yaml` + мини-ноутбук-ячейка в конце
walkthrough (108) «вставь свои Conclusions и прогони judge». Критерии
(weights в скобках):
- numbers_cited (2): каждый вывод ссылается на конкретные числа таблицы;
- hypotheses_resolved (2): H1–H4 явно подтверждены/опровергнуты по одному;
- crossover_named (2): назван масштаб/условия, при которых FULL_SHARD
  начинает окупаться, с обоснованием через communication/memory trade-off;
- limitations (1): ограничения сетапа названы (2 GPU, 1 узел, 85M,
  char-level data);
- no_folklore (1): нет утверждений, не следующих из данных.
pass_threshold: 0.7.

## Приёмка
- `pytest tests/test_make_results_table.py -q` — зелёный.
- Запуск скрипта на фикстурах из CLI создаёт table.md и 2 PNG
  (проверяется тестом через tmp_path).
- Рубрика валидируется загрузчиком из 020 (`pytest` тест на парсинг).
- Мок-тест: фикстурный «хороший» ответ проходит рубрику (замоканный
  API), фикстурный «плохой» (без чисел) — нет.
- `pytest -q` — зелёный.

## Не делать
Скрипт НЕ пишет ничего в README лабы напрямую (только table.md/PNG —
вставляет пользователь); не генерировать текст Conclusions ни в каком
виде (даже как «пример хорошего ответа» — вместо него в рубрике поле
text достаточно описательное).
