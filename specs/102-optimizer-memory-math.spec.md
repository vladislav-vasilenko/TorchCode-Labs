# 102: Задача optimizer_memory_math (track: distributed, Easy)

## Зависит от
000-conventions, 010, 030, 101

## Что сделать
`torch_judge/tasks/optimizer_memory_math.py` +
`templates/51_optimizer_memory_math.ipynb` + solution.

## Поведение
```python
def bytes_per_param(
    param_dtype: str,        # "fp32" | "bf16" | "fp16"
    grad_dtype: str,         # "fp32" | "bf16" | "fp16"
    optimizer: str,          # "sgd" | "sgd_momentum" | "adamw"
    master_weights: bool,    # fp32 master copy при mixed precision
) -> int
```
Возвращает байты на один параметр: param + grad + состояния оптимизатора
(momentum: +1 fp32-слот; adamw: m и v, по 4 байта каждый — состояния
оптимизатора всегда fp32) + master copy (4 байта, если True).
dtype→байты: fp32=4, bf16=2, fp16=2. Невалидные строки → ValueError.

## Тесты judge (минимум)
1. Канон fp32 AdamW без master: 4+4+4+4 = 16.
2. Канон bf16 mixed AdamW с master: 2+2+4+4+4 = 16.
3. SGD fp32 без momentum и без master: 8.
4. SGD с momentum fp32: 12.
5. Edge: невалидный dtype → ValueError (pytest.raises-стиль внутри
   кода теста через try/except + assert).

## Подсказки (3 уровня)
1. «Перечисли всё, что лежит в памяти на каждый параметр во время шага».
2. «Состояния Adam (m, v) держат в fp32 даже при bf16-параметрах — почему?»
3. Почти-таблица соответствий без финальной формулы.

## Приёмка
- `python -m pytest tests/tasks/test_optimizer_memory_math.py -q` — зелёный
  (тесты запускают judge-тесты задачи на эталонном решении).
- Judge на эталоне: all pass; на заведомо неверном решении (например,
  игнорирует master_weights) — есть failing test.
- Template исполняется nbconvert'ом; `pytest -q` — зелёный.

## Не делать
Не учитывать активации/буферы — это отдельная тема; явно сказать об этом
в условии.
