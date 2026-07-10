# 106: Задача fsdp_wrap_policy (track: distributed, Medium)

## Зависит от
000-conventions, 010, 030, 101

## Что сделать
`torch_judge/tasks/fsdp_wrap_policy.py` +
`templates/55_fsdp_wrap_policy.ipynb` + solution.

В условии — теор. вставка: зачем FSDP оборачивает подмодули (гранула
all-gather/reduce-scatter и освобождения памяти), почему «обернуть только
корень» ≈ нет шардирования по времени жизни, и что такое
transformer_auto_wrap_policy в реальном API.

## Поведение
```python
def should_wrap(
    module: torch.nn.Module,
    min_num_params: int,
    block_types: tuple[type, ...],
) -> bool
```
True, если модуль — экземпляр одного из block_types, ИЛИ число его
собственных+дочерних параметров (`sum(p.numel() for p in
module.parameters())`) ≥ min_num_params. Плюс вторая функция:
```python
def plan_wrapping(root: torch.nn.Module, min_num_params: int,
                  block_types: tuple[type, ...]) -> list[str]
```
Обход дерева модулей (named_modules, исключая корень ""), возвращает
имена модулей к обёртке по правилу «оборачивается самый ВЕРХНИЙ модуль,
удовлетворяющий should_wrap; его потомки не рассматриваются».

## Фикстура для тестов
В файле задачи (вне {fn}-кода) — мини-GPT: класс Block(nn.Module)
(attn: nn.Linear×4, mlp: nn.Linear×2), модель = Embedding + 4×Block +
lm_head(Linear). Размеры маленькие (d=32), CPU.

## Тесты judge (минимум)
1. block_types=(Block,), min_num_params=∞ (очень большое): план ==
   ровно 4 блока, в порядке обхода.
2. block_types=(), порог ниже numel блока: блоки выбраны по порогу, их
   внутренние Linear — нет (правило «самый верхний»).
3. Порог ниже numel lm_head: lm_head в плане.
4. Порог = 1: план не содержит вложенных дубликатов (никакое имя из
   плана не является префиксом другого имени из плана).
5. should_wrap: изолированные проверки обоих условий.

## Подсказки (3 уровня)
1. «Что произойдёт с пиковой памятью, если обёрнут только корень модели?»
2. «Почему после обёртки родителя детей пропускаем? Подумай про двойное
   шардирование».
3. Идея обхода: named_modules + отсечение поддерева по префиксу имени.

## Приёмка
Стандартная (000). Дополнительно: solution-ноутбук показывает соответствие
plan_wrapping и реального
`torch.distributed.fsdp.wrap.transformer_auto_wrap_policy` на той же
фикстуре (только демонстрация в markdown/код-ячейке, БЕЗ инициализации
process group — сравнение логики выбора, не реального FSDP).

## Не делать
Не инициализировать torch.distributed в judge-тестах; не оборачивать
модули без параметров.
