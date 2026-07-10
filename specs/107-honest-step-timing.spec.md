# 107: Задача honest_step_timing (track: distributed, Medium)

## Зависит от
000-conventions, 010, 030, 101

## Что сделать
`torch_judge/tasks/honest_step_timing.py` +
`templates/56_honest_step_timing.ipynb` + solution.

Теор. вставка в условии: CUDA-вызовы асинхронны; time.time() вокруг
step() без синхронизации меряет постановку в очередь, а не работу;
первые шаги включают компиляцию/аллокации → warmup; медиана устойчивее
среднего к выбросам.

## Поведение
```python
def timed_steps(
    step_fn: Callable[[], None],
    warmup: int,
    iters: int,
    sync_fn: Callable[[], None],   # инъекция: в реальности
) -> dict                          # torch.cuda.synchronize
```
Контракт:
1. warmup вызовов step_fn БЕЗ учёта времени; после warmup — sync_fn.
2. Затем iters измерений: каждый = sync_fn-барьер корректно охватывает
   работу (допустимы обе схемы: sync до старта таймера единожды + sync
   перед каждым снятием времени; проверяется свойствами ниже).
3. Возврат: {"ms_per_step_median": float, "ms_per_step_all": list[float]
   длины iters, "warmup": int, "iters": int}. Медиана — именно медиана.
4. warmup=0 допустим; iters < 1 → ValueError.

## Тесты judge (минимум, через инъекцию фейков)
1. Порядок и количество: счётчики-обёртки step_fn/sync_fn; step_fn
   вызван warmup+iters раз; sync_fn вызван ≥ iters раз И хотя бы один
   раз между последним warmup-вызовом и первым измерением.
2. Warmup исключён: step_fn-фейк, который «спит» (эмуляция через
   monkeypatch time.perf_counter, детерминированная последовательность
   времён) 100ms первые warmup раз и 10ms далее → медиана ≈ 10ms.
3. Медиана, не среднее: последовательность времён с одним выбросом
   (10,10,10,10,1000)ms → медиана 10.
4. len(ms_per_step_all) == iters; iters=0 → ValueError.
5. Тест-антипаттерн: решение, не вызывающее sync_fn вообще, валит тест 1
   (проверить на нарочно неверном эталоне в tests/, не в задаче).

## Подсказки (3 уровня)
1. «Что именно возвращает управление сразу при вызове CUDA-кернела?»
2. «Куда поставить синхронизацию, чтобы не мерить хвост предыдущего шага?»
3. Псевдосхема: warmup loop → sync → [t0=now; step; sync; t1=now] × iters.

## Приёмка
Стандартная (000). Время выполнения judge-тестов < 5 сек (все таймеры —
через monkeypatch perf_counter, без реального sleep).

## Не делать
Не завязываться на torch.cuda внутри задачи (sync_fn — инъекция; на CPU
передаётся no-op) — это и делает задачу judge-able без GPU.
