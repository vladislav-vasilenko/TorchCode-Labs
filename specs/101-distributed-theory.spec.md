# 101: Теоретический ноутбук трека distributed (track: distributed)

## Зависит от
000-conventions, 030-tracks-registry

## Что сделать
`templates/50_distributed_theory.ipynb` — вводный теор. ноутбук трека.
Только markdown + маленькие иллюстративные code-ячейки (CPU, < 10 сек
каждая). Русский язык, формулы LaTeX, схемы mermaid/ASCII.

## Содержание (обязательные секции, в этом порядке)
1. **Проблема:** почему одна GPU не хватает — память (params, grads,
   optimizer states, activations) и время. Арифметика памяти AdamW:
   вывод байт/параметр для fp32 (4+4+4+4=16) и bf16 mixed precision
   (2+2+4+4+4=16, с master weights; пояснить каждую компоненту).
   Мини code-ячейка: подсчёт для модели 85M и 7B.
2. **DDP:** полная реплика на каждом ранке; allreduce градиентов;
   bucketing и overlap backward↔allreduce (почему это скрывает
   коммуникацию); что НЕ шардится. Схема таймлайна шага.
3. **ZeRO stage 1/2/3:** что именно шардится на каждом стейдже
   (optimizer states / +grads / +params), таблица «память на ранк vs
   коммуникация на шаг». Ссылка arXiv:1910.02054, какие разделы читать.
4. **FSDP:** FULL_SHARD и SHARD_GRAD_OP, соответствие стейджам ZeRO;
   all-gather на forward, reduce-scatter на backward; auto_wrap_policy —
   зачем и на каком уровне оборачивать. Ссылки: PyTorch FSDP tutorial,
   FSDP API docs.
5. **Честные замеры:** почему без `torch.cuda.synchronize()` ms/step
   врут (async execution); warmup; медиана vs среднее.
6. **Куда это масштабируется:** абзац про tensor/pipeline/expert
   parallelism с ссылкой на HF Ultra-Scale Playbook (только карта
   местности, без деталей — это за рамками трека).
7. **Контрольные вопросы** (без ответов, 6–8 шт.) — взять за основу
   вопросы из `modern_nlp_labs/distributed-train-lab/STUDY.md`
   (секция «Вопросы, на которые у тебя должны быть ответы»).
8. **Маршрут трека:** список judge-задач 102–107 по порядку, затем
   walkthrough 108 и tooling 109, затем — заполнение skeleton'а
   в modern_nlp_labs и аренда GPU (вне этого репозитория).

## Приёмка
- `jupyter nbconvert --to notebook --execute
  templates/50_distributed_theory.ipynb` — без ошибок, суммарное время
  исполнения < 60 сек.
- Все 8 секций присутствуют (проверка: заголовки `##` в ноутбуке).
- Ссылки (ZeRO paper, DDP series, FSDP tutorial, Ultra-Scale Playbook) —
  корректные URL.
- `pytest -q` — зелёный (ноутбук ничего не ломает).

## Не делать
Не вставлять готовые куски решения skeleton'а; не превышать разумный
объём (ориентир: 25–40 ячеек) — это конспект-карта, а не учебник.
