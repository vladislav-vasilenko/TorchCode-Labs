# 020: LLM-judge для открытых ответов (track: infra)

## Зависит от
000-conventions

## Что сделать
Модуль оценки markdown-ответов (выводы экспериментов, research memo) по
рубрике.

Файлы:
- `torch_judge/llm_judge.py`
- `rubrics/` — каталог YAML-рубрик (формат ниже); в этом spec'е создать
  каталог + `rubrics/_example.yaml`.
- `tests/test_llm_judge.py` — с полностью замоканным API.

## Формат рубрики (`rubrics/<name>.yaml`)
```yaml
title: "Conclusions H1–H4"
criteria:
  - id: numbers_cited
    text: "Каждый вывод ссылается на конкретные числа из таблицы результатов"
    weight: 2
  - id: hypotheses_resolved
    text: "H1–H4 явно подтверждены или опровергнуты"
    weight: 2
pass_threshold: 0.7   # доля от максимального взвешенного балла
```

## Поведение
`evaluate(answer_md: str, rubric_path: str) -> JudgeResult`

- Онлайн-режим (env `LLM_JUDGE_API_KEY` задан): POST на OpenAI-совместимый
  `/v1/chat/completions` (`LLM_JUDGE_BASE_URL`, default
  `https://api.openai.com/v1`; `LLM_JUDGE_MODEL`, default разумный).
  Системный промпт: «оцени ответ по каждому критерию, score 0/1/2,
  короткий comment, верни строго JSON-массив
  [{id, score, comment}]». Парсинг устойчив к ```json-обёрткам.
  `JudgeResult`: per-criterion scores/comments, weighted_total,
  max_total, passed (>= pass_threshold).
- Offline-режим (ключа нет): не падать; вернуть `JudgeResult(passed=None)`
  и напечатать рубрику как чек-лист самопроверки («оцени себя сам по
  пунктам»).
- Ошибки сети/парсинга: одна повторная попытка, затем graceful
  degradation в offline-режим с warning.

Интеграция с ноутбуками: helper `torch_judge.llm_judge.check_cell(rubric,
answer_md)` для вызова из ячейки; печатает цветной отчёт per-criterion
(стиль вывода — как pass/fail апстрима).

## Приёмка
- `pytest tests/test_llm_judge.py -q` — зелёный. Обязательные тесты:
  (1) мок-API возвращает валидный JSON → корректные weighted_total и
  passed; (2) мок возвращает JSON в ```json-обёртке → парсится;
  (3) ключа нет → offline-fallback, passed is None, чек-лист напечатан;
  (4) мок падает дважды → fallback без исключения;
  (5) невалидная рубрика (нет criteria) → понятная ошибка.
- В judge-тестах нет реальных сетевых вызовов: `pytest -q --disable-socket`
  (добавить pytest-socket в dev-зависимости) — зелёный, кроме явно
  помеченных integration-тестов (их в CI не гонять).

## Не делать
Не встраивать конкретного провайдера жёстко; не хранить ключи в коде; не
делать LLM-judge обязательным для прохождения треков (offline-путь всегда
работает).
