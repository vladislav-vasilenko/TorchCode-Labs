# План выполнения спеков

| Spec | Статус | Ветка | Модель | Заметки |
|---|---|---|---|---|
| 010-hints-engine | ✅ done | spec/010-hints-engine | Sol High | pytest: 7 тестов; web-кнопка не проверена руками |
| 020-llm-judge | ✅ done | spec/020-llm-judge | Sol High | default gpt-5.6-luna; двухуровневые моки |
| 030-tracks-registry | ✅ done | spec/030-tracks-registry | Sol High | фикстуры через monkeypatch |
| 040-upstream-theory | ✅ done | spec/040-upstream-theory | Sol High + Terra Med | постфактум-спек; заголовки материалов вариативны; ppo-спойлер убран |
| 101-distributed-theory | ✅ done | spec/101-distributed-theory | Sol Med | 29 ячеек, 7 структурных тестов (AST-запрет решений) |
| 101-fix-critique | ✅ done | spec/101-fix-critique | Terra Med | аудит-правки: bf16-модель, world size, SHARD_GRAD_OP, barrier-замеры |
| 102-optimizer-memory-math | ✅ done | spec/102-optimizer-memory-math | Sol High | TASK + template/solution; 6 новых тестов, полный pytest: 31 passed |
| 103-ring-allreduce | ✅ done | spec/103-ring-allreduce | Terra Med | TASK + template/solution; 6 новых тестов, полный pytest: 37 passed |
| 104-ddp-gradient-sync | ✅ done | spec/104-ddp-gradient-sync | Terra Med | TASK + template/solution; 8 новых тестов, полный pytest: 45 passed |
| 105-zero-shard-states | ⬜ todo | — | план: Sol Med | |
| 106-fsdp-wrap-policy | ⬜ todo | — | план: Terra Med | |
| 107-honest-step-timing | ⬜ todo | — | план: Terra Med | |
| 108-ddp-skeleton-walkthrough | ⬜ todo | — | план: Sol Med | |
| 109-results-tooling-conclusions | ⬜ todo | — | план: Terra Med | |
| 2XX inference track | 📝 спеки не написаны | — | — | после эпика 1 |
| 3XX nanogpt/positional track | 📝 спеки не написаны | — | — | после эпика 1 |

## Кандидаты / перед интервью
- [ ] accumulation / DistributedSampler / no_sync — досмотреть при ревью 108
- [ ] α–β модель коллективов, топология сети — только если дойдёт до GigaChat-core интервью
- [ ] FSDP чекпоинтинг (state_dict_type, resume при смене world size) — доки за 1–2 дня до интервью
