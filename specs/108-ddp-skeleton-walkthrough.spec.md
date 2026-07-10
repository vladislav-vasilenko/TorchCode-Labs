# 108: Walkthrough скелета distributed-train-lab (track: distributed)

## Зависит от
000-conventions, 101–107 (весь трек задач)

## Что сделать
`templates/57_ddp_skeleton_walkthrough.ipynb` — путеводитель по
самостоятельному заполнению `modern_nlp_labs/distributed-train-lab/
skeleton/train_my.py` (TODO 1–6). ЭТО НЕ РЕШЕНИЕ: ноутбук объясняет, ЧТО
должен делать каждый TODO, какие API использовать и какие ловушки есть,
но не содержит готового кода TODO. Ссылку на файл скелета дать как на
внешний репозиторий (клонировать рядом).

## Содержание (по секции на TODO)
Для каждого TODO 1–6 скелета (process group init; DDP-обёртка; FSDP;
DeepSpeed; честный тайминг; запись JSON) — блок из четырёх частей:
1. **Что должно произойти** (контракт в 2–4 строках).
2. **Какими API** (точные имена: `dist.init_process_group`, env
   RANK/LOCAL_RANK/WORLD_SIZE от torchrun, `DistributedDataParallel`,
   `FullyShardedDataParallel` + ShardingStrategy + auto_wrap_policy,
   `deepspeed.initialize` + ds_config, и т.д.) со ссылками на доки.
3. **Ловушки** (2–3 шт. на TODO: gloo vs nccl на CPU/GPU; device_ids;
   какой модуль оборачивать; sampler и шардирование данных; sync перед
   таймером; кто пишет JSON — только rank 0; barrier перед выходом).
4. **Связь с решённой judge-задачей** трека (102–107): «ты уже
   реализовал семантику X — здесь она появляется как API Y».

Финальные секции:
- **CPU-smoke чек-лист**: команда
  `PYTHONPATH=../src torchrun --standalone --nproc_per_node=2
  train_my.py --mode ddp --model tiny --steps 20` (backend gloo, device
  cpu) и признаки успеха (согласованный loss двух процессов, JSON
  записан) — как в STUDY.md лабы.
- **GPU-чек-лист аренды**: краткий список (vast.ai/RunPod, 2×4090,
  порядок прогонов из scripts/run_all.sh, «каждую ячейку ×3, коммить
  JSON сразу») — без выводов и без чисел.

## Приёмка
- nbconvert --execute проходит (в ноутбуке нет ячеек, требующих скелет:
  код-ячейки — только иллюстрации API на локальных объектах или
  выключенные `%%script echo skipped` командные ячейки).
- Ноутбук НЕ содержит цельного работоспособного кода ни одного TODO
  (ревью-критерий; допустимы фрагменты ≤ 2 строк как иллюстрация API).
- Все 6 TODO покрыты всеми четырьмя частями; обе финальные секции есть.
- `pytest -q` — зелёный.

## Не делать
Не копировать эталон `src/distributed_train_lab/train.py` из лабы ни
целиком, ни блоками; не давать ds_config целиком (только структуру и
ссылку на доку DeepSpeed).
