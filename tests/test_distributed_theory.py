import ast
import json
import time
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "50_distributed_theory.ipynb"
)

REQUIRED_HEADINGS = [
    "## 1. Проблема: почему одной GPU не хватает",
    "## 2. DDP: полная реплика и синхронизация градиентов",
    "## 3. ZeRO stage 1/2/3: шардирование состояния",
    "## 4. FSDP: параметры собираются только на время работы",
    "## 5. Честные замеры шага",
    "## 6. Куда это масштабируется",
    "## 7. Контрольные вопросы",
    "## 8. Маршрут трека",
]

REQUIRED_URLS = [
    "https://arxiv.org/abs/1910.02054",
    "https://docs.pytorch.org/tutorials/intermediate/ddp_series_intro.html",
    "https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html",
    "https://docs.pytorch.org/docs/stable/fsdp.html",
    "https://huggingface.co/spaces/nanotron/ultrascale-playbook",
]

STUDY_QUESTIONS = [
    (
        "Почему DDP держит полную копию оптимизатора на каждом GPU и сколько "
        "байт на параметр стоит AdamW в fp32 / bf16 mixed?"
    ),
    "Что шардят ZeRO-1/2/3 и какой ценой по коммуникации на шаг?",
    (
        "Чем FSDP FULL_SHARD отличается от SHARD_GRAD_OP и каким стейджам "
        "ZeRO они соответствуют?"
    ),
    "Почему замер без torch.cuda.synchronize() даёт неверные ms/step?",
    (
        "Почему на 85M-модели и 2 GPU FULL_SHARD (скорее всего) медленнее "
        "DDP, и при каком масштабе баланс переворачивается?"
    ),
]


def load_notebook() -> dict:
    with NOTEBOOK_PATH.open(encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def dotted_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_notebook_has_expected_format_and_size() -> None:
    notebook = load_notebook()

    assert notebook["nbformat"] == 4
    assert 25 <= len(notebook["cells"]) <= 40
    assert {cell["cell_type"] for cell in notebook["cells"]} <= {
        "markdown",
        "code",
    }


def test_required_sections_are_level_two_and_in_order() -> None:
    notebook = load_notebook()
    markdown_sources = [
        cell_source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    level_two_headings = [
        line
        for source in markdown_sources
        for line in source.splitlines()
        if line.startswith("## ")
    ]

    assert level_two_headings == REQUIRED_HEADINGS


def test_required_primary_source_urls_are_embedded() -> None:
    notebook_text = "\n".join(
        cell_source(cell)
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "markdown"
    )

    for url in REQUIRED_URLS:
        assert url in notebook_text


def test_study_questions_and_source_are_embedded() -> None:
    notebook_text = "\n".join(
        cell_source(cell)
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "modern_nlp_labs/distributed-train-lab/STUDY.md" in notebook_text
    for question in STUDY_QUESTIONS:
        assert question in notebook_text


def test_track_route_is_complete_and_ordered() -> None:
    notebook_text = "\n".join(
        cell_source(cell)
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "markdown"
    )
    positions = [
        notebook_text.index(f"**{spec_number} —")
        for spec_number in range(102, 110)
    ]

    assert positions == sorted(positions)
    assert "самостоятельно заполнить skeleton" in notebook_text
    assert "арендовать GPU" in notebook_text


def test_code_cells_do_not_contain_distributed_skeleton_calls() -> None:
    banned_calls = {
        "init_process_group",
        "torch.distributed.init_process_group",
        "DistributedDataParallel",
        "torch.nn.parallel.DistributedDataParallel",
        "deepspeed.initialize",
    }
    code_sources = [
        cell_source(cell)
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]

    assert all("modern_nlp_labs" not in source for source in code_sources)
    for source in code_sources:
        tree = ast.parse(source)
        calls = {
            dotted_call_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert calls.isdisjoint(banned_calls)


def test_each_code_cell_executes_on_cpu_in_under_ten_seconds() -> None:
    namespace: dict = {}
    code_sources = [
        cell_source(cell)
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]

    assert code_sources
    for source in code_sources:
        started_at = time.perf_counter()
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
        elapsed = time.perf_counter() - started_at
        assert elapsed < 10
