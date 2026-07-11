import importlib.util
import json
from pathlib import Path

from torch_judge import engine
from torch_judge.hints import get_hints

RENDER_HINTS_PATH = Path(__file__).parents[1] / "scripts" / "render_hints.py"
RENDER_HINTS_SPEC = importlib.util.spec_from_file_location("render_hints", RENDER_HINTS_PATH)
assert RENDER_HINTS_SPEC is not None and RENDER_HINTS_SPEC.loader is not None
render_hints_module = importlib.util.module_from_spec(RENDER_HINTS_SPEC)
RENDER_HINTS_SPEC.loader.exec_module(render_hints_module)


def test_get_hints_legacy_task() -> None:
    assert get_hints({"hint": "legacy"}) == ["legacy"]


def test_get_hints_multilevel_task_preserves_order() -> None:
    assert get_hints(
        {"hints": ["first", "second", "third"], "hint": "legacy"}
    ) == [
        "first",
        "second",
        "third",
    ]


def test_get_hints_without_hints() -> None:
    assert get_hints({}) == []


def test_engine_keeps_legacy_hint_output(monkeypatch, capsys) -> None:
    task = {"title": "Legacy", "hint": "legacy hint"}
    monkeypatch.setattr(engine, "get_task", lambda task_id: task)

    engine.hint("legacy")

    assert capsys.readouterr().out == (
        f"\n{engine._YELLOW}💡 Hint for Legacy:{engine._RESET}\n"
        "   legacy hint\n\n"
    )


def test_engine_reveals_multilevel_hints_in_order(monkeypatch, capsys) -> None:
    task = {"title": "Multilevel", "hints": ["first", "second", "third"]}
    monkeypatch.setattr(engine, "get_task", lambda task_id: task)

    for expected in task["hints"]:
        engine.hint("multilevel")
        assert expected in capsys.readouterr().out


def test_render_hints_inserts_three_details_blocks_before_submit() -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["Problem"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["from torch_judge import check\n", 'check("demo")'],
            },
        ]
    }

    render_hints_module.render_hints(
        notebook, {"hints": ["one", "two", "three"]}
    )

    hints_cell = notebook["cells"][1]
    assert hints_cell["metadata"]["tags"] == [render_hints_module.HINTS_CELL_TAG]
    assert "".join(hints_cell["source"]).count("<details>") == 3

