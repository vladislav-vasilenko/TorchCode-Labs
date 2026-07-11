"""Render TASK hints as collapsible blocks in a template notebook."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_judge.hints import get_hints  # noqa: E402
from torch_judge.tasks import get_task  # noqa: E402

HINTS_CELL_TAG = "torch-judge-hints"


def details_blocks(hints: list[str]) -> str:
    """Build one collapsible details block per hint level."""
    return "\n\n".join(
        f"<details>\n<summary>Tip {index}</summary>\n\n{hint}\n\n</details>"
        for index, hint in enumerate(hints, 1)
    )


def render_hints(notebook: dict, task: dict) -> dict:
    """Insert or replace the generated hints cell in a notebook."""
    cell = {
        "cell_type": "markdown",
        "id": HINTS_CELL_TAG,
        "metadata": {"tags": [HINTS_CELL_TAG]},
        "source": details_blocks(get_hints(task)).splitlines(keepends=True),
    }
    cells = notebook.setdefault("cells", [])

    for index, existing in enumerate(cells):
        if HINTS_CELL_TAG in existing.get("metadata", {}).get("tags", []):
            cells[index] = cell
            return notebook

    insert_at = next(
        (
            index
            for index, existing in enumerate(cells)
            if existing.get("cell_type") == "code"
            and "from torch_judge import check" in "".join(existing.get("source", []))
        ),
        len(cells),
    )
    cells.insert(insert_at, cell)
    return notebook


def task_id_from_path(path: Path) -> str:
    """Infer a task id from a numbered template filename."""
    match = re.match(r"^\d+_(.+)\.ipynb$", path.name)
    if match is None:
        raise ValueError(f"Cannot infer task id from {path.name!r}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--task-id", help="Override the task id inferred from filename")
    args = parser.parse_args()

    task_id = args.task_id or task_id_from_path(args.template)
    task = get_task(task_id)
    if task is None:
        parser.error(f"unknown task: {task_id}")

    notebook = json.loads(args.template.read_text(encoding="utf-8"))
    render_hints(notebook, task)
    args.template.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
