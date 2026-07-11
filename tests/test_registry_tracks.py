import pytest

from torch_judge.tasks import _registry


@pytest.fixture
def distributed_task(monkeypatch):
    task_id = "fixture_distributed_task"
    task = {
        "title": "Fixture distributed task",
        "difficulty": "Medium",
        "track": "distributed",
    }
    monkeypatch.setitem(_registry.TASKS, task_id, task)
    return task_id, task


def test_list_tasks_returns_all_tasks_including_upstream() -> None:
    listed_task_ids = {task_id for task_id, _ in _registry.list_tasks()}

    assert listed_task_ids == set(_registry.TASKS)
    assert "relu" in listed_task_ids


def test_list_tasks_filters_marked_tasks(distributed_task) -> None:
    task_id, task = distributed_task

    distributed_tasks = _registry.list_tasks(track="distributed")

    assert (task_id, task) in distributed_tasks
    assert all(
        listed_task.get("track", "core") == "distributed"
        for _, listed_task in distributed_tasks
    )


def test_task_without_track_is_in_core(monkeypatch) -> None:
    task_id = "fixture_legacy_task"
    task = {
        "title": "Fixture legacy task",
        "difficulty": "Easy",
    }
    monkeypatch.setitem(_registry.TASKS, task_id, task)

    assert (task_id, task) in _registry.list_tasks(track="core")
