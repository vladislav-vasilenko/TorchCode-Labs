"""Helpers for reading task hints with legacy compatibility."""

from __future__ import annotations


def get_hints(task: dict) -> list[str]:
    """Return all task hints, falling back to the legacy single hint."""
    if "hints" in task:
        return task["hints"]
    if "hint" in task:
        return [task["hint"]]
    return []
