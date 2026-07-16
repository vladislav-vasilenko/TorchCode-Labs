"""Judge-level tests for the optimizer_memory_math task."""

from __future__ import annotations

import pytest

from torch_judge.tasks.optimizer_memory_math import TASK


def reference_bytes_per_param(
    param_dtype: str,
    grad_dtype: str,
    optimizer: str,
    master_weights: bool,
) -> int:
    """Reference implementation used to exercise the task's judge tests."""
    dtype_bytes = {"fp32": 4, "bf16": 2, "fp16": 2}
    optimizer_slots = {"sgd": 0, "sgd_momentum": 1, "adamw": 2}

    try:
        parameter_bytes = dtype_bytes[param_dtype]
        gradient_bytes = dtype_bytes[grad_dtype]
        state_slots = optimizer_slots[optimizer]
    except KeyError as error:
        raise ValueError(f"Unsupported configuration value: {error.args[0]}") from None

    return parameter_bytes + gradient_bytes + state_slots * 4 + (4 if master_weights else 0)


def run_judge_test(candidate, test_code: str) -> None:
    namespace = {"candidate": candidate}
    exec(compile(test_code.replace("{fn}", "candidate"), "<judge-test>", "exec"), namespace)


@pytest.mark.parametrize("judge_test", TASK["tests"], ids=lambda test: test["name"])
def test_each_judge_test_passes_on_reference_solution(judge_test: dict) -> None:
    run_judge_test(reference_bytes_per_param, judge_test["code"])


def test_judge_rejects_solution_that_ignores_master_weights() -> None:
    def ignores_master_weights(
        param_dtype: str,
        grad_dtype: str,
        optimizer: str,
        master_weights: bool,
    ) -> int:
        return reference_bytes_per_param(param_dtype, grad_dtype, optimizer, False)

    master_weights_test = TASK["tests"][1]
    with pytest.raises(AssertionError):
        run_judge_test(ignores_master_weights, master_weights_test["code"])
