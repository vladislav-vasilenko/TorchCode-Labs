"""Judge-level tests for the ddp_gradient_sync task."""

from __future__ import annotations

import pytest
import torch

from torch_judge.tasks.ddp_gradient_sync import TASK


def reference_ddp_gradient_sync(
    rank_grads: list[dict[str, torch.Tensor | None]],
) -> dict[str, torch.Tensor | None]:
    """Return validated per-parameter gradient means across ranks."""
    if not rank_grads:
        raise ValueError("rank_grads must contain at least one rank")

    expected_keys = set(rank_grads[0])
    if any(set(rank_grad) != expected_keys for rank_grad in rank_grads[1:]):
        raise ValueError("All ranks must have the same parameter keys")

    synchronized: dict[str, torch.Tensor | None] = {}
    for name in rank_grads[0]:
        gradients = [rank_grad[name] for rank_grad in rank_grads]
        missing = [gradient is None for gradient in gradients]
        if all(missing):
            synchronized[name] = None
            continue
        if any(missing):
            raise ValueError(f"Gradient {name!r} must be None on all ranks or none")

        tensors = [gradient for gradient in gradients if gradient is not None]
        expected_shape = tensors[0].shape
        if any(tensor.shape != expected_shape for tensor in tensors[1:]):
            raise ValueError(f"Gradient {name!r} has inconsistent shapes")

        output_dtype = tensors[0].dtype
        accumulator_dtype = (
            torch.float32
            if output_dtype in (torch.float16, torch.bfloat16)
            else output_dtype
        )
        synchronized[name] = (
            torch.stack([tensor.to(accumulator_dtype) for tensor in tensors])
            .mean(dim=0)
            .to(output_dtype)
        )

    return synchronized


def run_judge_test(candidate, test_code: str) -> None:
    namespace = {"candidate": candidate}
    exec(compile(test_code.replace("{fn}", "candidate"), "<judge-test>", "exec"), namespace)


@pytest.mark.parametrize("judge_test", TASK["tests"], ids=lambda test: test["name"])
def test_each_judge_test_passes_on_reference_solution(judge_test: dict) -> None:
    run_judge_test(reference_ddp_gradient_sync, judge_test["code"])


def test_judge_rejects_sum_instead_of_average() -> None:
    def sums_rank_grads(rank_grads):
        return {
            name: torch.stack([rank[name] for rank in rank_grads]).sum(dim=0)
            for name in rank_grads[0]
        }

    average_test = TASK["tests"][0]
    with pytest.raises(AssertionError):
        run_judge_test(sums_rank_grads, average_test["code"])
