"""Judge-level tests for the ring_allreduce task."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torch_judge.tasks.ring_allreduce import TASK


def reference_ring_allreduce(
    rank_tensors: list[torch.Tensor],
    on_send: Callable[[int, int], None] | None = None,
) -> list[torch.Tensor]:
    """Reference single-process simulation of the two ring phases."""
    world_size = len(rank_tensors)
    if world_size == 0:
        return []

    chunks = [list(torch.tensor_split(tensor.clone(), world_size)) for tensor in rank_tensors]

    for step in range(world_size - 1):
        sent = []
        for rank in range(world_size):
            chunk_index = (rank - step) % world_size
            destination = (rank + 1) % world_size
            if on_send is not None:
                on_send(rank, destination)
            sent.append(chunks[rank][chunk_index])
        for rank in range(world_size):
            chunk_index = (rank - step - 1) % world_size
            chunks[rank][chunk_index] = chunks[rank][chunk_index] + sent[(rank - 1) % world_size]

    for step in range(world_size - 1):
        sent = []
        for rank in range(world_size):
            chunk_index = (rank + 1 - step) % world_size
            destination = (rank + 1) % world_size
            if on_send is not None:
                on_send(rank, destination)
            sent.append((chunk_index, chunks[rank][chunk_index]))
        for rank in range(world_size):
            chunk_index, chunk = sent[(rank - 1) % world_size]
            chunks[rank][chunk_index] = chunk

    return [torch.cat(rank_chunks, dim=0) / world_size for rank_chunks in chunks]


def run_judge_test(candidate, test_code: str) -> None:
    namespace = {"candidate": candidate}
    exec(compile(test_code.replace("{fn}", "candidate"), "<judge-test>", "exec"), namespace)


@pytest.mark.parametrize("judge_test", TASK["tests"], ids=lambda test: test["name"])
def test_each_judge_test_passes_on_reference_solution(judge_test: dict) -> None:
    run_judge_test(reference_ring_allreduce, judge_test["code"])


def test_judge_rejects_a_solution_that_skips_ring_sends() -> None:
    def direct_mean(rank_tensors, on_send=None):
        del on_send
        mean = torch.stack(rank_tensors).mean(dim=0)
        return [mean.clone() for _ in rank_tensors]

    send_count_test = TASK["tests"][1]
    with pytest.raises(AssertionError):
        run_judge_test(direct_mean, send_count_test["code"])
