"""Single-process ring all-reduce simulation task."""

TASK = {
    "title": "Ring All-Reduce",
    "difficulty": "Medium",
    "function_name": "ring_allreduce",
    "track": "distributed",
    "hints": [
        "Почему наивный all-to-all шлёт O(N²) данных, а кольцо — O(N)?",
        (
            "Для N=3 проследите две фазы в таблице «итерация × ранг × чанк»: "
            "сначала чанк накапливает сумму, затем готовая сумма обходит кольцо."
        ),
        (
            "Индексная арифметика: на итерации t rank i отправляет чанк "
            "(i - t) mod N в reduce-scatter. После этой фазы готовый чанк "
            "продолжает обход в all-gather."
        ),
    ],
    "hint": "Почему наивный all-to-all шлёт O(N²) данных, а кольцо — O(N)?",
    "tests": [
        {
            "name": "matches the mean for four ranks",
            "code": (
                "import torch\n"
                "torch.manual_seed(103)\n"
                "inputs = [torch.randn(3, 5) for _ in range(4)]\n"
                "expected = torch.stack(inputs).mean(dim=0)\n"
                "outputs = {fn}(inputs)\n"
                "assert len(outputs) == 4, f'Expected 4 outputs, got {len(outputs)}'\n"
                "for output in outputs:\n"
                "    assert torch.allclose(output, expected, atol=1e-6), 'Output must equal the input mean'\n"
            ),
        },
        {
            "name": "performs the expected number of sends",
            "code": (
                "import torch\n"
                "for world_size in (3, 5):\n"
                "    sends = []\n"
                "    inputs = [torch.full((7,), float(rank)) for rank in range(world_size)]\n"
                "    {fn}(inputs, on_send=lambda src, dst: sends.append((src, dst)))\n"
                "    expected_sends = 2 * (world_size - 1) * world_size\n"
                "    assert len(sends) == expected_sends, (\n"
                "        f'Expected {expected_sends} sends for N={world_size}, got {len(sends)}'\n"
                "    )\n"
            ),
        },
        {
            "name": "handles uneven chunks",
            "code": (
                "import torch\n"
                "inputs = [torch.arange(10, dtype=torch.float32) + rank for rank in range(4)]\n"
                "expected = torch.stack(inputs).mean(dim=0)\n"
                "outputs = {fn}(inputs)\n"
                "for output in outputs:\n"
                "    assert torch.allclose(output, expected, atol=1e-6), 'Uneven chunks must be reconstructed correctly'\n"
            ),
        },
        {
            "name": "does not mutate inputs",
            "code": (
                "import torch\n"
                "torch.manual_seed(7)\n"
                "inputs = [torch.randn(4, 2) for _ in range(3)]\n"
                "before = [tensor.clone() for tensor in inputs]\n"
                "{fn}(inputs)\n"
                "for original, snapshot in zip(inputs, before):\n"
                "    assert torch.equal(original, snapshot), 'Input tensors must not be mutated'\n"
            ),
        },
        {
            "name": "single rank sends nothing",
            "code": (
                "import torch\n"
                "input_tensor = torch.tensor([1.5, -2.0])\n"
                "sends = []\n"
                "outputs = {fn}([input_tensor], on_send=lambda src, dst: sends.append((src, dst)))\n"
                "assert len(outputs) == 1\n"
                "assert torch.allclose(outputs[0], input_tensor, atol=1e-6)\n"
                "assert sends == [], f'N=1 must not send chunks, got {sends}'\n"
            ),
        },
    ],
}
