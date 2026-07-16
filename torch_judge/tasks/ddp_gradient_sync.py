"""DDP gradient synchronization semantics task."""

TASK = {
    "title": "DDP Gradient Sync",
    "difficulty": "Easy",
    "function_name": "ddp_gradient_sync",
    "track": "distributed",
    "hints": [
        "Что именно DDP синхронизирует и в какой момент шага?",
        "Почему average, а не sum? Подумай про эквивалентность большому батчу.",
        (
            "Схема: stack по ранкам → mean(dim=0). Перед этим проверь "
            "одинаковые ключи и shape, а также отдельно обработай None-кейсы."
        ),
    ],
    "hint": "Что именно DDP синхронизирует и в какой момент шага?",
    "tests": [
        {
            "name": "averages two parameters across three ranks",
            "code": (
                "import torch\n"
                "torch.manual_seed(104)\n"
                "rank_grads = [\n"
                "    {'weight': torch.randn(2, 3), 'bias': torch.randn(3)}\n"
                "    for _ in range(3)\n"
                "]\n"
                "expected = {\n"
                "    name: torch.stack([rank[name] for rank in rank_grads]).mean(dim=0)\n"
                "    for name in rank_grads[0]\n"
                "}\n"
                "result = {fn}(rank_grads)\n"
                "assert set(result) == {'weight', 'bias'}\n"
                "for name in expected:\n"
                "    assert torch.allclose(result[name], expected[name], atol=1e-6), (\n"
                "        f'{name} must equal the mean across ranks'\n"
                "    )\n"
            ),
        },
        {
            "name": "preserves bf16 dtype and shape",
            "code": (
                "import torch\n"
                "rank_grads = [\n"
                "    {'weight': torch.full((2, 4), value, dtype=torch.bfloat16)}\n"
                "    for value in (1.0, 2.0, 3.0)\n"
                "]\n"
                "result = {fn}(rank_grads)['weight']\n"
                "assert result.dtype == torch.bfloat16, f'Expected bf16, got {result.dtype}'\n"
                "assert result.shape == (2, 4), f'Expected shape (2, 4), got {result.shape}'\n"
                "expected = torch.full((2, 4), 2.0, dtype=torch.bfloat16)\n"
                "assert torch.allclose(result, expected, atol=1e-6), 'Constant bf16 gradients must average exactly'\n"
            ),
        },
        {
            "name": "keeps an all-None frozen gradient",
            "code": (
                "import torch\n"
                "rank_grads = [\n"
                "    {'weight': torch.full((2,), float(rank)), 'frozen': None}\n"
                "    for rank in range(3)\n"
                "]\n"
                "result = {fn}(rank_grads)\n"
                "assert result['frozen'] is None, 'An all-None gradient must stay None'\n"
                "assert torch.allclose(result['weight'], torch.ones(2), atol=1e-6)\n"
            ),
        },
        {
            "name": "rejects inconsistent keys",
            "code": (
                "import torch\n"
                "rank_grads = [\n"
                "    {'weight': torch.ones(2)},\n"
                "    {'bias': torch.ones(2)},\n"
                "]\n"
                "try:\n"
                "    {fn}(rank_grads)\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    assert False, 'Different key sets must raise ValueError'\n"
            ),
        },
        {
            "name": "rejects a partially missing gradient",
            "code": (
                "import torch\n"
                "rank_grads = [\n"
                "    {'weight': torch.ones(2)},\n"
                "    {'weight': None},\n"
                "]\n"
                "try:\n"
                "    {fn}(rank_grads)\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    assert False, 'None on only some ranks must raise ValueError'\n"
            ),
        },
        {
            "name": "rejects an empty rank list",
            "code": (
                "try:\n"
                "    {fn}([])\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    assert False, 'An empty rank list must raise ValueError'\n"
            ),
        },
        {
            "name": "rejects inconsistent shapes",
            "code": (
                "import torch\n"
                "rank_grads = [\n"
                "    {'weight': torch.ones(2, 3)},\n"
                "    {'weight': torch.ones(3, 2)},\n"
                "]\n"
                "try:\n"
                "    {fn}(rank_grads)\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    assert False, 'Different gradient shapes must raise ValueError'\n"
            ),
        },
    ],
}
