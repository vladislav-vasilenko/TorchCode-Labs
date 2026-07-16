"""Optimizer-state memory arithmetic task."""

TASK = {
    "title": "Optimizer Memory Math",
    "difficulty": "Easy",
    "function_name": "bytes_per_param",
    "track": "distributed",
    "hints": [
        "Перечисли всё, что лежит в памяти на каждый параметр во время шага.",
        "Состояния Adam (m, v) держат в fp32 даже при bf16-параметрах — почему?",
        (
            "Почти-таблица: fp32 = 4 B, bf16/fp16 = 2 B; gradient занимает "
            "место своего dtype; SGD = 0 state slots, SGD+momentum = 1 fp32 slot, "
            "AdamW = 2 fp32 slots; master copy = ещё 1 fp32 slot."
        ),
    ],
    "hint": "Перечисли всё, что лежит в памяти на каждый параметр во время шага.",
    "tests": [
        {
            "name": "fp32 AdamW without master weights",
            "code": (
                "result = {fn}('fp32', 'fp32', 'adamw', False)\n"
                "assert result == 16, f'Expected 16 bytes, got {result}'\n"
            ),
        },
        {
            "name": "bf16 mixed AdamW with master weights",
            "code": (
                "result = {fn}('bf16', 'bf16', 'adamw', True)\n"
                "assert result == 16, f'Expected 16 bytes, got {result}'\n"
            ),
        },
        {
            "name": "fp32 SGD without momentum",
            "code": (
                "result = {fn}('fp32', 'fp32', 'sgd', False)\n"
                "assert result == 8, f'Expected 8 bytes, got {result}'\n"
            ),
        },
        {
            "name": "fp32 SGD with momentum",
            "code": (
                "result = {fn}('fp32', 'fp32', 'sgd_momentum', False)\n"
                "assert result == 12, f'Expected 12 bytes, got {result}'\n"
            ),
        },
        {
            "name": "invalid dtype raises ValueError",
            "code": (
                "try:\n"
                "    {fn}('fp64', 'fp32', 'adamw', False)\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    assert False, 'Invalid dtype must raise ValueError'\n"
            ),
        },
    ],
}
