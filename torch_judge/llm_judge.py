"""LLM-assisted evaluation of Markdown answers against YAML rubrics.

The online judge uses an OpenAI-compatible chat completions endpoint. Set
``LLM_JUDGE_MODEL`` to select the model instead of the visible default.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_DIM = "\033[90m"
_BOLD = "\033[1m"

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-5.6-luna"
_DEFAULT_REASONING_EFFORT = "low"
_SYSTEM_PROMPT = (
    "Оцени ответ по каждому критерию. Для каждого критерия укажи score "
    "0, 1 или 2 и короткий comment. Верни строго JSON-массив "
    "[{\"id\": \"...\", \"score\": 0, \"comment\": \"...\"}] "
    "без дополнительного текста."
)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class RubricError(ValueError):
    """Raised when a rubric is missing required fields or has invalid values."""


class _UnsupportedReasoningError(ValueError):
    """Raised when a backend rejects the optional reasoning parameter."""


@dataclass(frozen=True)
class CriterionResult:
    """Score and feedback for one rubric criterion."""

    id: str
    text: str
    weight: float
    score: int | None
    comment: str


@dataclass(frozen=True)
class JudgeResult:
    """Aggregate result returned by :func:`evaluate`."""

    criteria: list[CriterionResult]
    weighted_total: float
    max_total: float
    passed: bool | None


@dataclass(frozen=True)
class _Criterion:
    id: str
    text: str
    weight: float


@dataclass(frozen=True)
class _Rubric:
    title: str
    criteria: list[_Criterion]
    pass_threshold: float


def _load_rubric(rubric_path: str) -> _Rubric:
    path = Path(rubric_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RubricError(f"Could not load rubric '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise RubricError("Invalid rubric: expected a YAML mapping.")

    raw_criteria = raw.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise RubricError("Invalid rubric: 'criteria' must be a non-empty list.")

    criteria: list[_Criterion] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_criteria, 1):
        if not isinstance(item, dict):
            raise RubricError(f"Invalid rubric criterion #{index}: expected a mapping.")

        criterion_id = item.get("id")
        text = item.get("text")
        weight = item.get("weight")
        if not isinstance(criterion_id, str) or not criterion_id.strip():
            raise RubricError(
                f"Invalid rubric criterion #{index}: 'id' must be a non-empty string."
            )
        if criterion_id in seen_ids:
            raise RubricError(f"Invalid rubric: duplicate criterion id '{criterion_id}'.")
        if not isinstance(text, str) or not text.strip():
            raise RubricError(
                f"Invalid rubric criterion '{criterion_id}': 'text' must be a non-empty string."
            )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise RubricError(
                f"Invalid rubric criterion '{criterion_id}': 'weight' must be a positive number."
            )
        criteria.append(_Criterion(criterion_id, text, float(weight)))
        seen_ids.add(criterion_id)

    threshold = raw.get("pass_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise RubricError("Invalid rubric: 'pass_threshold' must be a number from 0 to 1.")

    title = raw.get("title", path.stem)
    if not isinstance(title, str) or not title.strip():
        raise RubricError("Invalid rubric: 'title' must be a non-empty string when provided.")

    return _Rubric(title, criteria, float(threshold))


def _offline_result(rubric: _Rubric) -> JudgeResult:
    print(
        f"\n{_YELLOW}{_BOLD}LLM judge is offline — "
        f"оцени себя сам по пунктам:{_RESET}"
    )
    print(f"{_BOLD}{rubric.title}{_RESET}")
    for criterion in rubric.criteria:
        print(
            f"  {_YELLOW}☐{_RESET} [{criterion.id}] {criterion.text} "
            f"(weight: {criterion.weight:g})"
        )
    print()

    return JudgeResult(
        criteria=[
            CriterionResult(
                id=criterion.id,
                text=criterion.text,
                weight=criterion.weight,
                score=None,
                comment="Not evaluated: LLM judge is offline.",
            )
            for criterion in rubric.criteria
        ],
        weighted_total=0.0,
        max_total=sum(2 * criterion.weight for criterion in rubric.criteria),
        passed=None,
    )


def _build_messages(answer_md: str, rubric: _Rubric) -> list[dict[str, str]]:
    criteria = [
        {"id": criterion.id, "text": criterion.text, "weight": criterion.weight}
        for criterion in rubric.criteria
    ]
    user_prompt = (
        f"Рубрика:\n{json.dumps(criteria, ensure_ascii=False)}\n\n"
        f"Ответ в Markdown:\n{answer_md}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _post_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None,
) -> str:
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if reasoning_effort is not None:
        request_payload["reasoning_effort"] = reasoning_effort
    payload = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        normalized_error = error_body.casefold()
        unsupported_markers = (
            "unsupported",
            "not supported",
            "unrecognized",
            "unknown parameter",
            "extra inputs are not permitted",
        )
        mentions_reasoning = (
            "reasoning_effort" in normalized_error
            or "reasoning effort" in normalized_error
        )
        if (
            reasoning_effort is not None
            and exc.code in (400, 422)
            and mentions_reasoning
            and any(marker in normalized_error for marker in unsupported_markers)
        ):
            raise _UnsupportedReasoningError(error_body) from exc
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {error_body}") from exc

    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API response does not contain choices[0].message.content.") from exc
    if not isinstance(content, str):
        raise ValueError("API response content must be a string.")
    return content


def _parse_scores(content: str, rubric: _Rubric) -> list[CriterionResult]:
    stripped = content.strip()
    fenced = _JSON_FENCE_RE.fullmatch(stripped)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        raw_scores = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(raw_scores, list):
        raise ValueError("LLM response must be a JSON array.")

    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            raise ValueError("Every LLM result must be a JSON object.")
        criterion_id = item.get("id")
        if not isinstance(criterion_id, str) or criterion_id in by_id:
            raise ValueError("Every LLM result must have a unique string 'id'.")
        by_id[criterion_id] = item

    expected_ids = {criterion.id for criterion in rubric.criteria}
    if set(by_id) != expected_ids:
        raise ValueError("LLM result criterion ids do not match the rubric.")

    results: list[CriterionResult] = []
    for criterion in rubric.criteria:
        item = by_id[criterion.id]
        score = item.get("score")
        comment = item.get("comment")
        if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1, 2):
            raise ValueError(f"Score for criterion '{criterion.id}' must be 0, 1, or 2.")
        if not isinstance(comment, str):
            raise ValueError(f"Comment for criterion '{criterion.id}' must be a string.")
        results.append(
            CriterionResult(
                id=criterion.id,
                text=criterion.text,
                weight=criterion.weight,
                score=score,
                comment=comment,
            )
        )
    return results


def evaluate(answer_md: str, rubric_path: str) -> JudgeResult:
    """Evaluate Markdown against a YAML rubric.

    Online mode requires ``LLM_JUDGE_API_KEY``. Configure the compatible
    endpoint with ``LLM_JUDGE_BASE_URL``, select a model with
    ``LLM_JUDGE_MODEL``, and set reasoning effort with
    ``LLM_JUDGE_REASONING`` (default: ``"low"``). If a backend rejects the
    reasoning parameter, the request is repeated without it. Without a key, or
    after two failed API/parsing attempts, the function prints a self-review
    checklist and returns a result whose ``passed`` value is ``None``.
    """
    rubric = _load_rubric(rubric_path)
    api_key = os.getenv("LLM_JUDGE_API_KEY")
    if not api_key:
        warnings.warn(
            "LLM judge is offline because LLM_JUDGE_API_KEY is not set; "
            "the online model can be selected with LLM_JUDGE_MODEL.",
            UserWarning,
            stacklevel=2,
        )
        return _offline_result(rubric)

    configured_model = os.getenv("LLM_JUDGE_MODEL")
    model = configured_model or _DEFAULT_MODEL
    if configured_model is None:
        warnings.warn(
            f"LLM_JUDGE_MODEL is not set; using default model '{model}'.",
            UserWarning,
            stacklevel=2,
        )

    messages = _build_messages(answer_md, rubric)
    last_error: Exception | None = None
    reasoning_effort: str | None = os.getenv(
        "LLM_JUDGE_REASONING", _DEFAULT_REASONING_EFFORT
    )
    for _ in range(2):
        try:
            try:
                content = _post_chat_completion(
                    api_key=api_key,
                    base_url=os.getenv("LLM_JUDGE_BASE_URL", _DEFAULT_BASE_URL),
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                )
            except _UnsupportedReasoningError:
                reasoning_effort = None
                content = _post_chat_completion(
                    api_key=api_key,
                    base_url=os.getenv("LLM_JUDGE_BASE_URL", _DEFAULT_BASE_URL),
                    model=model,
                    messages=messages,
                    reasoning_effort=None,
                )
            criteria = _parse_scores(content, rubric)
            weighted_total = sum(
                criterion.weight * criterion.score
                for criterion in criteria
                if criterion.score is not None
            )
            max_total = sum(2 * criterion.weight for criterion in criteria)
            return JudgeResult(
                criteria=criteria,
                weighted_total=weighted_total,
                max_total=max_total,
                passed=weighted_total / max_total >= rubric.pass_threshold,
            )
        except Exception as exc:  # Network and response errors share the retry path.
            last_error = exc

    warnings.warn(
        f"LLM judge failed twice ({last_error}); using offline self-review. "
        "Configure the model with LLM_JUDGE_MODEL.",
        UserWarning,
        stacklevel=2,
    )
    return _offline_result(rubric)


def check_cell(rubric: str, answer_md: str) -> JudgeResult:
    """Evaluate a notebook answer and print a colored criterion report.

    The online model is selected with ``LLM_JUDGE_MODEL`` and reasoning effort
    with ``LLM_JUDGE_REASONING``. In offline mode, :func:`evaluate` prints the
    rubric as a self-review checklist instead.
    """
    result = evaluate(answer_md, rubric)
    if result.passed is None:
        return result

    print(f"\n{_BOLD}🧪 LLM rubric report{_RESET}")
    print("─" * 50)
    for criterion in result.criteria:
        if criterion.score == 2:
            color, icon = _GREEN, "✅"
        elif criterion.score == 1:
            color, icon = _YELLOW, "◐"
        else:
            color, icon = _RED, "❌"
        print(f"  {color}{icon} [{criterion.id}] {criterion.score}/2{_RESET} {criterion.text}")
        print(f"     {_DIM}{criterion.comment}{_RESET}")
    print("─" * 50)
    if result.passed:
        print(
            f"  {_GREEN}{_BOLD}✅ Passed: "
            f"{result.weighted_total:g}/{result.max_total:g}{_RESET}\n"
        )
    else:
        print(
            f"  {_RED}{_BOLD}❌ Failed: "
            f"{result.weighted_total:g}/{result.max_total:g}{_RESET}\n"
        )
    return result


__all__ = ["CriterionResult", "JudgeResult", "RubricError", "check_cell", "evaluate"]
