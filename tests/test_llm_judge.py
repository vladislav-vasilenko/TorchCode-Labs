import io
import json

import pytest

from torch_judge import llm_judge


@pytest.fixture
def rubric_path(tmp_path):
    path = tmp_path / "rubric.yaml"
    path.write_text(
        """\
title: Test rubric
criteria:
  - id: evidence
    text: Cites evidence
    weight: 2
  - id: conclusion
    text: States a conclusion
    weight: 1
pass_threshold: 0.7
""",
        encoding="utf-8",
    )
    return str(path)


def _api_json(*scores):
    return json.dumps(
        [
            {"id": criterion_id, "score": score, "comment": comment}
            for criterion_id, score, comment in scores
        ]
    )


class _MockApiResponse:
    def __init__(self, content):
        payload = {"choices": [{"message": {"content": content}}]}
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self._body


def test_valid_api_response_calculates_weighted_result(monkeypatch, rubric_path):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-model")
    monkeypatch.delenv("LLM_JUDGE_REASONING", raising=False)
    api_content = _api_json(
        ("evidence", 2, "Specific numbers cited."),
        ("conclusion", 1, "Conclusion is partial."),
    )

    def mock_urlopen(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-key"
        payload = json.loads(request.data)
        assert payload["model"] == "test-model"
        assert payload["reasoning_effort"] == "low"
        assert timeout == 30
        return _MockApiResponse(api_content)

    monkeypatch.setattr(
        llm_judge.urllib.request,
        "urlopen",
        mock_urlopen,
    )

    result = llm_judge.evaluate("answer", rubric_path)

    assert result.weighted_total == 5
    assert result.max_total == 6
    assert result.passed is True
    assert [(item.id, item.score) for item in result.criteria] == [
        ("evidence", 2),
        ("conclusion", 1),
    ]


def test_json_code_fence_is_parsed(monkeypatch, rubric_path):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-model")
    content = _api_json(
        ("evidence", 1, "Weak evidence."),
        ("conclusion", 1, "Partial conclusion."),
    )
    monkeypatch.setattr(
        llm_judge,
        "_post_chat_completion",
        lambda **kwargs: f"```json\n{content}\n```",
    )

    result = llm_judge.evaluate("answer", rubric_path)

    assert result.weighted_total == 3
    assert result.max_total == 6
    assert result.passed is False


def test_missing_key_returns_offline_checklist(monkeypatch, rubric_path, capsys):
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)

    with pytest.warns(UserWarning, match="LLM_JUDGE_MODEL"):
        result = llm_judge.evaluate("answer", rubric_path)

    output = capsys.readouterr().out
    assert result.passed is None
    assert result.weighted_total == 0
    assert "оцени себя сам по пунктам" in output
    assert "Cites evidence" in output
    assert "States a conclusion" in output


def test_two_api_failures_fall_back_without_exception(
    monkeypatch, rubric_path, capsys
):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-model")
    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise OSError("network unavailable")

    monkeypatch.setattr(llm_judge, "_post_chat_completion", fail)

    with pytest.warns(UserWarning, match="failed twice.*LLM_JUDGE_MODEL"):
        result = llm_judge.evaluate("answer", rubric_path)

    assert calls == 2
    assert result.passed is None
    assert "оцени себя сам по пунктам" in capsys.readouterr().out


def test_missing_criteria_has_clear_error(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("title: Invalid\npass_threshold: 0.7\n", encoding="utf-8")

    with pytest.raises(llm_judge.RubricError, match="criteria.*non-empty list"):
        llm_judge.evaluate("answer", str(path))


def test_default_model_is_visible_in_warning(monkeypatch, rubric_path):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(
        llm_judge,
        "_post_chat_completion",
        lambda **kwargs: _api_json(
            ("evidence", 2, "Good."),
            ("conclusion", 2, "Good."),
        ),
    )

    with pytest.warns(UserWarning, match="gpt-5.6-luna"):
        result = llm_judge.evaluate("answer", rubric_path)

    assert result.passed is True


def test_unsupported_reasoning_is_retried_without_parameter(monkeypatch, rubric_path):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-model")
    monkeypatch.setenv("LLM_JUDGE_REASONING", "medium")
    reasoning_values = []
    api_content = _api_json(
        ("evidence", 2, "Good."),
        ("conclusion", 2, "Good."),
    )

    def mock_urlopen(request, timeout):
        payload = json.loads(request.data)
        reasoning_values.append(payload.get("reasoning_effort"))
        if "reasoning_effort" in payload:
            raise llm_judge.urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(b"Unsupported parameter: reasoning_effort"),
            )
        return _MockApiResponse(api_content)

    monkeypatch.setattr(
        llm_judge.urllib.request,
        "urlopen",
        mock_urlopen,
    )

    result = llm_judge.evaluate("answer", rubric_path)

    assert reasoning_values == ["medium", None]
    assert result.passed is True


def test_parsing_failure_is_retried(monkeypatch, rubric_path):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-model")
    responses = iter(
        [
            "not json",
            _api_json(
                ("evidence", 2, "Good."),
                ("conclusion", 2, "Good."),
            ),
        ]
    )
    monkeypatch.setattr(
        llm_judge, "_post_chat_completion", lambda **kwargs: next(responses)
    )

    result = llm_judge.evaluate("answer", rubric_path)

    assert result.passed is True


def test_check_cell_prints_colored_report(monkeypatch, rubric_path, capsys):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-model")
    monkeypatch.setattr(
        llm_judge,
        "_post_chat_completion",
        lambda **kwargs: _api_json(
            ("evidence", 2, "Good evidence."),
            ("conclusion", 1, "Needs detail."),
        ),
    )

    result = llm_judge.check_cell(rubric_path, "answer")

    output = capsys.readouterr().out
    assert result.passed is True
    assert llm_judge._GREEN in output
    assert "[evidence] 2/2" in output
    assert "Passed: 5/6" in output
