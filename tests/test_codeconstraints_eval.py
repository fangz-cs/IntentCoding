import argparse
import json
from pathlib import Path

import pytest

from intentcoding.codeconstraints_eval import (
    EvaluationConfig,
    completion_from_record,
    evaluate_completion,
    evaluate_files,
    run,
)

NO_ARG_PROMPT = """
def func():
    '''Return a collection.
    '''
"""
WITH_N_PROMPT = """
def func(n):
    '''Return a collection based on n.
    '''

"""


def problem(
    task_id: str,
    constraints: dict[str, object],
    *,
    takes_n: bool,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "prompt": WITH_N_PROMPT if takes_n else NO_ARG_PROMPT,
        "constraints": constraints,
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_level4_accepts_a_fenced_full_function() -> None:
    record = problem(
        "level_4_task_0",
        {
            "returnType": "tuple",
            "dataType": "float",
            "sizeCons": "greater_or_equal",
            "length": 3,
        },
        takes_n=True,
    )
    completion = """Here is the implementation:
```python
def func(n):
    return tuple(float(n + offset) for offset in range(3))
```
"""

    result = evaluate_completion(record, completion)

    assert result.status == "passed"
    assert result.all_pass
    assert result.checks == {
        "returnType": True,
        "dataType": True,
        "sizeCons": True,
        "length": True,
    }


def test_level4_reports_each_failed_constraint() -> None:
    record = problem(
        "level_4_task_1",
        {
            "returnType": "tuple",
            "dataType": "float",
            "sizeCons": "greater",
            "length": 3,
        },
        takes_n=True,
    )

    result = evaluate_completion(
        record,
        "    return [float(n - 1), float(n - 1)]",
    )

    assert result.status == "failed"
    assert result.checks == {
        "returnType": False,
        "dataType": True,
        "sizeCons": False,
        "length": False,
    }


def test_level3_checks_only_active_constraints() -> None:
    record = problem(
        "level_3_task_0",
        {
            "dataType": "float",
            "sizeCons": "greater_or_equal",
            "length": 2,
        },
        takes_n=True,
    )

    result = evaluate_completion(
        record,
        "return [float(n), float(n + 1)]",
    )

    assert result.all_pass
    assert result.checks == {
        "returnType": None,
        "dataType": True,
        "sizeCons": True,
        "length": True,
    }


@pytest.mark.parametrize(
    ("record", "completion", "applicable"),
    [
        (
            problem(
                "level_2_datatype",
                {"dataType": "integer"},
                takes_n=False,
            ),
            "return [1, 2]",
            {"returnType", "dataType"},
        ),
        (
            problem(
                "level_2_length",
                {"length": 2},
                takes_n=False,
            ),
            "return [None, None]",
            {"returnType", "length"},
        ),
        (
            problem(
                "level_2_size",
                {"sizeCons": "less"},
                takes_n=True,
            ),
            "return [n - 1]",
            {"returnType", "sizeCons"},
        ),
    ],
)
def test_level2_preserves_implicit_list_constraint(
    record: dict[str, object],
    completion: str,
    applicable: set[str],
) -> None:
    result = evaluate_completion(record, completion)

    assert result.all_pass
    assert {
        name
        for name, value in result.checks.items()
        if value is not None
    } == applicable


def test_syntax_errors_and_timeouts_are_explicit() -> None:
    record = problem(
        "level_2_length",
        {"length": 1},
        takes_n=False,
    )

    syntax_error = evaluate_completion(record, "return (")
    timeout = evaluate_completion(
        record,
        "while True:\n    pass",
        EvaluationConfig(timeout_seconds=0.25, memory_limit_mb=128),
    )

    assert syntax_error.status == "syntax_error"
    assert syntax_error.error_type == "SyntaxError"
    assert timeout.status == "timeout"
    assert timeout.error_type == "TimeoutExpired"


def test_completion_reader_supports_release_and_legacy_formats() -> None:
    assert (
        completion_from_record(
            {
                "completions": [
                    {"text": "first", "score": -2.0},
                    {"text": "second", "score": -1.0},
                ]
            }
        )
        == "first"
    )
    assert (
        completion_from_record(
            {"completions": [["low", 0.1], ["high", 0.9]]}
        )
        == "high"
    )
    assert completion_from_record({"raw_completion": "body"}) == "body"


def test_cli_evaluates_release_jsonl_without_copying_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "evaluation.jsonl"
    write_jsonl(
        dataset,
        [
            problem(
                "datatype",
                {"dataType": "integer"},
                takes_n=False,
            ),
            problem("length", {"length": 2}, takes_n=False),
        ],
    )
    write_jsonl(
        predictions,
        [
            {
                "task_id": "datatype",
                "completions": [{"text": "return [1]", "score": -0.1}],
            },
            {
                "task_id": "length",
                "completions": [{"text": "return []", "score": -0.2}],
            },
        ],
    )
    args = argparse.Namespace(
        dataset=dataset,
        predictions=predictions,
        output=output,
        completion_key=None,
        timeout=2.0,
        memory_limit_mb=128,
        allow_partial=False,
        overwrite=False,
        allow_code_execution=True,
    )

    summary = run(args)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["accuracy"] == 0.5
    assert json.loads(capsys.readouterr().out) == summary
    details = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [detail["all_pass"] for detail in details] == [True, False]
    assert all("prompt" not in detail for detail in details)
    assert all("completion" not in detail for detail in details)


def test_file_evaluation_requires_complete_unique_task_ids(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        dataset,
        [
            problem("one", {"length": 1}, takes_n=False),
            problem("two", {"length": 1}, takes_n=False),
        ],
    )
    write_jsonl(
        predictions,
        [{"task_id": "one", "completion": "return [0]"}],
    )

    with pytest.raises(ValueError, match="missing 1 task_id"):
        evaluate_files(dataset, predictions)

    write_jsonl(
        predictions,
        [
            {"task_id": "one", "completion": "return [0]"},
            {"task_id": "one", "completion": "return [0]"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate task_id"):
        evaluate_files(dataset, predictions, allow_partial=True)
