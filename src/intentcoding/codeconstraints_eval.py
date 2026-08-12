"""Execution-based evaluation for the CodeConstraints benchmark."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

CHECK_NAMES = ("returnType", "dataType", "sizeCons", "length")
RETURN_TYPES = frozenset({"list", "tuple", "set"})
DATA_TYPES = frozenset({"integer", "float"})
SIZE_CONSTRAINTS = frozenset(
    {"greater", "greater_or_equal", "less", "less_or_equal"}
)
MAX_COMPLETION_CHARS = 1_000_000

Record = dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    """Resource limits for one generated completion."""

    timeout_seconds: float = 2.0
    memory_limit_mb: int = 512

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")


@dataclass(frozen=True)
class TaskEvaluation:
    """Constraint checks for one CodeConstraints task."""

    task_id: Any
    status: str
    checks: dict[str, bool | None]
    all_pass: bool
    error_type: str | None = None

    def as_dict(self) -> Record:
        result: Record = {
            "task_id": self.task_id,
            "status": self.status,
            "checks": self.checks,
            "all_pass": self.all_pass,
        }
        if self.error_type is not None:
            result["error_type"] = self.error_type
        return result


def _expected_checks(constraints: Any) -> dict[str, Any]:
    if not isinstance(constraints, Mapping) or not constraints:
        raise ValueError("constraints must be a non-empty object")

    unknown = set(constraints) - set(CHECK_NAMES)
    if unknown:
        raise ValueError(f"unknown constraints: {sorted(unknown)}")

    expected = dict(constraints)
    if len(expected) == 1 and "returnType" not in expected:
        expected["returnType"] = "list"

    return_type = expected.get("returnType")
    if return_type is not None and return_type not in RETURN_TYPES:
        raise ValueError(f"unsupported returnType: {return_type!r}")

    data_type = expected.get("dataType")
    if data_type is not None and data_type not in DATA_TYPES:
        raise ValueError(f"unsupported dataType: {data_type!r}")

    size_constraint = expected.get("sizeCons")
    if (
        size_constraint is not None
        and size_constraint not in SIZE_CONSTRAINTS
    ):
        raise ValueError(f"unsupported sizeCons: {size_constraint!r}")

    length = expected.get("length")
    if length is not None and (
        isinstance(length, bool) or not isinstance(length, int) or length < 1
    ):
        raise ValueError("length must be a positive integer")

    return {
        name: expected[name]
        for name in CHECK_NAMES
        if name in expected
    }


def _takes_n(prompt: Any) -> bool:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    try:
        tree = ast.parse(prompt)
    except SyntaxError as error:
        raise ValueError("prompt is not valid Python") from error

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "func"
    ]
    if len(functions) != 1:
        raise ValueError("prompt must define exactly one function named 'func'")

    arguments = functions[0].args
    positional = [*arguments.posonlyargs, *arguments.args]
    if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        raise ValueError("func must accept either zero arguments or only n")
    if len(positional) == 0:
        return False
    if len(positional) == 1 and positional[0].arg == "n":
        return True
    raise ValueError("func must accept either zero arguments or only n")


def _fenced_code(text: str) -> str:
    matches = list(
        re.finditer(
            r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return text
    blocks = [match.group(1) for match in matches]
    return next(
        (block for block in blocks if re.search(r"\bdef\s+func\s*\(", block)),
        blocks[0],
    )


def _node_source(source: str, node: ast.AST) -> str | None:
    lines = source.splitlines()
    end_line = getattr(node, "end_lineno", None)
    if end_line is None:
        return None
    start_line = getattr(node, "lineno", 1)
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start_line = min(
            start_line,
            *(decorator.lineno for decorator in decorators),
        )
    return textwrap.dedent("\n".join(lines[start_line - 1 : end_line]))


def _named_function_source(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "func"
            ):
                return _node_source(text, node)

    lines = text.splitlines()
    for start, line in enumerate(lines):
        if not re.match(r"^\s*(?:async\s+)?def\s+func\s*\(", line):
            continue
        base_indent = len(line) - len(line.lstrip())
        block = [line]
        for following in lines[start + 1 :]:
            if following.strip():
                indent = len(following) - len(following.lstrip())
                if indent <= base_indent:
                    break
            block.append(following)
        candidate = textwrap.dedent("\n".join(block))
        try:
            candidate_tree = ast.parse(candidate)
        except SyntaxError:
            return None
        for node in candidate_tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "func"
            ):
                return _node_source(candidate, node)
    return None


def _function_body(text: str) -> str:
    text = re.sub(
        r"^\s*(?:Assistant:|###\s*Answer\s*:?)\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip("\n")
    if not text:
        return ""

    lines = text.splitlines()
    first_nonempty = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_nonempty is None:
        return ""
    lines = lines[first_nonempty:]
    first_indent = len(lines[0]) - len(lines[0].lstrip())

    body_lines: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index > 0 and stripped.startswith(
            ("```", "Human:", "User:", "Assistant:", "###")
        ):
            break
        if (
            index > 0
            and first_indent > 0
            and stripped
            and len(line) - len(line.lstrip()) < first_indent
        ):
            break
        body_lines.append(line)

    body = textwrap.dedent("\n".join(body_lines)).rstrip()
    return textwrap.indent(body, "    ", lambda line: bool(line.strip()))


def prepare_completion_source(prompt: str, completion: str) -> str:
    """Combine a benchmark prompt with a body or full-function completion."""

    if not isinstance(completion, str):
        raise ValueError("completion must be a string")
    if len(completion) > MAX_COMPLETION_CHARS:
        raise ValueError(
            f"completion exceeds {MAX_COMPLETION_CHARS} characters"
        )

    code = _fenced_code(completion)
    full_function = _named_function_source(code)
    if full_function is not None:
        return full_function.rstrip() + "\n"

    body = _function_body(code)
    if not body:
        return prompt.rstrip() + "\n"
    return prompt.rstrip() + "\n" + body + "\n"


def _failure(
    task_id: Any,
    expected: Mapping[str, Any],
    status: str,
    error_type: str,
) -> TaskEvaluation:
    checks = {
        name: False if name in expected else None
        for name in CHECK_NAMES
    }
    return TaskEvaluation(
        task_id=task_id,
        status=status,
        checks=checks,
        all_pass=False,
        error_type=error_type,
    )


def _resource_limiter(config: EvaluationConfig) -> Any:
    if os.name != "posix":
        return None

    def apply_limits() -> None:
        import resource

        cpu_seconds = max(1, math.ceil(config.timeout_seconds))
        memory_bytes = config.memory_limit_mb * 1024 * 1024
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (cpu_seconds, cpu_seconds + 1),
        )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (memory_bytes, memory_bytes),
        )
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))

    return apply_limits


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


def _execute(
    task_id: Any,
    source: str,
    expected: dict[str, Any],
    takes_n: bool,
    config: EvaluationConfig,
) -> TaskEvaluation:
    try:
        compile(source, "<completion>", "exec")
    except SyntaxError:
        return _failure(task_id, expected, "syntax_error", "SyntaxError")

    payload = json.dumps(
        {
            "source": source,
            "expected": expected,
            "takes_n": takes_n,
        }
    )
    worker = Path(__file__).with_name("_codeconstraints_worker.py")
    command = [sys.executable, "-I", "-S", str(worker)]
    environment = {
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }

    with tempfile.TemporaryDirectory(prefix="intentcoding-eval-") as directory:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=directory,
            env=environment,
            start_new_session=os.name == "posix",
            preexec_fn=_resource_limiter(config),
        )
        try:
            stdout, _ = process.communicate(
                payload,
                timeout=config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            process.communicate()
            return _failure(
                task_id,
                expected,
                "timeout",
                "TimeoutExpired",
            )
        finally:
            if process.poll() is not None:
                _kill_process_group(process)

    if process.returncode != 0:
        return _failure(
            task_id,
            expected,
            "runtime_error",
            f"ProcessExit{process.returncode}",
        )

    try:
        worker_result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("evaluation worker returned invalid JSON") from error
    if not isinstance(worker_result, dict):
        raise RuntimeError("evaluation worker returned a non-object result")

    worker_checks = worker_result.get("checks")
    if not isinstance(worker_checks, dict):
        raise RuntimeError("evaluation worker omitted constraint checks")
    if set(worker_checks) != set(expected):
        raise RuntimeError("evaluation worker returned unexpected checks")
    if any(type(value) is not bool for value in worker_checks.values()):
        raise RuntimeError("evaluation worker returned non-boolean checks")

    checks = {
        name: worker_checks[name] if name in expected else None
        for name in CHECK_NAMES
    }
    worker_status = worker_result.get("status")
    if worker_status == "runtime_error":
        error_type = worker_result.get("error_type")
        if not isinstance(error_type, str) or not error_type:
            raise RuntimeError("evaluation worker omitted its error type")
        return TaskEvaluation(
            task_id=task_id,
            status="runtime_error",
            checks=checks,
            all_pass=False,
            error_type=error_type,
        )
    if worker_status != "completed":
        raise RuntimeError(
            f"evaluation worker returned unknown status: {worker_status!r}"
        )

    all_pass = all(worker_checks.values())
    return TaskEvaluation(
        task_id=task_id,
        status="passed" if all_pass else "failed",
        checks=checks,
        all_pass=all_pass,
    )


def evaluate_completion(
    problem: Mapping[str, Any],
    completion: str,
    config: EvaluationConfig | None = None,
) -> TaskEvaluation:
    """Evaluate one generated completion against a CodeConstraints record."""

    if "task_id" not in problem:
        raise ValueError("problem is missing task_id")
    expected = _expected_checks(problem.get("constraints"))
    prompt = problem.get("prompt")
    takes_n = _takes_n(prompt)
    source = prepare_completion_source(prompt, completion)
    return _execute(
        problem["task_id"],
        source,
        expected,
        takes_n,
        config or EvaluationConfig(),
    )


def _load_jsonl(path: Path) -> list[tuple[int, Record]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: list[tuple[int, Record]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: each row must be an object"
                )
            records.append((line_number, record))
    return records


def _index_records(path: Path) -> dict[str, Record]:
    indexed: dict[str, Record] = {}
    for line_number, record in _load_jsonl(path):
        if "task_id" not in record:
            raise ValueError(f"{path}:{line_number}: missing task_id")
        key = str(record["task_id"])
        if key in indexed:
            raise ValueError(f"{path}:{line_number}: duplicate task_id {key!r}")
        indexed[key] = record
    return indexed


def _ranked_completion(value: Any) -> str:
    if not isinstance(value, list) or not value:
        raise ValueError("completions must be a non-empty list")

    first = value[0]
    if isinstance(first, dict):
        text = first.get("text")
        if not isinstance(text, str):
            raise ValueError("completions[0].text must be a string")
        return text
    if isinstance(first, str):
        return first

    legacy = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) < 2
            or not isinstance(item[0], str)
            or isinstance(item[1], bool)
            or not isinstance(item[1], (int, float))
        ):
            raise ValueError(
                "legacy completions must contain [text, score] entries"
            )
        legacy.append(item)
    return max(legacy, key=lambda item: item[1])[0]


def completion_from_record(
    record: Mapping[str, Any],
    completion_key: str | None = None,
) -> str:
    """Read a completion from release or legacy prediction formats."""

    if completion_key is not None:
        if completion_key not in record:
            raise ValueError(
                f"prediction is missing completion key {completion_key!r}"
            )
        value = record[completion_key]
        if completion_key == "completions":
            return _ranked_completion(value)
        if not isinstance(value, str):
            raise ValueError(f"{completion_key!r} must contain a string")
        return value

    if "completions" in record:
        return _ranked_completion(record["completions"])
    for key in ("completion", "o_completion", "raw_completion"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(
        "prediction must contain completions, completion, "
        "o_completion, or raw_completion"
    )


def summarize_evaluations(
    evaluations: Sequence[TaskEvaluation],
) -> Record:
    """Aggregate task and per-constraint accuracy."""

    total = len(evaluations)
    passed = sum(result.all_pass for result in evaluations)
    status_counts = Counter(result.status for result in evaluations)
    constraints: Record = {}
    for name in CHECK_NAMES:
        applicable = [
            result.checks[name]
            for result in evaluations
            if result.checks[name] is not None
        ]
        if not applicable:
            continue
        constraint_passed = sum(applicable)
        constraints[name] = {
            "passed": constraint_passed,
            "total": len(applicable),
            "accuracy": constraint_passed / len(applicable),
        }

    passed_counts = Counter(
        sum(value is True for value in result.checks.values())
        for result in evaluations
    )
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "constraints": constraints,
        "passed_constraint_counts": {
            str(count): frequency
            for count, frequency in sorted(passed_counts.items())
        },
    }


def evaluate_files(
    dataset_path: Path,
    predictions_path: Path,
    *,
    config: EvaluationConfig | None = None,
    completion_key: str | None = None,
    allow_partial: bool = False,
) -> tuple[list[TaskEvaluation], Record]:
    """Evaluate prediction JSONL against a CodeConstraints JSONL split."""

    dataset = _index_records(dataset_path)
    predictions = _index_records(predictions_path)
    unknown = sorted(set(predictions) - set(dataset))
    if unknown:
        raise ValueError(
            f"predictions contain {len(unknown)} unknown task_id(s): "
            f"{unknown[:5]}"
        )
    missing = sorted(set(dataset) - set(predictions))
    if missing and not allow_partial:
        raise ValueError(
            f"predictions are missing {len(missing)} task_id(s): {missing[:5]}"
        )

    evaluation_config = config or EvaluationConfig()
    evaluations = []
    for task_id, problem in dataset.items():
        if task_id not in predictions:
            continue
        try:
            completion = completion_from_record(
                predictions[task_id],
                completion_key,
            )
        except ValueError as error:
            raise ValueError(
                f"{predictions_path}: task_id {task_id!r}: {error}"
            ) from error
        evaluations.append(
            evaluate_completion(problem, completion, evaluation_config)
        )
    if not evaluations:
        raise ValueError("dataset and predictions have no tasks to evaluate")
    return evaluations, summarize_evaluations(evaluations)


def _write_evaluations(
    path: Path,
    evaluations: Sequence[TaskEvaluation],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; use --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for evaluation in evaluations:
                handle.write(
                    json.dumps(evaluation.as_dict(), ensure_ascii=False)
                    + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated code on CodeConstraints."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional per-task JSONL output without prompts or completions.",
    )
    parser.add_argument(
        "--completion-key",
        help="Read completions from this field instead of auto-detecting.",
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--memory-limit-mb", type=int, default=512)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Required acknowledgement that predictions contain executable code.",
    )
    return parser


def run(args: argparse.Namespace) -> Record:
    if not args.allow_code_execution:
        raise ValueError(
            "evaluation executes model-generated Python; "
            "pass --allow-code-execution to continue"
        )
    if args.output is not None and args.output.resolve() in {
        args.dataset.resolve(),
        args.predictions.resolve(),
    }:
        raise ValueError("output must differ from dataset and predictions")

    config = EvaluationConfig(
        timeout_seconds=args.timeout,
        memory_limit_mb=args.memory_limit_mb,
    )
    evaluations, summary = evaluate_files(
        args.dataset,
        args.predictions,
        config=config,
        completion_key=args.completion_key,
        allow_partial=args.allow_partial,
    )
    if args.output is not None:
        _write_evaluations(
            args.output,
            evaluations,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
