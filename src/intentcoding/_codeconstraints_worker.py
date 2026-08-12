"""Isolated worker for executing one CodeConstraints completion."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
import os
import random
import sys
from typing import Any

RETURN_TYPES = {
    "list": list,
    "tuple": tuple,
    "set": set,
}
DATA_TYPES = {
    "integer": int,
    "float": float,
}


def _failed_checks(expected: dict[str, Any]) -> dict[str, bool]:
    return {name: False for name in expected}


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    expected = payload["expected"]
    checks = {name: True for name in expected}
    namespace = {
        "__name__": "__codeconstraints_candidate__",
        "random": random,
    }

    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                exec(
                    compile(payload["source"], "<completion>", "exec"),
                    namespace,
                )
                function = namespace["func"]
                inputs = range(1, 120) if payload["takes_n"] else (None,)

                for n in inputs:
                    answer = function(n) if payload["takes_n"] else function()

                    return_type = expected.get("returnType")
                    if (
                        return_type is not None
                        and type(answer) is not RETURN_TYPES[return_type]
                    ):
                        checks["returnType"] = False

                    data_type = expected.get("dataType")
                    if data_type is not None:
                        expected_type = DATA_TYPES[data_type]
                        if any(type(item) is not expected_type for item in answer):
                            checks["dataType"] = False

                    size_constraint = expected.get("sizeCons")
                    if size_constraint is not None:
                        predicates = {
                            "greater": lambda item: item > n,
                            "greater_or_equal": lambda item: item >= n,
                            "less": lambda item: item < n,
                            "less_or_equal": lambda item: item <= n,
                        }
                        if any(
                            not predicates[size_constraint](item)
                            for item in answer
                        ):
                            checks["sizeCons"] = False

                    length = expected.get("length")
                    if length is not None and len(answer) != length:
                        checks["length"] = False
    except BaseException as error:
        return {
            "status": "runtime_error",
            "error_type": type(error).__name__,
            "checks": _failed_checks(expected),
        }

    return {
        "status": "completed",
        "checks": checks,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    result = _evaluate(payload)
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
