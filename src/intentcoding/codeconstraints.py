"""Deterministic construction of the CodeConstraints benchmark.

This module is a release-safe refactor of the original research builder. It
preserves the prompt templates, constraint schema, split sizes, and mask fields
used by the paper while replacing global randomness and hard-coded paths with a
reproducible API and CLI.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from intentcoding.masking import MASK_MARKER

DATA_TYPES = ("integer", "float")
RETURN_TYPES = ("list", "tuple", "set")
SIZE_CONSTRAINTS = (
    "greater",
    "greater_or_equal",
    "less",
    "less_or_equal",
)

SPLIT_FILENAMES = {
    "level2_datatype": "level2_datatype_without_sys.jsonl",
    "level2_length": "level2_len_without_sys.jsonl",
    "level2_size": "level2_size_without_sys.jsonl",
    "level3": "level3_without_sys.jsonl",
    "level4": "level4_mask_all_without_sys.jsonl",
}

Record = dict[str, Any]


@dataclass(frozen=True)
class BuildConfig:
    """CodeConstraints split sizes and reproducibility settings."""

    seed: int = 42
    level2_per_subset: int = 100
    level3_size: int = 100
    level4_size: int = 100
    mask_marker: str = MASK_MARKER

    def __post_init__(self) -> None:
        for name in (
            "level2_per_subset",
            "level3_size",
            "level4_size",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not self.mask_marker:
            raise ValueError("mask_marker must not be empty")


def _split_rng(seed: int, split_name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{split_name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _data_type(rng: random.Random) -> str:
    return DATA_TYPES[rng.randint(0, len(DATA_TYPES) - 1)]


def _format_constraint(
    rng: random.Random,
    data_type: str,
) -> tuple[str, str]:
    return_type = RETURN_TYPES[rng.randint(0, len(RETURN_TYPES) - 1)]
    return f"return a {return_type} of {data_type}s.", return_type


def _length_constraint(
    rng: random.Random,
    return_type: str,
) -> tuple[str, int]:
    length = rng.randint(1, 100)
    return f"The length of the {return_type} should be {length}.", length


def _size_constraint(
    rng: random.Random,
    data_type: str,
) -> tuple[str, str]:
    relation = SIZE_CONSTRAINTS[
        rng.randint(0, len(SIZE_CONSTRAINTS) - 1)
    ]
    phrases = {
        "greater": "greater than n",
        "greater_or_equal": "greater than or equal to n",
        "less": "less than n",
        "less_or_equal": "less than or equal to n",
    }
    return f"The {data_type}s should be {phrases[relation]}.", relation


def _render_prompt(requirement: str, takes_n: bool) -> str:
    argument = "n" if takes_n else ""
    trailing = "\n\n" if takes_n else "\n"
    return (
        f"\ndef func({argument}):\n"
        f"    '''{requirement}\n"
        f"    '''{trailing}"
    )


def _masked_prompt(marker: str, takes_n: bool) -> str:
    return _render_prompt(marker, takes_n=takes_n)


def generate_level2(
    count: int,
    subset: str,
    rng: random.Random,
    mask_marker: str = MASK_MARKER,
) -> list[Record]:
    """Generate one of the three Level 2 subsets from the paper."""

    if count < 0:
        raise ValueError("count must not be negative")
    if subset not in {"datatype", "length", "size"}:
        raise ValueError("subset must be 'datatype', 'length', or 'size'")

    records: list[Record] = []
    for index in range(count):
        if subset == "datatype":
            data_type = _data_type(rng)
            prompt = _render_prompt(
                f"Return a list of {data_type}s.",
                takes_n=False,
            )
            constraints = {"dataType": data_type}
            takes_n = False
        elif subset == "length":
            length_sentence, length = _length_constraint(rng, "list")
            prompt = _render_prompt(
                f"Return a list. {length_sentence}",
                takes_n=False,
            )
            constraints = {"length": length}
            takes_n = False
        else:
            size_sentence, relation = _size_constraint(rng, "integer")
            prompt = _render_prompt(
                "Given a positive integer n, return a list of integers. "
                f"{size_sentence}",
                takes_n=True,
            )
            constraints = {"sizeCons": relation}
            takes_n = True

        records.append(
            {
                "task_id": f"level_2_task_{index}",
                "prompt": prompt,
                "prompt_mask": _masked_prompt(mask_marker, takes_n),
                "constraints": constraints,
            }
        )
    return records


def generate_level3(
    count: int,
    rng: random.Random,
    mask_marker: str = MASK_MARKER,
) -> list[Record]:
    """Generate Level 3 by sampling one of four three-constraint groups."""

    if count < 0:
        raise ValueError("count must not be negative")

    combinations = ("drl", "drs", "dls", "rls")
    records: list[Record] = []
    for index in range(count):
        combination = combinations[rng.randint(0, len(combinations) - 1)]
        data_type = _data_type(rng)
        format_sentence, return_type = _format_constraint(rng, data_type)
        requirements = [
            f"Given a positive integer n, {format_sentence}"
        ]

        constraints: Record = {}
        if combination != "dls":
            constraints["returnType"] = return_type
        if combination != "rls":
            constraints["dataType"] = data_type

        if "l" in combination:
            length_sentence, length = _length_constraint(rng, return_type)
            requirements.append(length_sentence)
            constraints["length"] = length
        if "s" in combination:
            size_sentence, relation = _size_constraint(rng, data_type)
            requirements.append(size_sentence)
            constraints["sizeCons"] = relation

        canonical_constraints = {
            key: constraints[key]
            for key in ("returnType", "dataType", "sizeCons", "length")
            if key in constraints
        }
        records.append(
            {
                "task_id": f"level_3_task_{index}",
                "prompt": _render_prompt(
                    " ".join(requirements),
                    takes_n=True,
                ),
                "prompt_mask": _masked_prompt(mask_marker, takes_n=True),
                "constraints": canonical_constraints,
            }
        )
    return records


def generate_level4(
    count: int,
    rng: random.Random,
    mask_marker: str = MASK_MARKER,
) -> list[Record]:
    """Generate Level 4 and its full and fine-grained intent masks."""

    if count < 0:
        raise ValueError("count must not be negative")

    records: list[Record] = []
    for index in range(count):
        data_type = _data_type(rng)
        format_sentence, return_type = _format_constraint(rng, data_type)
        size_sentence, relation = _size_constraint(rng, data_type)
        length_sentence, length = _length_constraint(rng, return_type)
        prefix = f"Given a positive integer n, {format_sentence}"

        records.append(
            {
                "task_id": f"level_4_task_{index}",
                "prompt": _render_prompt(
                    f"{prefix} {size_sentence} {length_sentence}",
                    takes_n=True,
                ),
                "prompt_mask": _masked_prompt(
                    mask_marker,
                    takes_n=True,
                ),
                "prompt_mask_size": _render_prompt(
                    f"{prefix} {mask_marker} {length_sentence}",
                    takes_n=True,
                ),
                "prompt_mask_len": _render_prompt(
                    f"{prefix} {size_sentence} {mask_marker}",
                    takes_n=True,
                ),
                "prompt_mask_sizeandlen": _render_prompt(
                    f"{prefix} {mask_marker}",
                    takes_n=True,
                ),
                "constraints": {
                    "returnType": return_type,
                    "dataType": data_type,
                    "sizeCons": relation,
                    "length": length,
                },
            }
        )
    return records


def build_splits(config: BuildConfig | None = None) -> dict[str, list[Record]]:
    """Build all five paper splits without writing them to disk."""

    config = config or BuildConfig()
    return {
        "level2_datatype": generate_level2(
            config.level2_per_subset,
            "datatype",
            _split_rng(config.seed, "level2_datatype"),
            config.mask_marker,
        ),
        "level2_length": generate_level2(
            config.level2_per_subset,
            "length",
            _split_rng(config.seed, "level2_length"),
            config.mask_marker,
        ),
        "level2_size": generate_level2(
            config.level2_per_subset,
            "size",
            _split_rng(config.seed, "level2_size"),
            config.mask_marker,
        ),
        "level3": generate_level3(
            config.level3_size,
            _split_rng(config.seed, "level3"),
            config.mask_marker,
        ),
        "level4": generate_level4(
            config.level4_size,
            _split_rng(config.seed, "level4"),
            config.mask_marker,
        ),
    }


def _stage_jsonl(path: Path, records: Sequence[Record]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        committed = True
        return temporary
    finally:
        if not committed:
            temporary.unlink(missing_ok=True)


def write_splits(
    output_dir: Path,
    splits: dict[str, list[Record]],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write generated splits to ``output_dir``."""

    unknown = set(splits) - set(SPLIT_FILENAMES)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        split: output_dir / SPLIT_FILENAMES[split]
        for split in splits
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing dataset files: {names}"
        )

    staged: dict[str, Path] = {}
    try:
        for split, records in splits.items():
            staged[split] = _stage_jsonl(targets[split], records)
        for split, temporary in staged.items():
            temporary.replace(targets[split])
        return targets
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the CodeConstraints benchmark JSONL splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/codeconstraints"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level2-per-subset", type=int, default=100)
    parser.add_argument("--level3-size", type=int, default=100)
    parser.add_argument("--level4-size", type=int, default=100)
    parser.add_argument("--mask-marker", default=MASK_MARKER)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = BuildConfig(
        seed=args.seed,
        level2_per_subset=args.level2_per_subset,
        level3_size=args.level3_size,
        level4_size=args.level4_size,
        mask_marker=args.mask_marker,
    )
    splits = build_splits(config)
    targets = write_splits(
        args.output_dir,
        splits,
        overwrite=args.overwrite,
    )
    for split, path in targets.items():
        print(f"{split}: {len(splits[split])} records -> {path}")


if __name__ == "__main__":
    main()

